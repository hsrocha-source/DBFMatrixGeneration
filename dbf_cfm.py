# -*- coding: utf-8 -*-
"""First DBF matrix: conjugate-field-matched (CFM) receive beams from the CST embedded element
patterns (EEPs) and physical optics on the shaped reflector. Runs on the GPU when JAX has one.

Pipeline (docs/dbf_matrix_procedure.md):
  1. Read the 17 x 2 EEPs (.ffs) per frequency and normalize them to 1 W stimulated power
     (realized patterns). Check the field convention (peak / RMS), the phase reference and the
     time convention.
  2. De-embed every EEP to its own element centre and rotate CST -> reflector frame,
     (x, y, z)_CST -> (x, -y, -z): the array faces the reflector (-z) and H stays along +x.
  3. Give each of the 480 flight-array elements the EEP of its class (cst/port_map.md).
  4. Transmit-mode PO: element field on the reflector -> PO current J = 2 n x H -> secondary far
     field at every beam direction -> steering matrix A (beams x elements), per polarization.
  5. CFM weights w_k = conj(a_k) / |a_k| (unit norm); beam gain (2 pi / eta) |a_k|^2, which
     assumes uncorrelated, equal receiver noise.
The S-parameters only feed a power-balance report; they do not enter the weights.

Usage
-----
    python dbf_cfm.py --design designs/design_fD0.38.json --ffs-dir EEP_ffs \\
        --touchstone model.s160p --out dbf_out

The position of the simulated array in CST and the time convention are estimated from the
patterns (each element's phase centre against the port-map layout); override them with
--cst-array-center X Y Z and --time-convention.

Outputs in --out: dbf_cfm_<f>MHz.npz per frequency, summary.txt, PNG plots.
"""
import argparse
import glob
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import reflector_go as G  # noqa: E402
from cst_io import ETA0, read_ffs, read_port_map, read_touchstone, port_from_filename  # noqa: E402
from reflector_po import po_surface_grid, wrap_phase  # noqa: E402

ROT = np.array([1.0, -1.0, -1.0])     # CST -> reflector frame: 180 deg about x (its own inverse)
HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================ element classes
def classify(i, j_cst, nx, ny):
    """EEP class of element (i, j_cst) of an nx x ny array, indices in CST orientation
    (i along +x from the left, j_cst along +y from the bottom)."""
    left, right, bottom, top = i == 0, i == nx - 1, j_cst == 0, j_cst == ny - 1
    if left or right or bottom or top:
        if (left or right) and (bottom or top):
            return 'corner_' + ('B' if bottom else 'T') + ('L' if left else 'R')
        return 'edge_' + ('B' if bottom else 'T' if top else 'L' if left else 'R')
    left, right, bottom, top = i == 1, i == nx - 2, j_cst == 1, j_cst == ny - 2
    if left or right or bottom or top:
        if (left or right) and (bottom or top):
            return 'n1_corner_' + ('B' if bottom else 'T') + ('L' if left else 'R')
        return 'n1_edge_' + ('B' if bottom else 'T' if top else 'L' if left else 'R')
    return 'centre'


def flight_array_classes():
    """Class per flight-array element n = j * N_ELEM_X + i (reflector frame). The CST -> reflector
    rotation flips y, so reflector row j is CST-orientation row N_ELEM_Y - 1 - j."""
    return [classify(i, G.N_ELEM_Y - 1 - j, G.N_ELEM_X, G.N_ELEM_Y)
            for j in range(G.N_ELEM_Y) for i in range(G.N_ELEM_X)]


def cst_element_centre(col, row, n_cols, n_rows, centre):
    """CST-frame centre of model element (col, row), 1-based, for an array centred at `centre`."""
    return np.asarray(centre, float) + np.array([(col - (n_cols + 1) / 2) * G.ELEMENT_PITCH,
                                                 (row - (n_rows + 1) / 2) * G.ELEMENT_PITCH, 0.0])


# ============================================================================ EEP files
class EEPIndex:
    """Index of the .ffs files by (port, frequency); fields are loaded on demand."""

    def __init__(self, ffs_dir, ports):
        self.files = {}                                     # (port, f_khz) -> (path, header)
        for path in sorted(glob.glob(os.path.join(ffs_dir, '*.ffs'))):
            port = port_from_filename(path)
            if port not in ports:
                continue
            hdr = read_ffs(path, header_only=True)
            for f in hdr.freqs:
                self.files[(port, int(round(f / 1e3)))] = (path, hdr)
        self._cache = {}

    def freqs(self, ports):
        sets = [{fk for (p, fk) in self.files if p == port} for port in ports]
        missing = [p for p, s in zip(ports, sets) if not s]
        if missing:
            raise FileNotFoundError(f"no .ffs files for ports {missing}")
        return np.array(sorted(set.intersection(*sets))) * 1e3

    def header(self, port, f_hz):
        return self.files[(port, int(round(f_hz / 1e3)))][1]

    def get(self, port, f_hz):
        """(FarfieldSource, frequency index) for a port at a frequency."""
        key = (port, int(round(f_hz / 1e3)))
        if key not in self.files:
            raise FileNotFoundError(f"no .ffs file for port {port} at {f_hz / 1e6:.3f} MHz")
        path = self.files[key][0]
        if path not in self._cache:
            if len(self._cache) >= 80:             # about two frequencies' worth of files
                self._cache.clear()
            self._cache[path] = read_ffs(path)
        src = self._cache[path]
        return src, src.freq_index(f_hz)


def field_cartesian(src, fi):
    """r*E as Cartesian components in the CST frame on the file grid (n_theta, n_phi, 3), and
    the grid direction vectors."""
    xl = src.x_axis / np.linalg.norm(src.x_axis)
    zl = src.z_axis / np.linalg.norm(src.z_axis)
    yl = np.cross(zl, xl)
    th, ph = np.meshgrid(src.theta, src.phi, indexing='ij')
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    r_hat = (st * cp)[..., None] * xl + (st * sp)[..., None] * yl + ct[..., None] * zl
    t_hat = (ct * cp)[..., None] * xl + (ct * sp)[..., None] * yl - st[..., None] * zl
    p_hat = -sp[..., None] * xl + cp[..., None] * yl
    e = src.e_theta[fi][..., None] * t_hat + src.e_phi[fi][..., None] * p_hat
    return e, r_hat, (xl, yl, zl)


def power_convention(eeps, ports, f_hz, say):
    """Peak or RMS phasors in the files: compare the sphere integral with the header P_rad."""
    ratios = []
    for port in ports:
        src, fi = eeps.get(port, f_hz)
        ratios.append(src.integrated_power(fi) / src.p_rad[fi])
    ratios = np.array(ratios)
    med = np.nanmedian(ratios)
    if np.isnan(med):
        say("  field convention: grid does not cover the full sphere, cannot check; assuming peak phasors")
        return 1.0
    say(f"  sphere integral / header P_rad: median {med:.3f}, range {np.nanmin(ratios):.3f} ... {np.nanmax(ratios):.3f}")
    if abs(med - 1.0) < 0.1:
        say("  -> peak phasors (as expected)")
        return 1.0
    if abs(med - 0.5) < 0.05:
        say("  -> RMS phasors; fields scaled by sqrt(2) to peak")
        return np.sqrt(2.0)
    say("  [WARNING] neither peak (1) nor RMS (0.5): check the export. Proceeding without rescaling.")
    return 1.0


def roughness(e, r_hat, k, delta, mask):
    """Relative point-to-point variation of the field after removing the phase of a source
    displaced by delta. Smallest when delta is the true displacement (and convention)."""
    ee = e * np.exp(-1j * k * (r_hat @ delta))[..., None]
    d_t = np.abs(np.diff(ee, axis=0))**2
    d_p = np.abs(ee - np.roll(ee, 1, axis=1))**2
    m_t, m_p = mask[1:] & mask[:-1], mask
    return (d_t[m_t].sum() + d_p[m_p].sum()) / (np.abs(ee[mask])**2).sum()


def locate_array(eeps, classes, layout, f_hz, centre, z, mode, say):
    """Time convention and CST position of the simulated array, from the patterns themselves.

    Each EEP's phase carries its element's position: the displacement that makes the de-embedded
    field smoothest is the element's apparent phase centre. Under the right time convention the
    phase centres follow the known layout, so (phase centre - layout offset) is the same point for
    every element: the array centre. Under the wrong convention they come out point-mirrored and
    that difference scatters by the size of the array.

    layout: (C, 3) element offsets from the array centre in the CST frame. centre: None to use
    the estimate, or the (x, y, z) to use. z: array-plane z for the estimate. mode: 'auto',
    '+jwt' or '-jwt'. Returns (conj, centre_used)."""
    from scipy.optimize import minimize
    k = 2 * np.pi * f_hz / G.C0
    guess = np.array([0.0, 0.0, z]) if centre is None else np.asarray(centre, float)
    data = []
    for cls in classes:
        src, fi = eeps.get(cls['port_H'], f_hz)
        e, r_hat, _ = field_cartesian(src, fi)
        rows = src.theta <= np.radians(72.0)                   # the cone the reflector subtends
        data.append((e[rows], r_hat[rows], src.position))

    def implied_centres(conj):
        est = []
        for c, (e, r_hat, pos) in enumerate(data):
            ee = np.conj(e) if conj else e
            mask = np.ones(ee.shape[:2], bool)
            d0 = guess + layout[c] - pos                       # displacement from the file's phase reference
            res = minimize(lambda v: roughness(ee, r_hat, k, np.array([v[0], v[1], d0[2]]), mask), d0[:2],
                           method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-10, maxiter=600))
            est.append(res.x + pos[:2] - layout[c][:2])
        est = np.array(est)
        xy = np.median(est, axis=0)
        return xy, float(np.median(np.linalg.norm(est - xy, axis=1))), est

    hyps = ('+jwt', '-jwt') if mode == 'auto' else (mode,)
    fits = {h: implied_centres(h == '-jwt') for h in hyps}
    best = min(fits, key=lambda h: fits[h][1])
    xy, spread, est = fits[best]
    if spread > G.ELEMENT_PITCH:
        raise ValueError(
            "the phase centres of the simulated elements do not follow the 10 x 8 layout of the port map"
            + (" under either time convention" if mode == 'auto' else f" with {mode}")
            + f" (median scatter {1e3 * spread:.0f} mm). Check the port map orientation (which corner is"
            " port 1, H along x) and the units of the .ffs position.")
    if mode == 'auto':
        other = fits['-jwt' if best == '+jwt' else '+jwt'][1]
        say(f"  time convention: e^{{{best[0]}jwt}} (element phase centres follow the layout to {1e3 * spread:.1f} mm; "
            f"{1e3 * other:.0f} mm with the opposite sign)" + ("; fields are conjugated" if best == '-jwt' else ""))
    else:
        say(f"  time convention: {mode} (set on the command line)")

    estimate = np.array([xy[0], xy[1], guess[2]])
    if centre is None:
        used = estimate
        say(f"  CST array centre estimated from the patterns: x {estimate[0]:+.4f} m, y {estimate[1]:+.4f} m "
            f"(z {estimate[2]:+.4f} m as given)")
    else:
        used = guess
        say(f"  CST array centre as given: {used.round(4).tolist()} m; estimate from the patterns "
            f"x {estimate[0]:+.4f}, y {estimate[1]:+.4f} m")
        if np.hypot(*(estimate[:2] - used[:2])) > 0.5 * G.ELEMENT_PITCH:
            say("  [WARNING] the given centre is more than half a pitch from the estimate: the beams will be"
                " squinted and lose gain. Use --cst-array-center auto unless you are sure.")
    say(f"  apparent phase centre minus element centre (H ports, {f_hz / 1e6:.0f} MHz):")
    for cls, (dx, dy) in zip(classes, (est - used[:2]) * 1e3):
        say(f"    {cls['eep_class']:<14} dx {dx:7.1f} mm  dy {dy:7.1f} mm")
    return best == '-jwt', used


def pattern_table(eeps, classes, cst_pos, f_hz, conj, field_scale):
    """Gather table of the de-embedded, normalized EEPs for H and V ports.
    Returns (table_H, table_V) each (n_class * n_theta * n_phi, 3) complex64 in CST Cartesian
    components, the static grid tuple, and the local frame axes."""
    k = 2 * np.pi * f_hz / G.C0
    tables, ref = {'H': [], 'V': []}, None
    for c, cls in enumerate(classes):
        for pol in ('H', 'V'):
            src, fi = eeps.get(cls[f'port_{pol}'], f_hz)
            if ref is None:
                ref = src
            elif (src.theta.shape != ref.theta.shape or src.phi.shape != ref.phi.shape
                  or not np.allclose(src.theta, ref.theta) or not np.allclose(src.phi, ref.phi)):
                raise ValueError(f"{src.path}: angular grid differs from {ref.path}")
            elif not (np.allclose(src.x_axis, ref.x_axis) and np.allclose(src.z_axis, ref.z_axis)):
                raise ValueError(f"{src.path}: local frame differs from {ref.path}")
            e, r_hat, axes = field_cartesian(src, fi)
            if conj:
                e = np.conj(e)
            e = e * field_scale / np.sqrt(src.p_stim[fi])                 # realized, 1 W stimulated
            delta = cst_pos[c] - src.position
            e = e * np.exp(-1j * k * (r_hat @ delta))[..., None]          # phase reference -> element centre
            tables[pol].append(e)
    th, ph = ref.theta, ref.phi
    grid = (float(th[0]), float(th[1] - th[0]), len(th), float(ph[0]), float(ph[1] - ph[0]), len(ph))
    to_table = lambda lst: jnp.asarray(np.stack(lst).reshape(-1, 3), dtype=jnp.complex64)
    return to_table(tables['H']), to_table(tables['V']), grid, tuple(jnp.asarray(a, jnp.float32) for a in axes)


# ============================================================================ PO steering matrix
_slopes = jax.vmap(jax.grad(G.reflector_surface, argnums=(0, 1)), in_axes=(0, 0, None, None))


def _interp(table, base, th, ph, grid):
    """Bilinear interpolation of the gathered pattern table at (theta, phi) [rad]."""
    th0, dth, nth, ph0, dph, nph = grid
    ft = jnp.clip((th - th0) / dth, 0.0, nth - 1 - 1e-3)
    it = jnp.floor(ft).astype(jnp.int32)
    tt = (ft - it)[..., None]
    fp = jnp.mod(ph - ph0, 2 * jnp.pi) / dph
    ip = jnp.floor(fp).astype(jnp.int32)
    tp = (fp - ip)[..., None]
    ip, ip1 = ip % nph, (ip + 1) % nph
    r0 = base + it * nph
    r1 = r0 + nph
    return ((1 - tt) * ((1 - tp) * table[r0 + ip] + tp * table[r0 + ip1])
            + tt * ((1 - tp) * table[r1 + ip] + tp * table[r1 + ip1]))


@partial(jax.jit, static_argnames=('grid',))
def _surface_currents(params, F, k0, x, y, w, elem_pos, base, tab_H, tab_V, axes, grid):
    """PO currents J dS (ns, 2N, 3) of every element (H ports, then V ports) on a surface chunk."""
    z = G.reflector_surface(x, y, params, F)
    zx, zy = _slopes(x, y, params, F)
    p = jnp.stack([x, y, z], axis=-1)
    ndS = jnp.stack([-zx, -zy, jnp.ones_like(zx)], axis=-1) * w[:, None]
    Rv = p[:, None, :] - elem_pos[None]
    R = jnp.sqrt(jnp.sum(Rv * Rv, axis=-1))
    Rh = Rv / R[..., None]
    lit = (jnp.sum(ndS[:, None, :] * Rh, axis=-1) < 0.0).astype(p.dtype)
    rc = Rh * jnp.asarray(ROT, p.dtype)                            # direction in the CST frame
    xl, yl, zl = axes
    th = jnp.arccos(jnp.clip(jnp.sum(rc * zl, axis=-1), -1.0, 1.0))
    ph = jnp.arctan2(jnp.sum(rc * yl, axis=-1), jnp.sum(rc * xl, axis=-1))
    g = (jnp.exp(-1j * wrap_phase(k0 * R)) / R * lit)[..., None]
    J = []
    for tab in (tab_H, tab_V):
        E = _interp(tab, base, th, ph, grid) * jnp.asarray(ROT, p.dtype) * g     # reflector frame
        H = jnp.cross(Rh, E) / ETA0
        J.append(2.0 * jnp.cross(ndS[:, None, :], H))
    return p, jnp.concatenate(J, axis=1)


@partial(jax.jit, static_argnames=('blockage',))
def _radiate(p, J, k0, u_hat, pol_x, pol_y, frame, blockage):
    """Secondary far field of the chunk's currents, projected on pol_x and pol_y: (2 Mb, 2N)."""
    phase = jnp.exp(1j * wrap_phase(k0 * jnp.sum(u_hat[:, None, :] * p[None], axis=-1)))   # (Mb, ns)
    if blockage:
        n_arr, u_vec, v_vec = frame
        t = -G._dot3(p, n_arr)[None, :] / G._dot3(u_hat, n_arr)[:, None]
        hit = p[None] + t[..., None] * u_hat[:, None, :]
        blocked = ((jnp.abs(G._dot3(hit, u_vec)) < G.ARRAY_SIZE_X / 2)
                   & (jnp.abs(G._dot3(hit, v_vec)) < G.ARRAY_SIZE_Y / 2) & (t > 0))
        phase = phase * (~blocked)
    P = jnp.concatenate([pol_x[:, None, :] * phase[..., None], pol_y[:, None, :] * phase[..., None]], axis=0)
    ns = p.shape[0]
    A = P.reshape(P.shape[0], ns * 3) @ J.transpose(0, 2, 1).reshape(ns * 3, -1)
    return A * (-1j * k0 * ETA0 / (4.0 * jnp.pi))


def pol_vectors(d_in):
    """x- and y-like polarization vectors for propagation directions d_in (..., 3), as in reflector_po."""
    ex = np.array([1.0, 0.0, 0.0])
    px = ex - (d_in @ ex)[..., None] * d_in
    px /= np.linalg.norm(px, axis=-1, keepdims=True)
    return px, np.cross(px, d_in)


def steering_matrices(params, F, f_hz, d_in, elem_pos, elem_cls, tab_H, tab_V, grid, axes,
                      samples_per_lam=4, chunk=2048, beam_block=1024, blockage=True, say=print):
    """Co- and cross-pol steering matrices (M x N) for H and V ports at the directions d_in (M, 3).
    A[m, n] = p_hat(m) . r E_sec,n(u_m), u_m = -d_in[m], in V for 1 W stimulated power."""
    d_in = np.asarray(d_in, float)
    M, N = len(d_in), len(elem_pos)
    px, py = pol_vectors(d_in)
    Mb = min(beam_block, M)
    n_blk = -(-M // Mb)
    pad = n_blk * Mb - M
    f32 = lambda a: jnp.asarray(np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]), jnp.float32)
    u_hat, px, py = f32(-d_in), f32(px), f32(py)

    k0 = jnp.float32(2 * np.pi * f_hz / G.C0)
    xs, ys, ws, _, n_real = po_surface_grid(f_hz, samples_per_lam, chunk)
    params32 = {kk: jnp.float32(v) for kk, v in params.items()}
    F32 = jnp.float32(F)
    frame = G.array_frame(params32, F32)
    base = jnp.asarray(np.asarray(elem_cls) * grid[2] * grid[5], jnp.int32)[None, :]
    pos = jnp.asarray(elem_pos, jnp.float32)

    acc = [jnp.zeros((2 * Mb, 2 * N), jnp.complex64) for _ in range(n_blk)]
    with jax.default_matmul_precision("highest"):
        for c in range(xs.shape[0]):
            p, J = _surface_currents(params32, F32, k0, xs[c], ys[c], ws[c], pos, base, tab_H, tab_V, axes, grid)
            for b in range(n_blk):
                sl = slice(b * Mb, (b + 1) * Mb)
                acc[b] = acc[b] + _radiate(p, J, k0, u_hat[sl], px[sl], py[sl], frame, blockage)
            if (c + 1) % 10 == 0 or c + 1 == xs.shape[0]:
                say(f"    PO: surface chunk {c + 1}/{xs.shape[0]} ({n_real} samples, {M} directions)")
    acc = [np.asarray(a) for a in acc]
    Ax = np.concatenate([a[:Mb] for a in acc])[:M]
    Ay = np.concatenate([a[Mb:] for a in acc])[:M]
    return dict(H_co=Ax[:, :N], H_x=Ay[:, :N], V_co=Ay[:, N:], V_x=Ax[:, N:])


# ============================================================================ CFM
def cfm_weights(A_co):
    """Unit-norm CFM weights W (N x K), gain [dBi], and the number of elements holding 99 % of
    the beam power."""
    a2 = np.sum(np.abs(A_co)**2, axis=1)
    W = (np.conj(A_co) / np.sqrt(a2)[:, None]).T
    gain = 10 * np.log10(2 * np.pi / ETA0 * a2)
    srt = np.sort(np.abs(A_co)**2, axis=1)[:, ::-1]
    n99 = 1 + np.argmax(np.cumsum(srt, axis=1) >= 0.99 * a2[:, None], axis=1)
    return W.astype(np.complex64), gain, n99


def cfm(A_co, A_x):
    """cfm_weights plus the cross-pol / co-pol response of each beam [dB]."""
    W, gain, n99 = cfm_weights(A_co)
    a2 = np.sum(np.abs(A_co)**2, axis=1)
    xpol = 10 * np.log10(np.abs(np.sum(A_x * W.T, axis=1))**2 / a2 + 1e-30)
    return W, gain, xpol, n99


def beam_grid(az_max, az_step, el_max, el_step):
    az = np.round(np.arange(-az_max, az_max + 1e-9, az_step), 6)
    el = np.round(np.arange(-el_max, el_max + 1e-9, el_step), 6)
    EL, AZ = np.meshgrid(el, az, indexing='ij')                  # beam k = i_el * n_az + i_az
    return az, el, AZ.ravel(), EL.ravel()


def nearest_beam(beam_az, beam_el, az, el):
    return int(np.argmin((beam_az - az)**2 + (beam_el - el)**2))


# ============================================================================ S-parameter report
def power_balance(eeps, classes, freqs, ts, say):
    n = ts.s.shape[1]
    say(f"\n--- Power balance vs S-parameters ({os.path.basename(ts.path)}, {n} ports, Z0 = {ts.z0:g} ohm) ---")
    say("  P_acc/P_stim vs 1 - |S_nn|^2 and P_rad/P_stim vs 1 - sum_m |S_mn|^2 (difference = loss or inconsistency)")
    worst_acc, worst_rad = 0.0, 0.0
    f_mid = freqs[np.argmin(np.abs(freqs - 1300e6))]
    for f in freqs:
        s, df = ts.at(f)
        if abs(df) > 1e6:
            say(f"  [WARNING] nearest S-parameter point to {f / 1e6:.3f} MHz is {df / 1e6:+.3f} MHz away")
        for cls in classes:
            for pol in ('H', 'V'):
                port = cls[f'port_{pol}']
                if port > n:
                    continue
                h = eeps.header(port, f)
                fi = int(np.argmin(np.abs(h.freqs - f)))
                acc, rad = h.p_acc[fi] / h.p_stim[fi], h.p_rad[fi] / h.p_stim[fi]
                refl, tot = 1 - np.abs(s[port - 1, port - 1])**2, 1 - np.sum(np.abs(s[:, port - 1])**2)
                worst_acc, worst_rad = max(worst_acc, abs(acc - refl)), max(worst_rad, abs(rad - tot))
                if f == f_mid and pol == 'H':
                    say(f"  {f / 1e6:7.1f} MHz {cls['eep_class']:<14} port {port:3d}: acc {acc:.3f} vs {refl:.3f} | "
                        f"rad {rad:.3f} vs {tot:.3f}")
    say(f"  worst |difference| over all ports and frequencies: accepted {worst_acc:.3f}, radiated {worst_rad:.3f}")


# ============================================================================ plots
def plot_frequency(out, f_mhz, az, el, res, beam_az, beam_el, cuts):
    for pol in ('H', 'V'):
        g = res[pol]['gain'].reshape(len(el), len(az))
        fig, ax = plt.subplots(figsize=(4.5, 6))
        im = ax.imshow(g, origin='lower', aspect='auto', extent=[az[0], az[-1], el[0], el[-1]], cmap='viridis')
        fig.colorbar(im, ax=ax, label='CFM beam gain [dBi]')
        ax.set_xlabel('az [deg]')
        ax.set_ylabel('el [deg]')
        ax.set_title(f'{pol} pol, {f_mhz:.0f} MHz')
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'gain_map_{pol}_{f_mhz:.0f}MHz.png'), dpi=120)
        plt.close(fig)

    picks = [(0.0, 0.0), (0.0, el[-1]), (az[-1], el[-1])]
    fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 7), sharey=True)
    for ax, (a, e) in zip(axes, picks):
        k = nearest_beam(beam_az, beam_el, a, e)
        w = np.abs(res['H']['W'][:, k]).reshape(G.N_ELEM_Y, G.N_ELEM_X)
        im = ax.imshow(20 * np.log10(w / w.max() + 1e-12), origin='lower', vmin=-30, vmax=0, cmap='viridis',
                       extent=[-G.ARRAY_SIZE_X / 2, G.ARRAY_SIZE_X / 2, -G.ARRAY_SIZE_Y / 2, G.ARRAY_SIZE_Y / 2])
        ax.set_title(f'H, az {beam_az[k]:g}, el {beam_el[k]:g}\n{res["H"]["n99"][k]} el. for 99 %', fontsize=9)
        ax.set_xticks([-0.4, 0.0, 0.4])
        ax.set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    fig.colorbar(im, ax=list(axes), label='|w| [dB]', shrink=0.6)
    fig.savefig(os.path.join(out, f'weights_H_{f_mhz:.0f}MHz.png'), dpi=120)
    plt.close(fig)

    if cuts is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, xlabel, sel in ((axes[0], 'el', 'el [deg]', lambda: [(0.0, e) for e in np.linspace(el[0], el[-1], 5)]),
                                 (axes[1], 'az', 'az [deg]', lambda: [(a, 0.0) for a in np.linspace(az[0], az[-1], 5)])):
        ang, A_cut = cuts[key]
        for a, e in sel():
            k = nearest_beam(beam_az, beam_el, a, e)
            ax.plot(ang, 10 * np.log10(2 * np.pi / ETA0 * np.abs(A_cut @ res['H']['W'][:, k])**2 + 1e-30),
                    label=f'beam az {beam_az[k]:g}, el {beam_el[k]:g}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('gain [dBi]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        top = max(line.get_ydata().max() for line in ax.get_lines())
        ax.set_ylim(top - 40, top + 2)
    axes[0].set_title(f'H pol, elevation cut at az = 0, {f_mhz:.0f} MHz')
    axes[1].set_title(f'H pol, azimuth cut at el = 0, {f_mhz:.0f} MHz')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'cuts_H_{f_mhz:.0f}MHz.png'), dpi=120)
    plt.close(fig)


def plot_gain_vs_el(out, freqs, az, el, results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pol in zip(axes, ('H', 'V')):
        for f in freqs:
            g = results[f][pol]['gain'].reshape(len(el), len(az))
            line, = ax.plot(el, g[:, np.argmin(np.abs(az))], label=f'{f / 1e6:.0f} MHz, az 0')
            ax.plot(el, g[:, -1], ls='--', color=line.get_color(), lw=0.8)
        ax.set_title(f'{pol} pol (dashed: az = {az[-1]:g})')
        ax.set_xlabel('el [deg]')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('CFM beam gain [dBi]')
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'gain_vs_el.png'), dpi=120)
    plt.close(fig)


# ============================================================================ sub-band interpolation
def load_frequency_results(out_dir):
    """Steering vectors of a previous run: freqs (n_f,), {pol: (n_f, K, N)}, and beam / element metadata."""
    files = glob.glob(os.path.join(out_dir, 'dbf_cfm_*MHz.npz'))
    if not files:
        raise FileNotFoundError(f"no dbf_cfm_*MHz.npz files in {out_dir}")
    data = sorted((np.load(p) for p in files), key=lambda d: float(d['freq_hz']))
    ref = data[0]
    for d in data[1:]:
        if not (np.array_equal(d['beam_az'], ref['beam_az']) and np.array_equal(d['beam_el'], ref['beam_el'])):
            raise ValueError(f"{out_dir}: the per-frequency files use different beam grids")
    freqs = np.array([float(d['freq_hz']) for d in data])
    A = {pol: np.stack([d[f'A_{pol}'] for d in data]) for pol in ('H', 'V')}
    meta = {k: ref[k] for k in ('beam_az', 'beam_el', 'az_grid', 'el_grid', 'element_xyz', 'element_class')}
    meta['F'] = float(ref['F']) if 'F' in ref.files else G.F_OVER_D * G.D
    return freqs, A, meta


def lagrange(freqs, f, n=4):
    """Indices and coefficients of local polynomial interpolation at f from the n nearest
    samples (cubic for n = 4; exact at the samples)."""
    idx = np.sort(np.argsort(np.abs(freqs - f), kind='stable')[:min(n, len(freqs))])
    x = freqs[idx]
    c = np.array([np.prod([(f - x[m]) / (x[j] - x[m]) for m in range(len(x)) if m != j]) for j in range(len(x))])
    return idx, c


def remove_bulk_delay(A, freqs, F):
    """Per-beam bulk delay tau_k = 2F/c + a residual estimated from the mean phase step between
    neighbouring frequencies, and the steering vectors with that delay removed. The delay is common
    to all elements of a beam, so it does not change the beam; removing it makes the remaining
    phase slow enough to interpolate between 15 MHz samples."""
    tau0 = 2 * F / G.C0
    Ad = A * np.exp(2j * np.pi * freqs * tau0).astype(np.complex64)[:, None, None]
    dtau = np.zeros(A.shape[1])
    if len(freqs) > 1:
        s = np.sum(Ad[1:] * np.conj(Ad[:-1]), axis=2)                            # (n_f - 1, K)
        dtau = -np.mean(np.angle(s) / (2 * np.pi * np.diff(freqs))[:, None], axis=0)
        Ad = Ad * np.exp(2j * np.pi * freqs[:, None] * dtau[None, :]).astype(np.complex64)[..., None]
    return tau0 + dtau, Ad


def interpolation_diagnostics(Ad, freqs):
    """For de-delayed steering vectors Ad (n_f, K, N):
      * max_step_deg: largest phase step between neighbouring frequencies of an element within
        10 dB of the beam's strongest; must stay well below 180 deg for interpolation to work.
      * frac_big: largest share of a beam's power in elements whose phase steps by more than
        90 deg. Weak elements swept by a null of the focal pattern flip phase; that is harmless
        while their share is small.
      * width_mhz (K,): sub-band width for <= 0.1 dB CFM loss at the sub-band edges, from the
        power-weighted phase slope of the weights.
      * loo_db: worst CFM loss per interior frequency when it is interpolated from the others."""
    p = np.abs(Ad[len(freqs) // 2])**2
    w = p / p.sum(axis=1, keepdims=True)
    strong = p >= 0.1 * p.max(axis=1, keepdims=True)
    step = np.angle(Ad[1:] * np.conj(Ad[:-1]))                                   # (n_f - 1, K, N)
    max_step = float(np.degrees(np.max(np.abs(step) * strong[None])))
    frac_big = float(np.max(np.sum(w[None] * (np.abs(step) > np.pi / 2), axis=2)))
    slope = np.sqrt(np.max(np.sum(w[None] * (step / (np.diff(freqs) / 1e6)[:, None, None])**2, axis=2), axis=0))
    phi_edge = np.sqrt(0.1 / (10 / np.log(10)))                                  # 0.1 dB ~ 4.34 phi^2
    width = 2 * phi_edge / np.maximum(slope, 1e-12)
    loo = []
    for i in range(1, len(freqs) - 1):
        others = np.delete(np.arange(len(freqs)), i)
        idx, c = lagrange(freqs[others], freqs[i])
        a = np.tensordot(c.astype(np.complex64), Ad[others[idx]], axes=1)
        eta = np.abs(np.sum(Ad[i] * np.conj(a), axis=1))**2 / (np.sum(np.abs(Ad[i])**2, axis=1) * np.sum(np.abs(a)**2, axis=1))
        loo.append(float(np.max(-10 * np.log10(eta))))
    return dict(max_step_deg=max_step, frac_big=frac_big, width_mhz=width, loo_db=loo)


def write_subbands(out_dir, step_mhz=None, centres_mhz=None, say=print):
    """CFM weights at sub-band centres, interpolated from the per-frequency steering vectors in
    out_dir. Writes out_dir/subbands/subband_<f>MHz.npz, a summary and a gain-vs-frequency plot."""
    freqs, A, meta = load_frequency_results(out_dir)
    if centres_mhz:
        f_out = np.array(centres_mhz, float) * 1e6
    elif step_mhz:
        f_out = np.arange(freqs[0], freqs[-1] + 1e3, step_mhz * 1e6)
    else:
        raise ValueError("give a sub-band step or the sub-band centres")
    if f_out.min() < freqs[0] - 1e3 or f_out.max() > freqs[-1] + 1e3:
        raise ValueError(f"sub-band centres must lie within the computed band "
                         f"{freqs[0] / 1e6:g} ... {freqs[-1] / 1e6:g} MHz (no extrapolation)")
    sub = os.path.join(out_dir, 'subbands')
    os.makedirs(sub, exist_ok=True)
    log = open(os.path.join(sub, 'summary.txt'), 'w')

    def note(*a):
        say(*a)
        print(*a, file=log, flush=True)

    K, N = A['H'].shape[1:]
    note(f"\n--- Sub-band weights: {len(f_out)} centres {f_out[0] / 1e6:g} ... {f_out[-1] / 1e6:g} MHz from "
         f"{len(freqs)} computed frequencies ({K} beams, ~{len(f_out) * K * N * 8 * 2 / 1e6:.0f} MB) ---")
    note(f"  interpolation: local cubic in frequency on the steering vectors, after removing the bulk delay "
         f"2F/c = {2 * meta['F'] / G.C0 * 1e9:.1f} ns + a per-beam residual")
    tau, Ad = {}, {}
    for pol in ('H', 'V'):
        tau[pol], Ad[pol] = remove_bulk_delay(A[pol], freqs, meta['F'])
        A[pol] = None                                                            # free memory
        resid = 1e9 * (tau[pol] - 2 * meta['F'] / G.C0)
        note(f"  {pol}: per-beam residual delay {resid.min():+.2f} ... {resid.max():+.2f} ns")
        if len(freqs) < 2:
            continue
        d = interpolation_diagnostics(Ad[pol], freqs)
        note(f"      largest phase step between neighbouring frequencies, elements within 10 dB of the strongest: "
             f"{d['max_step_deg']:.1f} deg; beam power in elements stepping > 90 deg: max {100 * d['frac_big']:.2f} %")
        if d['loo_db']:
            note(f"      leave-one-out (each interior frequency from the others): worst CFM loss "
                 f"{max(d['loo_db']):.3f} dB (pessimistic: the gap is twice the sample spacing)")
        if d['frac_big'] > 0.01 or (d['loo_db'] and max(d['loo_db']) > 0.1):
            note("      [WARNING] the frequency sampling is too coarse to interpolate reliably: "
                 "compute more frequencies or use the computed ones directly")
        note(f"      sub-band width for <= 0.1 dB loss at the sub-band edges: worst beam "
             f"{d['width_mhz'].min():.1f} MHz, median {np.median(d['width_mhz']):.1f} MHz")
    if len(f_out) > 1:
        note(f"  sub-band spacing used: {np.min(np.diff(f_out)) / 1e6:g} MHz")

    gains = {'H': [], 'V': []}
    for f in f_out:
        idx, c = lagrange(freqs, f)
        save = {}
        for pol in ('H', 'V'):
            a = np.tensordot(c.astype(np.complex64), Ad[pol][idx], axes=1)
            a = a * np.exp(-2j * np.pi * f * tau[pol]).astype(np.complex64)[:, None]   # put the delay back
            W, gain, n99 = cfm_weights(a)
            gains[pol].append(gain)
            save.update({f'W_{pol}': W, f'gain_{pol}_dBi': gain, f'n99_{pol}': n99, f'tau_bulk_{pol}_s': tau[pol]})
        np.savez(os.path.join(sub, f'subband_{f / 1e6:.3f}MHz.npz'), freq_hz=f, source_freqs_hz=freqs, **save,
                 **{k: meta[k] for k in ('beam_az', 'beam_el', 'az_grid', 'el_grid', 'element_xyz', 'element_class')})
    note(f"  wrote {len(f_out)} files to {sub}")

    beam_az, beam_el, az, el = meta['beam_az'], meta['beam_el'], meta['az_grid'], meta['el_grid']
    fig, ax = plt.subplots(figsize=(7, 4))
    for a_, e_ in ((0.0, 0.0), (0.0, el[-1]), (az[-1], el[-1])):
        k = nearest_beam(beam_az, beam_el, a_, e_)
        line, = ax.plot(f_out / 1e6, [g[k] for g in gains['H']], label=f'H, beam az {beam_az[k]:g}, el {beam_el[k]:g}')
        ax.plot(freqs / 1e6, 10 * np.log10(2 * np.pi / ETA0 * np.sum(np.abs(Ad['H'][:, k])**2, axis=1)),
                'o', color=line.get_color())
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('CFM beam gain [dBi]')
    ax.set_title('Sub-band interpolation (lines) vs computed frequencies (dots)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(sub, 'subband_gain.png'), dpi=120)
    plt.close(fig)
    log.close()


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--design', help="reflector design JSON (reflector_go.py)")
    ap.add_argument('--ffs-dir', help="folder with the CST .ffs farfield sources")
    ap.add_argument('--touchstone', help=".sNp file; only used for the power-balance report")
    ap.add_argument('--port-map', default=os.path.join(HERE, 'cst', 'port_map.csv'))
    ap.add_argument('--out', default='dbf_out')
    ap.add_argument('--cst-array-center', nargs='+', default=['auto'], metavar='auto|X Y Z',
                    help="CST coordinates [m] of the centre of the simulated array; 'auto' (default) "
                         "estimates x and y from the patterns")
    ap.add_argument('--cst-array-z', type=float, default=0.0,
                    help="CST z [m] of the array plane, used with --cst-array-center auto")
    ap.add_argument('--time-convention', choices=('auto', '+jwt', '-jwt'), default='auto')
    ap.add_argument('--freqs-mhz', type=float, nargs='*', help="subset of frequencies (default: all found)")
    ap.add_argument('--az-max', type=float, default=G.REQ_AZ_MAX)
    ap.add_argument('--az-step', type=float, default=0.5)
    ap.add_argument('--el-max', type=float, default=G.REQ_EL_MAX)
    ap.add_argument('--el-step', type=float, default=0.1)
    ap.add_argument('--samples-per-lam', type=float, default=4, help="PO surface sampling")
    ap.add_argument('--chunk', type=int, default=2048, help="surface samples per GPU chunk")
    ap.add_argument('--beam-block', type=int, default=1024, help="beam directions per GPU block")
    ap.add_argument('--no-blockage', action='store_true')
    ap.add_argument('--save-cross', action='store_true', help="also save the cross-pol steering matrices")
    ap.add_argument('--subband-step-mhz', type=float, help="also write sub-band weights at this spacing")
    ap.add_argument('--subband-centers-mhz', type=float, nargs='+', help="... or at these sub-band centres")
    ap.add_argument('--subbands-only', action='store_true',
                    help="only (re)compute sub-band weights from the per-frequency files already in --out")
    args = ap.parse_args(argv)
    if args.cst_array_center == ['auto']:
        centre_arg = None
    elif len(args.cst_array_center) == 3:
        centre_arg = [float(v) for v in args.cst_array_center]
    else:
        ap.error("--cst-array-center takes 'auto' or three numbers X Y Z")
    subbands = dict(step_mhz=args.subband_step_mhz, centres_mhz=args.subband_centers_mhz)

    if args.subbands_only:
        if not (args.subband_step_mhz or args.subband_centers_mhz):
            ap.error("--subbands-only needs --subband-step-mhz or --subband-centers-mhz")
        write_subbands(args.out, **subbands)
        return
    if not (args.design and args.ffs_dir):
        ap.error("--design and --ffs-dir are required (unless --subbands-only)")

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'summary.txt'), 'w')

    def say(*a):
        print(*a)
        print(*a, file=log, flush=True)

    say(f"JAX devices: {jax.devices()}")
    params, F = G.load_params(args.design)
    say(f"Reflector: {args.design} (D = {G.D:g} m, f/D = {F / G.D:.3f})")

    pm = read_port_map(args.port_map)
    classes = [r for r in pm if r['eep_class']]
    n_cols, n_rows = max(r['col'] for r in pm), max(r['row'] for r in pm)
    names = [c['eep_class'] for c in classes]
    layout = np.array([cst_element_centre(c['col'], c['row'], n_cols, n_rows, (0.0, 0.0, 0.0)) for c in classes])
    ports = [c[f'port_{p}'] for c in classes for p in ('H', 'V')]
    say(f"Port map: {args.port_map} ({n_cols} x {n_rows} CST model, {len(classes)} EEP classes, {len(ports)} ports)")

    elem_names = flight_array_classes()
    missing = sorted(set(elem_names) - set(names))
    if missing:
        raise ValueError(f"port map has no EEP for classes {missing}")
    elem_cls = np.array([names.index(n) for n in elem_names])
    elem_pos = np.asarray(G.element_positions(G.array_frame(params, F))[0])

    eeps = EEPIndex(args.ffs_dir, set(ports))
    freqs = eeps.freqs(ports)
    if args.freqs_mhz:
        freqs = np.array([freqs[np.argmin(np.abs(freqs - f * 1e6))] for f in args.freqs_mhz])
    f_mid = freqs[np.argmin(np.abs(freqs - 1300e6))]
    say(f"EEPs: {len(eeps.files)} (port, frequency) entries in {args.ffs_dir}; "
        f"frequencies [MHz]: {', '.join(f'{f / 1e6:g}' for f in freqs)}")

    say(f"\n--- Checks at {f_mid / 1e6:.0f} MHz ---")
    field_scale = power_convention(eeps, ports, f_mid, say)
    conj, cst_centre = locate_array(eeps, classes, layout, f_mid, centre_arg, args.cst_array_z,
                                    args.time_convention, say)
    cst_pos = layout + cst_centre
    centre = classes[names.index('centre')]
    for f in freqs:
        src, fi = eeps.get(centre['port_H'], f)
        g = 2 * np.pi / ETA0 * field_scale**2 * np.max(np.abs(src.e_theta[fi])**2 + np.abs(src.e_phi[fi])**2) / src.p_stim[fi]
        say(f"  centre element H, {f / 1e6:7.1f} MHz: max realized gain {10 * np.log10(g):5.2f} dBi, "
            f"P_rad/P_stim {src.p_rad[fi] / src.p_stim[fi]:.3f}")
    if args.touchstone:
        power_balance(eeps, classes, freqs, read_touchstone(args.touchstone), say)

    az, el, beam_az, beam_el = beam_grid(args.az_max, args.az_step, args.el_max, args.el_step)
    d_beams = np.asarray(G.incident_direction(beam_az, beam_el), float)
    K = len(beam_az)
    say(f"\nBeams: {len(az)} az x {len(el)} el = {K} (az {az[0]:g} ... {az[-1]:g}, step {args.az_step:g}; "
        f"el {el[0]:g} ... {el[-1]:g}, step {args.el_step:g}); output ~{K * 480 * 8 * 4 / 1e6:.0f} MB per frequency")
    cut_el = np.round(np.arange(-(args.el_max + 7), args.el_max + 7 + 1e-9, 0.05), 6)
    cut_az = np.round(np.arange(-(args.az_max + 5), args.az_max + 5 + 1e-9, 0.05), 6)
    d_cuts = np.concatenate([np.asarray(G.incident_direction(0.0 * cut_el, cut_el)),
                             np.asarray(G.incident_direction(cut_az, 0.0 * cut_az))])

    results = {}
    for f in freqs:
        f_mhz = f / 1e6
        lam = G.C0 / f
        g_ref = 10 * np.log10(4 * np.pi * np.pi * (G.D / 2)**2 / lam**2)
        say(f"\n=== {f_mhz:.1f} MHz (uniform-aperture directivity 4 pi A / lambda^2 = {g_ref:.2f} dBi) ===")
        tab_H, tab_V, grid, axes = pattern_table(eeps, classes, cst_pos, f, conj, field_scale)
        d_all = np.concatenate([d_beams, d_cuts]) if f == f_mid else d_beams
        A = steering_matrices(params, F, f, d_all, elem_pos, elem_cls, tab_H, tab_V, grid, axes,
                              args.samples_per_lam, args.chunk, args.beam_block, not args.no_blockage, say)
        res = {}
        for pol in ('H', 'V'):
            A_co, A_x = A[f'{pol}_co'][:K], A[f'{pol}_x'][:K]
            W, gain, xpol, n99 = cfm(A_co, A_x)
            res[pol] = dict(W=W, A=A_co.astype(np.complex64), A_x=A_x.astype(np.complex64),
                            gain=gain, xpol=xpol, n99=n99)
            pick = lambda a, e: gain[nearest_beam(beam_az, beam_el, a, e)]
            g0 = pick(0, 0)
            asym = max(abs(pick(a, e) - pick(-a, -e)) for a, e in ((0, el[-1]), (az[-1], 0), (az[-1], el[-1])))
            asym = max(asym, abs(pick(az[-1], el[-1]) - pick(-az[-1], el[-1])))
            say(f"  {pol}: G(0,0) {g0:6.2f} dBi (aperture eff. {100 * 10**((g0 - g_ref) / 10):4.1f} %) | "
                f"G(0,+-{el[-1]:g}) {pick(0, el[-1]):6.2f}/{pick(0, -el[-1]):6.2f} | "
                f"G(+-{az[-1]:g},0) {pick(az[-1], 0):6.2f}/{pick(-az[-1], 0):6.2f} | "
                f"corners min {min(pick(a, e) for a in (az[0], az[-1]) for e in (el[0], el[-1])):6.2f} | "
                f"grid min {gain.min():6.2f} dBi")
            say(f"      mirror asymmetry max {asym:.2f} dB | max beam cross-pol {xpol.max():6.1f} dB | "
                f"elements for 99 % power: median {int(np.median(n99))}, max {n99.max()}")
        cuts = None
        if f == f_mid:
            n_el = len(cut_el)
            cuts = {'el': (cut_el, A['H_co'][K:K + n_el]), 'az': (cut_az, A['H_co'][K + n_el:])}
            for key, ang, other in (('el', cut_el, beam_el), ('az', cut_az, beam_az)):
                on_cut = np.where((beam_az == 0) if key == 'el' else (beam_el == 0))[0]
                err = [ang[np.argmax(np.abs(cuts[key][1] @ res['H']['W'][:, k]))] - other[k] for k in on_cut]
                if err:
                    say(f"  H pointing along the {key} cut: max |peak - beam direction| {np.max(np.abs(err)):.3f} deg "
                        f"over {len(err)} beams")
        results[f] = res

        out_npz = os.path.join(args.out, f'dbf_cfm_{f_mhz:.1f}MHz.npz')
        extra = {'A_H_x': res['H']['A_x'], 'A_V_x': res['V']['A_x']} if args.save_cross else {}
        np.savez(out_npz, W_H=res['H']['W'], W_V=res['V']['W'], A_H=res['H']['A'], A_V=res['V']['A'],
                 gain_H_dBi=res['H']['gain'], gain_V_dBi=res['V']['gain'],
                 xpol_H_dB=res['H']['xpol'], xpol_V_dB=res['V']['xpol'], n99_H=res['H']['n99'], n99_V=res['V']['n99'],
                 beam_az=beam_az, beam_el=beam_el, az_grid=az, el_grid=el, freq_hz=f,
                 element_xyz=elem_pos, element_class=np.array(elem_names), time_convention_conjugated=conj,
                 field_scale=field_scale, design=os.path.abspath(args.design), F=F,
                 cst_array_center=cst_centre, **extra)
        say(f"  saved {out_npz}")
        plot_frequency(args.out, f_mhz, az, el, res, beam_az, beam_el, cuts)
        for pol in ('H', 'V'):                        # keep only what the summary plot needs
            res[pol] = {'gain': res[pol]['gain']}

    plot_gain_vs_el(args.out, freqs, az, el, results)
    if args.subband_step_mhz or args.subband_centers_mhz:
        write_subbands(args.out, say=say, **subbands)
    say(f"\nDone. Plots and summary in {args.out}")
    log.close()


if __name__ == "__main__":
    main()
