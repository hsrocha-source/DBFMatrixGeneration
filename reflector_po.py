# -*- coding: utf-8 -*-
"""Physical-optics check of the shaped reflector + array, receive mode.

A unit plane wave for beam (az, el) induces PO currents J_s = 2 n x H_inc on the lit, unshadowed
part of the reflector. The exact free-space field of those currents is evaluated at every element
centre, i.e. the elements are treated as ideal point probes (no element pattern, no coupling).
That makes this a focal-field check of the reflector design. The DBF steering vectors need the
embedded element patterns instead, see docs/dbf_matrix_procedure.md.

TPU-safe: float32 / complex64, explicit sum-products, phases wrapped before exp(), and the whole
check runs under "highest" matmul precision (scoped, so the GO code keeps its default precision).

Usage
-----
    python reflector_po.py --params design_fD0.38.json [--freq-mhz 1300] [--samples-per-lam 4]
"""
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from reflector_go import (ARRAY_SIZE_X, ARRAY_SIZE_Y, C0, D, ELEMENT_PITCH, N_ELEM_X, N_ELEM_Y,
                          X_OFFSET, Y_OFFSET, _dot3, array_frame, element_positions,
                          incident_direction, load_params, reflector_surface, scan_test_angles, zero_params)

PO_FREQ_HZ = 1315e6
ETA0 = 376.730313668
TWO_PI = 2.0 * np.pi


def wrap_phase(ph):
    """Reduce a large phase to [-pi, pi] before exp(). Helps TPU transcendentals, costs nothing."""
    return ph - TWO_PI * jnp.round(ph / TWO_PI)


def po_surface_grid(freq_hz=PO_FREQ_HZ, samples_per_lam=4, chunk=2048):
    """Midpoint-rule samples of the projected aperture, shaped (n_chunks, chunk).

    Samples sit at cell centres, so dA is exactly the cell area. Padding repeats a real
    in-aperture sample with zero weight (0 * NaN is still NaN, so the padding must sit where the
    surface is finite). Returns xs, ys, ws, dA, n_real.
    """
    h = C0 / freq_hz / samples_per_lam
    n = int(np.ceil(D / h))
    h = D / n
    xs = X_OFFSET - D / 2 + (np.arange(n) + 0.5) * h
    ys = Y_OFFSET - D / 2 + (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, ys)
    inside = ((X - X_OFFSET)**2 + (Y - Y_OFFSET)**2) <= (D / 2)**2
    x, y = X[inside], Y[inside]
    w = np.full(x.shape, h * h)
    n_real = len(x)
    pad = (-n_real) % chunk
    x = np.concatenate([x, np.full(pad, x[0])])
    y = np.concatenate([y, np.full(pad, y[0])])
    w = np.concatenate([w, np.zeros(pad)])
    f32 = lambda a: jnp.asarray(a, dtype=jnp.float32).reshape(-1, chunk)
    return f32(x), f32(y), f32(w), h * h, n_real


def pol_vectors(d):
    """x- and y-like polarisation unit vectors orthogonal to d (py equals +y at boresight)."""
    xhat = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    px = xhat - _dot3(xhat, d) * d
    px = px / jnp.linalg.norm(px)
    return px, jnp.cross(px, d)


_surf_slopes = jax.vmap(jax.grad(reflector_surface, argnums=(0, 1)), in_axes=(0, 0, None, None))


@jax.jit
def po_fields_at_elements(params, F, k0, d, pol, xs, ys, ws):
    """Scattered E and H (V/m, A/m) at every element centre for a unit-amplitude plane wave
    travelling along d with polarisation pol. Time convention exp(+j w t). Returns (E, H, frame)."""
    frame = array_frame(params, F)
    n_arr, u_vec, v_vec = frame
    r_elem, _, _ = element_positions(frame)                                     # (Ne, 3)

    def chunk_contribution(chunk):
        x, y, w = chunk
        z = reflector_surface(x, y, params, F)
        zx, zy = _surf_slopes(x, y, params, F)
        p = jnp.stack([x, y, z], axis=-1)                                       # (n, 3)
        ndS = jnp.stack([-zx, -zy, jnp.ones_like(zx)], axis=-1) * w[:, None]    # n_hat * dS, no sqrt needed

        # lit side, and the incoming ray does not pass through the array plate
        lit = _dot3(ndS, d) < 0.0
        t_in = _dot3(p, n_arr) / _dot3(d, n_arr)
        p_blk = p - t_in[:, None] * d
        blocked = (jnp.abs(_dot3(p_blk, u_vec)) < ARRAY_SIZE_X / 2) & \
                  (jnp.abs(_dot3(p_blk, v_vec)) < ARRAY_SIZE_Y / 2)
        mask = (lit & ~blocked).astype(p.dtype)

        # incident plane wave and PO current J_s = 2 n x H_inc
        kp = wrap_phase(k0 * _dot3(p, d))
        E_inc = pol[None, :] * jnp.exp(-1j * kp)[:, None]
        H_inc = jnp.cross(d[None, :], E_inc) / ETA0
        JdS = 2.0 * jnp.cross(ndS, H_inc) * mask[:, None]                       # (n, 3), A*m

        # exact free-space field of each current element at each element centre
        Rv = r_elem[None, :, :] - p[:, None, :]                                 # (n, Ne, 3)
        R = jnp.sqrt(jnp.sum(Rv * Rv, axis=-1))                                 # (n, Ne)
        Rh = Rv / R[..., None]
        g = jnp.exp(-1j * wrap_phase(k0 * R))
        Jn = JdS[:, None, :]
        JdR = jnp.sum(Jn * Rh, axis=-1)
        trans = Jn - JdR[..., None] * Rh
        longi = 3.0 * JdR[..., None] * Rh - Jn
        E = (-1j * ETA0 / (4.0 * jnp.pi * k0)) * g[..., None] * (
            k0**2 * trans / R[..., None] + longi * (1.0 / R**3 + 1j * k0 / R**2)[..., None])
        H = (1j * k0 / (4.0 * jnp.pi)) * (g * (1.0 + 1.0 / (1j * k0 * R)) / R)[..., None] * jnp.cross(Jn, Rh)
        return jnp.sum(E, axis=0), jnp.sum(H, axis=0)                          # (Ne, 3) each

    E_ch, H_ch = jax.lax.map(chunk_contribution, (xs, ys, ws))
    return jnp.sum(E_ch, axis=0), jnp.sum(H_ch, axis=0), frame


def po_metrics(E, H, frame, d, copol_is_u):
    n_arr, u_vec, v_vec = frame
    Eu = jnp.sum(E * u_vec, axis=-1)          # explicit sum-products: complex dot_general is unreliable on TPU
    Ev = jnp.sum(E * v_vec, axis=-1)

    # power through the element cells towards the array (Poynting flux * cell area)
    Sn = jnp.sum(0.5 * jnp.real(jnp.cross(E, jnp.conj(H))) * n_arr, axis=-1)
    P_cap = jnp.sum(Sn) * ELEMENT_PITCH**2
    P_inc = (1.0 / (2.0 * ETA0)) * jnp.pi * (D / 2)**2 * jnp.abs(d[2])      # unit plane wave on the projected aperture
    eff = float(P_cap / P_inc)

    Eco = (Eu if copol_is_u else Ev).reshape(N_ELEM_Y, N_ELEM_X)
    Ecx = (Ev if copol_is_u else Eu).reshape(N_ELEM_Y, N_ELEM_X)
    P2 = jnp.abs(Eco)**2 + jnp.abs(Ecx)**2
    xpol = float(jnp.sum(jnp.abs(Ecx)**2) / jnp.sum(P2))

    _, XI, YI = element_positions(frame)
    cx = float(jnp.sum(XI * P2) / jnp.sum(P2))
    cy = float(jnp.sum(YI * P2) / jnp.sum(P2))

    # adjacent-element phase step of the co-pol field, only where |E| is within 20 dB of its peak.
    # Saturates at 0.5 lambda wherever neighbours straddle a null of the focal (Airy) pattern; at
    # f/D = 0.38 the first null is ~1 pitch from the spot centre, so 0.5 is normal, not aliasing.
    amp = jnp.abs(Eco)
    strong = amp > 0.1 * jnp.max(amp)
    dphx = jnp.abs(jnp.angle(Eco[:, 1:] * jnp.conj(Eco[:, :-1])))
    dphy = jnp.abs(jnp.angle(Eco[1:, :] * jnp.conj(Eco[:-1, :])))
    mx = strong[:, 1:] & strong[:, :-1]
    my = strong[1:, :] & strong[:-1, :]
    dph_max = float(jnp.maximum(jnp.max(jnp.where(mx, dphx, 0.0)), jnp.max(jnp.where(my, dphy, 0.0)))) / TWO_PI

    return dict(eff=eff, xpol=xpol, cx=cx, cy=cy, dph_max=dph_max, Eco=np.asarray(Eco))


def run_po_check(params, F, angles, freq_hz=PO_FREQ_HZ, pols=('x', 'y'), plot_pol='x', samples_per_lam=4):
    """PO focal-field metrics per beam and polarisation; maps of |E_co| for plot_pol (None: no plot)."""
    with jax.default_matmul_precision("highest"):
        lam = C0 / freq_hz
        k0 = jnp.float32(TWO_PI / lam)
        xs, ys, ws, dA, n_surf = po_surface_grid(freq_hz, samples_per_lam)
        params32 = {k: jnp.float32(v) for k, v in params.items()}
        F32 = jnp.float32(F)

        z_chk = reflector_surface(xs.ravel(), ys.ravel(), params32, F32)
        zx_chk, zy_chk = _surf_slopes(xs.ravel(), ys.ravel(), params32, F32)
        n_bad = int((jnp.isnan(z_chk) | jnp.isnan(zx_chk) | jnp.isnan(zy_chk)).sum())
        assert n_bad == 0, f"reflector_surface or its gradient is NaN at {n_bad} PO samples"

        print(f"PO check @ {freq_hz / 1e6:.0f} MHz | lambda = {lam:.4f} m | f/D = {F / D:.3f} | "
              f"surface samples = {n_surf} ({lam / np.sqrt(dA):.1f} per lambda) | array {N_ELEM_X} x {N_ELEM_Y} | "
              f"device {jax.devices()[0].platform}")
        print(f"{'az':>6} {'el':>6} {'pol':>4} | {'eff':>7} | {'xpol':>7} | {'cx [m]':>8} {'cy [m]':>8} | {'max dphi [lam]':>14}")
        results, maps = [], []
        for az, el in angles:
            d = incident_direction(az, el).astype(jnp.float32)
            px, py = pol_vectors(d)
            for name, pol in (('x', px), ('y', py)):
                if name not in pols:
                    continue
                E, H, frame = po_fields_at_elements(params32, F32, k0, d, pol, xs, ys, ws)
                m = po_metrics(E, H, frame, d, copol_is_u=(name == 'x'))
                m.update(az=az, el=el, pol=name)
                results.append(m)
                if name == plot_pol:
                    maps.append(m)
                print(f"{az:6.1f} {el:6.1f} {name:>4} | {100 * m['eff']:6.1f}% | {100 * m['xpol']:6.2f}% | "
                      f"{m['cx']:8.3f} {m['cy']:8.3f} | {m['dph_max']:14.3f}")

    if maps:
        vmax = max(np.max(np.abs(m['Eco'])) for m in maps)
        fig, axes = plt.subplots(1, len(maps), figsize=(2.0 * len(maps), 7), sharey=True)
        axes = np.atleast_1d(axes)
        ext = [-ARRAY_SIZE_X / 2, ARRAY_SIZE_X / 2, -ARRAY_SIZE_Y / 2, ARRAY_SIZE_Y / 2]
        for ax, m in zip(axes, maps):
            db = 20 * np.log10(np.abs(m['Eco']) / vmax + 1e-12)
            im = ax.imshow(db, origin='lower', extent=ext, vmin=-30, vmax=0, cmap='viridis', aspect='equal')
            ax.plot(m['cx'], m['cy'], 'wx', ms=8, mew=2)
            ax.set_title(f"az {m['az']:g}, el {m['el']:g}\n{100 * m['eff']:.0f}%", fontsize=9)
            ax.set_xlabel('x [m]')
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        axes[0].set_ylabel('y [m]')
        fig.colorbar(im, ax=axes.ravel().tolist(), label=f'|E| co-pol ({plot_pol}) [dB]', shrink=0.6)
        plt.show()
    return results


def po_az_3db_point(params, F, el_deg, az_max=6.0, n=13, pol='x', freq_hz=PO_FREQ_HZ, samples_per_lam=4):
    """Azimuth where the PO capture falls to half of its az = 0 value, at fixed elevation.
    Returns (az_3dB or None, az samples, capture relative to az = 0)."""
    az = np.linspace(0.0, az_max, n)
    res = run_po_check(params, F, [(a, el_deg) for a in az], freq_hz=freq_hz, pols=(pol,), plot_pol=None,
                       samples_per_lam=samples_per_lam)
    eff = np.array([r['eff'] for r in res])
    rel = eff / eff[0]
    below = np.where(rel < 0.5)[0]
    if len(below) == 0:
        print(f"el = {el_deg:g} deg: still at {100 * rel[-1]:.0f}% of az = 0 at az = {az_max:g} deg, no -3 dB crossing")
        return None, az, rel
    i = below[0]
    a3 = np.interp(0.5, [rel[i], rel[i - 1]], [az[i], az[i - 1]])    # rel decreases, so reverse for np.interp
    print(f"el = {el_deg:g} deg: -3 dB in azimuth at {a3:.2f} deg (capture {100 * eff[0]:.1f}% at az = 0)")
    return a3, az, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', required=True, help="design JSON written by reflector_go.py")
    ap.add_argument('--freq-mhz', type=float, default=PO_FREQ_HZ / 1e6)
    ap.add_argument('--samples-per-lam', type=int, default=4, help="re-run with 6 to confirm convergence")
    ap.add_argument('--az3db', action='store_true', help="also find the PO azimuth -3 dB points at el = 0 and 15 deg")
    args = ap.parse_args()

    params, F = load_params(args.params)
    freq_hz = args.freq_mhz * 1e6
    kw = dict(freq_hz=freq_hz, samples_per_lam=args.samples_per_lam)

    print("Sanity check: plain paraboloid, boresight (expect well above 80%)")
    run_po_check(zero_params(), F, [(0.0, 0.0)], pols=('x',), plot_pol=None, **kw)
    print("\nDesign at the scan-range test angles")
    run_po_check(params, F, scan_test_angles(), **kw)
    if args.az3db:
        for el in (0.0, 15.0):
            po_az_3db_point(params, F, el, **kw)


if __name__ == "__main__":
    main()
