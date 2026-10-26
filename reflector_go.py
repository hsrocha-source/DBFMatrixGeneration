# -*- coding: utf-8 -*-
"""Shaped reflector for the 48 x 10 L-band DBF feed array: geometry, GO ray tracing,
Zernike surface optimisation and GO diagnostics.

Conventions (shared by reflector_po.py and docs/dbf_matrix_procedure.md)
-----------------------------------------------------------------------
* Focus at the origin. Base paraboloid z = (x^2 + y^2) / (4F) - F (vertex at z = -F), plus a
  Zernike perturbation on the rim-normalised disk centred at (X_OFFSET, Y_OFFSET).
* The array plane contains the origin. n_arr points from the reflector centre to the array
  (+z for the centred design). u_vec ~ +x is the azimuth axis (N_ELEM_X elements), v_vec ~ +y
  the elevation axis (N_ELEM_Y elements).
* Beam (az, el) is the plane wave PROPAGATING along
      d_in = (sin az, sin el, -sqrt(1 - sin^2 az - sin^2 el)),
  i.e. az and el are defined through the direction cosines of d_in. The target lies along -d_in.
* Element n = j * N_ELEM_X + i, with i = 0..N_ELEM_X-1 along u_vec and j = 0..N_ELEM_Y-1 along v_vec.

Usage
-----
    python reflector_go.py                         # optimise, save design JSON, GO reports
    python reflector_go.py --params design.json    # reports for a saved design
    python reflector_go.py --params design.json --stl shaped_reflector.stl
"""
import argparse
import json

import jax
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import optax

# ============================================================================ configuration
D = 15.0                      # projected reflector diameter [m]
F_OVER_D = 0.38               # selected design
X_OFFSET = 0.0                # rim / Zernike centre [m]; non-zero = offset geometry (tilted array)
Y_OFFSET = 0.0

ELEMENT_PITCH = 0.0926        # [m]
N_ELEM_X = 10                 # azimuth
N_ELEM_Y = 48                 # elevation
ARRAY_SIZE_X = N_ELEM_X * ELEMENT_PITCH
ARRAY_SIZE_Y = N_ELEM_Y * ELEMENT_PITCH

C0 = 299792458.0
FREQS_HZ = 1225e6 + 15e6 * np.arange(11)    # CST far-field frequencies, 1225 ... 1375 MHz
LAMBDA_MIN = C0 / FREQS_HZ[-1]              # worst case for the sampling checks

REQ_AZ_MAX, REQ_EL_MAX = 3.0, 15.0          # required scan [deg]
OPT_AZ_MAX, OPT_EL_MAX = 4.0, 15.0          # optimisation grid (1 deg azimuth margin)
OPT_N_AZ, OPT_N_EL = 17, 31

SPILL_MARGIN = 0.1                          # footprint should stay this far inside the array edge [m]
W_SPILL, W_REG, W_LOST = 2.0, 0.1, 50.0     # loss weights

ZERNIKE_NOLL = tuple(range(4, 23))          # Z4 .. Z22, params keys 'z4' .. 'z22'
ZERNIKE_REG_WEIGHT = np.array([1.0 if j < 9 else (5.0 if j < 16 else 20.0) for j in ZERNIKE_NOLL])


def _dot3(a, b):
    # Explicit sum-product instead of jnp.dot: TPUs run dot_general in bfloat16 by default.
    return jnp.sum(a * b, axis=-1)


def _unit(a):
    return a / jnp.linalg.norm(a)


# ============================================================================ surface
def zero_params():
    return {f'z{j}': 0.0 for j in ZERNIKE_NOLL}


def params_vector(params):
    return jnp.asarray([params[f'z{j}'] for j in ZERNIKE_NOLL])


def zernike_basis(x, y):
    """Noll Z4..Z22 on the rim-normalised disk, stacked on a new last axis.

    Cartesian form (rho^m cos(m phi) = Re((s + i t)^m)): identical to the polar form inside the
    rim, but polynomial, so the autodiff normal is finite at the centre. Defined for points
    inside the rim only. Quasi-orthogonal here: the Zernikes are orthogonal on the unit disk only.
    """
    s = (x - X_OFFSET) / (D / 2.0)
    t = (y - Y_OFFSET) / (D / 2.0)
    r2 = s * s + t * t
    c1, s1 = s, t                                  # rho^m cos(m phi), rho^m sin(m phi)
    c2, s2 = s * c1 - t * s1, s * s1 + t * c1
    c3, s3 = s * c2 - t * s2, s * s2 + t * c2
    c4, s4 = s * c3 - t * s3, s * s3 + t * c3
    c5, s5 = s * c4 - t * s4, s * s4 + t * c4
    sq = np.sqrt
    Z = [
        sq(3.0) * (2 * r2 - 1),                            # Z4  defocus
        sq(6.0) * s2,                                      # Z5  astigmatism
        sq(6.0) * c2,                                      # Z6
        sq(8.0) * (3 * r2 - 2) * s1,                       # Z7  coma
        sq(8.0) * (3 * r2 - 2) * c1,                       # Z8
        sq(8.0) * s3,                                      # Z9  trefoil
        sq(8.0) * c3,                                      # Z10
        sq(5.0) * (6 * r2**2 - 6 * r2 + 1),                # Z11 spherical
        sq(10.0) * (4 * r2 - 3) * c2,                      # Z12
        sq(10.0) * (4 * r2 - 3) * s2,                      # Z13
        sq(10.0) * c4,                                     # Z14
        sq(10.0) * s4,                                     # Z15
        sq(12.0) * (10 * r2**2 - 12 * r2 + 3) * c1,        # Z16
        sq(12.0) * (10 * r2**2 - 12 * r2 + 3) * s1,        # Z17
        sq(12.0) * (5 * r2 - 4) * c3,                      # Z18
        sq(12.0) * (5 * r2 - 4) * s3,                      # Z19
        sq(12.0) * c5,                                     # Z20
        sq(12.0) * s5,                                     # Z21
        sq(7.0) * (20 * r2**3 - 30 * r2**2 + 12 * r2 - 1), # Z22
    ]
    return jnp.stack(jnp.broadcast_arrays(*Z), axis=-1)


def reflector_surface(x, y, params, F):
    """Surface height z(x, y) [m]. Broadcasts over x and y."""
    z_base = (x**2 + y**2) / (4.0 * F) - F
    return z_base + jnp.sum(zernike_basis(x, y) * params_vector(params), axis=-1)


_surface_slopes = jax.grad(reflector_surface, argnums=(0, 1))


def surface_normal(x, y, params, F):
    """Unit normal on the concave (focus) side at scalar (x, y)."""
    zx, zy = _surface_slopes(x, y, params, F)
    return _unit(jnp.stack([-zx, -zy, jnp.ones_like(zx)]))


def array_frame(params, F):
    """(n_arr, u_vec, v_vec): array normal (reflector centre -> array) and in-plane az / el axes."""
    z_c = reflector_surface(X_OFFSET, Y_OFFSET, params, F)
    n_arr = _unit(-jnp.stack([jnp.asarray(X_OFFSET), jnp.asarray(Y_OFFSET), z_c]))
    ex = jnp.array([1.0, 0.0, 0.0])
    u_vec = _unit(ex - _dot3(ex, n_arr) * n_arr)
    v_vec = jnp.cross(n_arr, u_vec)
    return n_arr, u_vec, v_vec


def element_positions(frame):
    """Element centres (N, 3) in the reflector frame (n = j * N_ELEM_X + i), and the local
    coordinates XI, YI with shape (N_ELEM_Y, N_ELEM_X)."""
    _, u_vec, v_vec = frame
    xi = (jnp.arange(N_ELEM_X) - (N_ELEM_X - 1) / 2.0) * ELEMENT_PITCH
    yi = (jnp.arange(N_ELEM_Y) - (N_ELEM_Y - 1) / 2.0) * ELEMENT_PITCH
    XI, YI = jnp.meshgrid(xi, yi, indexing='xy')
    r = XI[..., None] * u_vec + YI[..., None] * v_vec
    return r.reshape(-1, 3), XI, YI


def incident_direction(az_deg, el_deg):
    """Propagation direction d_in of beam (az, el). Broadcasts; returns (..., 3)."""
    u = jnp.sin(jnp.deg2rad(az_deg))
    v = jnp.sin(jnp.deg2rad(el_deg))
    w = -jnp.sqrt(jnp.maximum(1.0 - u**2 - v**2, 1e-8))
    return jnp.stack(jnp.broadcast_arrays(u, v, w), axis=-1)


# ============================================================================ GO ray tracing
def _reflect(x, y, d_in, params, F, frame):
    """Reflector hit point, reflected direction, distance t_out to the array plane, plane hit point."""
    n_arr = frame[0]
    p_surf = jnp.stack([x, y, reflector_surface(x, y, params, F)])
    n_surf = surface_normal(x, y, params, F)
    d_out = d_in - 2.0 * _dot3(d_in, n_surf) * n_surf
    t_out = -_dot3(p_surf, n_arr) / (_dot3(d_out, n_arr) + 1e-8)
    return p_surf, d_out, t_out, p_surf + t_out * d_out


def _shadow_coords(p_surf, d_in, frame):
    """Local (x, y) where the incoming ray to p_surf crosses the array plane."""
    n_arr, u_vec, v_vec = frame
    t_in = _dot3(p_surf, n_arr) / (_dot3(d_in, n_arr) + 1e-8)
    p_blk = p_surf - t_in * d_in
    return _dot3(p_blk, u_vec), _dot3(p_blk, v_vec)


def trace_ray_soft(x, y, d_in, params, F, frame, sharpness):
    """Differentiable trace used by the optimiser.

    Returns the hit point (x, y) on the array plane in array coordinates, the path length from
    the plane-wave reference plane through the origin, and a soft validity weight in [0, 1]:
    ~1 when the ray is not shadowed by the array and the reflected ray reaches the plane.
    """
    _, u_vec, v_vec = frame
    p_surf, _, t_out, p_hit = _reflect(x, y, d_in, params, F, frame)
    # The shadow mask is not differentiated with respect to the surface point.
    bx, by = _shadow_coords(jax.lax.stop_gradient(p_surf), d_in, frame)
    blocked = (jax.nn.sigmoid(sharpness * (ARRAY_SIZE_X / 2 - jnp.abs(bx)))
               * jax.nn.sigmoid(sharpness * (ARRAY_SIZE_Y / 2 - jnp.abs(by))))
    valid = (1.0 - blocked) * jax.nn.sigmoid(sharpness * t_out)
    path_length = _dot3(p_surf, d_in) + t_out
    return _dot3(p_hit, u_vec), _dot3(p_hit, v_vec), path_length, valid


def trace_ray_hard(x, y, d_in, params, F, frame):
    """Same geometry with hard masks for the diagnostics. Also returns cos of the arrival angle
    measured from the array normal."""
    n_arr, u_vec, v_vec = frame
    p_surf, d_out, t_out, p_hit = _reflect(x, y, d_in, params, F, frame)
    bx, by = _shadow_coords(p_surf, d_in, frame)
    blocked = (jnp.abs(bx) < ARRAY_SIZE_X / 2) & (jnp.abs(by) < ARRAY_SIZE_Y / 2)
    valid = jnp.logical_not(blocked) & (t_out > 0.0)
    path_length = _dot3(p_surf, d_in) + t_out
    return _dot3(p_hit, u_vec), _dot3(p_hit, v_vec), path_length, _dot3(d_out, n_arr), valid


trace_rays_soft = jax.vmap(trace_ray_soft, in_axes=(0, 0, None, None, None, None, None))
trace_rays_hard = jax.jit(jax.vmap(trace_ray_hard, in_axes=(0, 0, None, None, None, None)))


def on_array(x_hit, y_hit):
    return (jnp.abs(x_hit) <= ARRAY_SIZE_X / 2) & (jnp.abs(y_hit) <= ARRAY_SIZE_Y / 2)


# ============================================================================ grids
def aperture_grid(n=60):
    """Square n x n sample grid over the projected aperture, and its in-rim mask."""
    x_lin = jnp.linspace(X_OFFSET - D / 2, X_OFFSET + D / 2, n)
    y_lin = jnp.linspace(Y_OFFSET - D / 2, Y_OFFSET + D / 2, n)
    X, Y = jnp.meshgrid(x_lin, y_lin)
    mask = ((X - X_OFFSET)**2 + (Y - Y_OFFSET)**2) <= (D / 2)**2
    return X, Y, mask


def aperture_rays(n=60):
    X, Y, mask = aperture_grid(n)
    return X[mask], Y[mask]


def optimisation_directions():
    az, el = jnp.meshgrid(jnp.linspace(-OPT_AZ_MAX, OPT_AZ_MAX, OPT_N_AZ),
                          jnp.linspace(-OPT_EL_MAX, OPT_EL_MAX, OPT_N_EL))
    return incident_direction(az.ravel(), el.ravel())


def scan_test_angles(az_max=REQ_AZ_MAX, el_max=REQ_EL_MAX):
    """Boresight, the four corners, the azimuth edge and the elevation edge of the scan range."""
    return [(0.0, 0.0), (az_max, el_max), (-az_max, el_max), (az_max, -el_max),
            (-az_max, -el_max), (az_max, 0.0), (0.0, el_max)]


# ============================================================================ optimisation
def loss_fn(params, x_ap, y_ap, d_grid, F, sharpness):
    frame = array_frame(params, F)

    def per_direction(d_in):
        x_hit, y_hit, _, valid = trace_rays_soft(x_ap, y_ap, d_in, params, F, frame, sharpness)
        n_valid = jnp.sum(valid) + 1e-8
        # smooth distance past (array edge - margin), squared
        over_x = jax.nn.softplus(sharpness * (jnp.abs(x_hit) - (ARRAY_SIZE_X / 2 - SPILL_MARGIN))) / sharpness
        over_y = jax.nn.softplus(sharpness * (jnp.abs(y_hit) - (ARRAY_SIZE_Y / 2 - SPILL_MARGIN))) / sharpness
        spill = jnp.sum((over_x**2 + over_y**2) * valid) / n_valid
        lost = jnp.mean(1.0 - valid)
        return spill, lost

    spill, lost = jax.vmap(per_direction)(d_grid)
    reg = jnp.sum(ZERNIKE_REG_WEIGHT * params_vector(params)**2)
    return W_SPILL * jnp.mean(spill) + W_REG * reg + W_LOST * jnp.mean(lost)


@jax.jit
def wavefront_diagnostics(params, F, X, Y, mask, d_grid):
    """Worst case over d_grid of
      * the aperture path slope |d(path)/d(aperture position)| [m/m], and
      * sin(theta_arr), theta_arr = arrival angle of the captured rays from the array normal.
    A plane wave arriving at theta_arr has a path difference pitch * sin(theta_arr) between adjacent
    elements, so the focal field is sampled without aliasing while that stays below lambda / 2.
    (Path differences between neighbouring rays are not used for this: rays converge near the
    array, which is normal for a focusing reflector, and the ratio is meaningless there.)
    """
    frame = array_frame(params, F)
    dx = jnp.abs(X[0, 1] - X[0, 0])
    dy = jnp.abs(Y[1, 0] - Y[0, 0])
    m = mask.astype(X.dtype)
    trace_hard = jax.vmap(trace_ray_hard, in_axes=(0, 0, None, None, None, None))

    def per_direction(d_in):
        _, _, path, valid = (a.reshape(X.shape) for a in
                             trace_rays_soft(X.ravel(), Y.ravel(), d_in, params, F, frame, 500.0))
        w = valid * m
        aperture_slope = jnp.maximum(jnp.max(jnp.abs(jnp.diff(path, axis=1)) / dx * w[:, :-1] * w[:, 1:]),
                                     jnp.max(jnp.abs(jnp.diff(path, axis=0)) / dy * w[:-1, :] * w[1:, :]))
        x_hit, y_hit, _, cos_arr, ok = trace_hard(X.ravel(), Y.ravel(), d_in, params, F, frame)
        captured = ok & on_array(x_hit, y_hit) & mask.ravel()
        sin_arr = jnp.max(jnp.where(captured, jnp.sqrt(jnp.clip(1.0 - cos_arr**2, 0.0, 1.0)), 0.0))
        return aperture_slope, sin_arr

    aperture_slope, sin_arr = jax.vmap(per_direction)(d_grid)
    return jnp.max(aperture_slope), jnp.max(sin_arr)


def optimize_reflector(F, x_ap, y_ap, X, Y, mask, epochs=5000, learning_rate=1e-3,
                       report_every=1500, init_params=None):
    params = dict(init_params) if init_params is not None else zero_params()
    d_grid = optimisation_directions()
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, sharpness):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_ap, y_ap, d_grid, F, sharpness)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    print(f"GO optimisation, D = {D:g} m, f/D = {F / D:.3f}, {epochs} epochs")
    for i in range(epochs):
        sharpness = 5.0 + 495.0 * (i / epochs)      # soft masks harden during the run
        loss, params, opt_state = step(params, opt_state, sharpness)

        if i % report_every == 0:
            ap_slope, sin_arr = (float(v) for v in wavefront_diagnostics(params, F, X, Y, mask, d_grid))
            elem_dl = sin_arr * ELEMENT_PITCH
            print(f"Epoch {i:5d} | loss {float(loss):.6f}")
            print(f"    max aperture path slope : {ap_slope:.4f} m/m ({ap_slope / LAMBDA_MIN:.2f} lambda/m)")
            print(f"    max arrival angle       : {np.degrees(np.arcsin(sin_arr)):.1f} deg -> adjacent-element dL "
                  f"{elem_dl:.4f} m ({elem_dl / LAMBDA_MIN:.2f} lambda at {FREQS_HZ[-1] / 1e6:.0f} MHz)")
            if elem_dl > 0.5 * LAMBDA_MIN:
                print("    [WARNING] adjacent-element path difference > lambda/2: the array under-samples the focal field.")
    return params


# ============================================================================ GO reports
def print_capture_table(params, F, x_ap, y_ap, angles=None):
    angles = scan_test_angles() if angles is None else angles
    frame = array_frame(params, F)
    n_tot = len(x_ap)
    print(f"\n--- GO capture (f/D = {F / D:.3f}) ---")
    for az, el in angles:
        x_hit, y_hit, _, _, valid = trace_rays_hard(x_ap, y_ap, incident_direction(az, el), params, F, frame)
        n_valid = int(jnp.sum(valid))
        n_cap = int(jnp.sum(valid & on_array(x_hit, y_hit)))
        print(f"  Az {az:5.1f} deg, El {el:6.1f} deg | unshadowed {n_valid:4d}/{n_tot:4d} | "
              f"captured {n_cap:4d} | eff {100.0 * n_cap / n_tot:5.1f}%")


def plot_ray_footprints(params, F, x_ap, y_ap, angles=None, plot_margin=0.3):
    """Unshadowed ray hits on the array plane per beam; 'x' marks the centroid of the captured rays."""
    angles = scan_test_angles() if angles is None else angles
    frame = array_frame(params, F)
    fig, ax = plt.subplots(figsize=(3.5, 11))
    ax.add_patch(patches.Rectangle((-ARRAY_SIZE_X / 2, -ARRAY_SIZE_Y / 2), ARRAY_SIZE_X, ARRAY_SIZE_Y,
                                   linewidth=2, edgecolor='black', facecolor='none', linestyle='--',
                                   label='array boundary'))
    colors = plt.cm.tab10.colors
    for idx, (az, el) in enumerate(angles):
        x_hit, y_hit, _, _, valid = (np.asarray(a) for a in trace_rays_hard(
            x_ap, y_ap, incident_direction(az, el), params, F, frame))
        captured = valid & np.asarray(on_array(x_hit, y_hit))
        color = colors[idx % len(colors)]
        ax.scatter(x_hit[valid], y_hit[valid], s=2, alpha=0.5, color=color, label=f'az {az:g}, el {el:g}')
        if captured.any():
            ax.plot(x_hit[captured].mean(), y_hit[captured].mean(), marker='x', color='black',
                    markersize=8, markeredgewidth=2)
    ax.set_xlim(-ARRAY_SIZE_X / 2 - plot_margin, ARRAY_SIZE_X / 2 + plot_margin)
    ax.set_ylim(-ARRAY_SIZE_Y / 2 - plot_margin, ARRAY_SIZE_Y / 2 + plot_margin)
    ax.set_aspect('equal')
    ax.set_xlabel("array x [m] (azimuth)")
    ax.set_ylabel("array y [m] (elevation)")
    ax.set_title("GO ray footprints on the array plane")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
    plt.tight_layout()
    plt.show()


@jax.jit
def _capture_fraction(params, F, d_in, x_ap, y_ap):
    frame = array_frame(params, F)
    x_hit, y_hit, _, _, valid = jax.vmap(trace_ray_hard, in_axes=(0, 0, None, None, None, None))(
        x_ap, y_ap, d_in, params, F, frame)
    return jnp.mean(valid & on_array(x_hit, y_hit))


go_capture_sweep = jax.jit(jax.vmap(_capture_fraction, in_axes=(None, None, 0, None, None)))


def half_power_angle(angles, eff):
    """First angle at which the capture drops below half of its value at angles[0] (linear interpolation)."""
    eff = np.asarray(eff)
    angles = np.asarray(angles)
    if eff[0] <= 0:
        return np.nan
    rel = eff / eff[0]
    below = np.where(rel < 0.5)[0]
    if len(below) == 0:
        return np.nan                       # no -3 dB crossing in the swept range
    i = below[0]
    a0, a1, r0, r1 = angles[i - 1], angles[i], rel[i - 1], rel[i]
    return a0 + (0.5 - r0) * (a1 - a0) / (r1 - r0)


def go_3db_report(params, F, x_ap, y_ap, az_max=8.0, az_step=0.25, el_max=20.0, el_step=0.5,
                  el_fixed=REQ_EL_MAX, az_fixed=REQ_AZ_MAX, plot=True):
    """Angles where the GO capture falls to half of its on-axis value (scan coverage, not beamwidth)."""
    az_s = np.arange(0.0, az_max + 1e-9, az_step)
    el_s = np.arange(0.0, el_max + 1e-9, el_step)
    sweeps = {
        'az, el = 0': (az_s, incident_direction(az_s, 0.0)),
        f'az, el = {el_fixed:g}': (az_s, incident_direction(az_s, el_fixed)),
        'el, az = 0': (el_s, incident_direction(0.0, el_s)),
        f'el, az = {az_fixed:g}': (el_s, incident_direction(az_fixed, el_s)),
    }
    curves = {k: (ang, np.asarray(go_capture_sweep(params, F, d, x_ap, y_ap))) for k, (ang, d) in sweeps.items()}
    result = {'eff0': float(curves['az, el = 0'][1][0])}
    print(f"\n--- GO -3 dB capture angles (f/D = {F / D:.3f}, on-axis capture {100 * result['eff0']:.1f}%) ---")
    for k, (ang, e) in curves.items():
        a3 = half_power_angle(ang, e)
        result[k] = a3
        print(f"  sweep {k:<14}: " + (f"{a3:6.2f} deg" if np.isfinite(a3) else "> swept range"))

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        for a, (k, (ang, e)) in zip(axes.ravel(), curves.items()):
            a.plot(ang, 10 * np.log10(np.maximum(e / max(e[0], 1e-12), 1e-6)))
            a.axhline(-3.0, color='k', ls='--', lw=1)
            a.set_title(f"sweep {k}")
            a.set_xlabel("angle [deg]")
            a.set_ylim(-15, 1)
            a.grid(alpha=0.3)
        for a in axes[:, 0]:
            a.set_ylabel("capture rel. to angle 0 [dB]")
        plt.tight_layout()
        plt.show()
    return result


def q_from_edge_level(theta_deg, level_db):
    """Exponent q of a cos^q(theta) power pattern that is level_db down at theta_deg."""
    return -level_db / (-10.0 * np.log10(np.cos(np.radians(theta_deg))))


def arrival_stats(params, F, az, el, x_ap, y_ap, theta_max=70.0, elem_pattern=None):
    frame = array_frame(params, F)
    x_hit, y_hit, _, cos_arr, valid = (np.asarray(a) for a in trace_rays_hard(
        x_ap, y_ap, incident_direction(az, el), params, F, frame))
    theta = np.degrees(np.arccos(np.clip(cos_arr, -1.0, 1.0)))
    captured = valid & np.asarray(on_array(x_hit, y_hit))
    n_tot = len(x_ap)
    n_cap = captured.sum()
    w = elem_pattern(theta) if elem_pattern is not None else np.ones_like(theta)
    return dict(
        eff=n_cap / n_tot,                                               # same as print_capture_table
        frac_out=(captured & (theta > theta_max)).sum() / max(n_cap, 1),  # captured share beyond theta_max
        eff_usable=(captured & (theta <= theta_max)).sum() / n_tot,      # captured AND within theta_max
        eff_weighted=(w * captured).sum() / n_tot,                       # captured, weighted by element power pattern
        theta_cap=theta[captured],
        theta_95=np.percentile(theta[captured], 95) if n_cap else np.nan,
    )


def arrival_report(params, F, x_ap, y_ap, angles=None, theta_max=70.0, edge_level_db=-8.0, plot=True):
    """Arrival angle of the captured rays at the array, with a cos^q element model."""
    if angles is None:
        angles = ((0.0, 0.0), (0.0, REQ_EL_MAX), (REQ_AZ_MAX, 0.0), (REQ_AZ_MAX, REQ_EL_MAX))
    q = q_from_edge_level(theta_max, edge_level_db)
    pattern = lambda th: np.clip(np.cos(np.radians(th)), 0.0, None)**q
    print(f"\n--- Arrival angles (f/D = {F / D:.3f}); element model cos^{q:.2f}(theta), "
          f"{edge_level_db:g} dB at {theta_max:g} deg ---")
    print(f"{'az':>5} {'el':>5} | {'eff':>6} | {'out>%g' % theta_max:>7} | {'usable':>7} | {'weighted':>8} | {'th95':>6}")
    rows = []
    for az, el in angles:
        s = arrival_stats(params, F, az, el, x_ap, y_ap, theta_max, pattern)
        rows.append((az, el, s))
        print(f"{az:5.1f} {el:5.1f} | {100 * s['eff']:5.1f}% | {100 * s['frac_out']:6.1f}% | "
              f"{100 * s['eff_usable']:6.1f}% | {100 * s['eff_weighted']:7.1f}% | {s['theta_95']:6.1f}")

    if plot:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bins = np.arange(0, 91, 2.5)
        for az, el, s in rows:
            ax.hist(s['theta_cap'], bins=bins, histtype='step', lw=1.5, label=f"az {az:g}, el {el:g}")
        ax.axvline(theta_max, color='k', ls='--', lw=1)
        ax.set_xlabel("arrival angle from array normal [deg]")
        ax.set_ylabel("captured rays")
        ax.set_title("Arrival-angle distribution of captured rays")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
    return rows


# ============================================================================ I/O
def save_params(params, F, path):
    data = {'D': D, 'F': float(F), 'f_over_D': float(F) / D,
            'zernike': {k: float(v) for k, v in params.items()}}
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)


def load_params(path):
    """Returns (params, F)."""
    with open(path) as fh:
        data = json.load(fh)
    if abs(data['D'] - D) > 1e-9:
        raise ValueError(f"design file is for D = {data['D']} m, module is configured for D = {D} m")
    return {k: float(v) for k, v in data['zernike'].items()}, float(data['F'])


def export_stl(params, F, filename, n_r=200, n_theta=600):
    """Triangulated surface for CAD / EM solvers (needs numpy-stl and scipy)."""
    from scipy.spatial import Delaunay
    from stl import mesh

    r, th = np.meshgrid(np.linspace(0.0, D / 2.0, n_r), np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False))
    # rounding + unique removes the repeated centre point
    pts = np.unique(np.round(np.column_stack([(r * np.cos(th)).ravel(), (r * np.sin(th)).ravel()]), 6), axis=0)
    x = pts[:, 0] + X_OFFSET
    y = pts[:, 1] + Y_OFFSET
    z = np.asarray(reflector_surface(jnp.asarray(x), jnp.asarray(y), params, F))
    tri = Delaunay(pts)

    surf = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
    surf.vectors = np.column_stack([x, y, z])[tri.simplices]
    surf.update_normals()
    if np.mean(surf.normals[:, 2]) < 0:          # normals on the concave (focus) side
        surf.vectors = surf.vectors[:, [0, 2, 1], :]
        surf.update_normals()
    surf.save(filename)
    print(f"Exported '{filename}' with {len(tri.simplices)} facets (F = {F:.3f} m, D = {D:g} m).")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', help="load a saved design (JSON) instead of optimising")
    ap.add_argument('--out', default=f'design_fD{F_OVER_D:.2f}.json', help="where to save an optimised design")
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--n-aperture', type=int, default=60, help="aperture rays per side")
    ap.add_argument('--stl', help="also export the surface to this STL file")
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    X, Y, mask = aperture_grid(args.n_aperture)
    x_ap, y_ap = X[mask], Y[mask]

    if args.params:
        params, F = load_params(args.params)
    else:
        F = F_OVER_D * D
        params = optimize_reflector(F, x_ap, y_ap, X, Y, mask, epochs=args.epochs)
        save_params(params, F, args.out)
        print(f"Saved design to {args.out}")

    plot = not args.no_plots
    print_capture_table(params, F, x_ap, y_ap)
    if plot:
        plot_ray_footprints(params, F, x_ap, y_ap)
    go_3db_report(params, F, x_ap, y_ap, plot=plot)
    arrival_report(params, F, x_ap, y_ap, plot=plot)
    if args.stl:
        export_stl(params, F, args.stl)


if __name__ == "__main__":
    main()
