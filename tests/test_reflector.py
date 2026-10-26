import os
import sys

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reflector_go as G  # noqa: E402


def random_params(seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    return {f'z{j}': float(rng.normal(0.0, scale)) for j in G.ZERNIKE_NOLL}


def disk_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n)) * G.D / 2 * 0.999
    t = rng.uniform(0, 2 * np.pi, n)
    return G.X_OFFSET + r * np.cos(t), G.Y_OFFSET + r * np.sin(t)


def zernike_polar(x, y):
    """Polar Noll Z4..Z22 exactly as in the original notebook."""
    rho = np.hypot(x - G.X_OFFSET, y - G.Y_OFFSET) / (G.D / 2)
    p = np.arctan2(y - G.Y_OFFSET, x - G.X_OFFSET)
    s, c = np.sin, np.cos
    return np.stack([
        np.sqrt(3) * (2 * rho**2 - 1),
        np.sqrt(6) * rho**2 * s(2 * p),
        np.sqrt(6) * rho**2 * c(2 * p),
        np.sqrt(8) * (3 * rho**3 - 2 * rho) * s(p),
        np.sqrt(8) * (3 * rho**3 - 2 * rho) * c(p),
        np.sqrt(8) * rho**3 * s(3 * p),
        np.sqrt(8) * rho**3 * c(3 * p),
        np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1),
        np.sqrt(10) * (4 * rho**4 - 3 * rho**2) * c(2 * p),
        np.sqrt(10) * (4 * rho**4 - 3 * rho**2) * s(2 * p),
        np.sqrt(10) * rho**4 * c(4 * p),
        np.sqrt(10) * rho**4 * s(4 * p),
        np.sqrt(12) * (10 * rho**5 - 12 * rho**3 + 3 * rho) * c(p),
        np.sqrt(12) * (10 * rho**5 - 12 * rho**3 + 3 * rho) * s(p),
        np.sqrt(12) * (5 * rho**5 - 4 * rho**3) * c(3 * p),
        np.sqrt(12) * (5 * rho**5 - 4 * rho**3) * s(3 * p),
        np.sqrt(12) * rho**5 * c(5 * p),
        np.sqrt(12) * rho**5 * s(5 * p),
        np.sqrt(7) * (20 * rho**6 - 30 * rho**4 + 12 * rho**2 - 1),
    ], axis=-1)


def test_zernike_cartesian_matches_polar():
    x, y = disk_points()
    np.testing.assert_allclose(np.asarray(G.zernike_basis(jnp.asarray(x), jnp.asarray(y))),
                               zernike_polar(x, y), atol=2e-5)


def test_surface_on_arrays_matches_pointwise():
    # The notebook version collapsed the Zernike term to a constant when called on arrays (STL export).
    params, F = random_params(), G.F_OVER_D * G.D
    x, y = (jnp.asarray(a) for a in disk_points())
    z_vec = G.reflector_surface(x, y, params, F)
    z_pt = jax.vmap(G.reflector_surface, in_axes=(0, 0, None, None))(x, y, params, F)
    np.testing.assert_allclose(np.asarray(z_vec), np.asarray(z_pt), atol=1e-6)


def test_normal_finite_at_centre():
    n = G.surface_normal(jnp.float32(G.X_OFFSET), jnp.float32(G.Y_OFFSET), random_params(), G.F_OVER_D * G.D)
    assert np.all(np.isfinite(np.asarray(n)))


def test_paraboloid_focuses_at_origin():
    F = G.F_OVER_D * G.D
    params = G.zero_params()
    x_ap, y_ap = G.aperture_rays(40)
    x_hit, y_hit, path, cos_arr, valid = G.trace_rays_hard(
        x_ap, y_ap, G.incident_direction(0.0, 0.0), params, F, G.array_frame(params, F))
    v = np.asarray(valid)
    assert v.sum() > 0.9 * len(x_ap)                       # only the array shadow is lost
    np.testing.assert_allclose(np.asarray(x_hit)[v], 0.0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(y_hit)[v], 0.0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(path)[v], 2 * F, rtol=1e-5)  # aperture plane through the focus
    assert np.all(np.asarray(cos_arr)[v] > 0)


def test_soft_trace_agrees_with_hard_trace():
    params, F = random_params(1, 0.003), G.F_OVER_D * G.D
    frame = G.array_frame(params, F)
    x_ap, y_ap = G.aperture_rays(40)
    d = G.incident_direction(3.0, 15.0)
    xs, ys, ls, vs = G.trace_rays_soft(x_ap, y_ap, d, params, F, frame, 500.0)
    xh, yh, lh, _, vh = G.trace_rays_hard(x_ap, y_ap, d, params, F, frame)
    np.testing.assert_allclose(np.asarray(xs), np.asarray(xh), atol=1e-5)
    np.testing.assert_allclose(np.asarray(ls), np.asarray(lh), rtol=1e-5)
    assert np.mean(np.asarray(vs > 0.5) == np.asarray(vh)) > 0.995


def test_loss_and_gradient_finite():
    params, F = random_params(2, 0.003), G.F_OVER_D * G.D
    x_ap, y_ap = G.aperture_rays(20)
    loss, grads = jax.value_and_grad(G.loss_fn)(params, x_ap, y_ap, G.optimisation_directions()[::25], F, 50.0)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(float(g)) for g in grads.values())


def test_element_ordering():
    params, F = G.zero_params(), G.F_OVER_D * G.D
    r, XI, YI = G.element_positions(G.array_frame(params, F))
    assert r.shape == (G.N_ELEM_X * G.N_ELEM_Y, 3)
    i, j = 3, 17
    np.testing.assert_allclose(np.asarray(r[j * G.N_ELEM_X + i, :2]), [XI[j, i], YI[j, i]], atol=1e-6)


def test_params_roundtrip(tmp_path):
    params, F = random_params(3), 5.7
    G.save_params(params, F, tmp_path / "design.json")
    params2, F2 = G.load_params(tmp_path / "design.json")
    assert F2 == F and params2 == params


def test_stl_export_is_shaped(tmp_path):
    stl = pytest.importorskip("stl")
    pytest.importorskip("scipy")
    params, F = random_params(4), G.F_OVER_D * G.D
    G.export_stl(params, F, str(tmp_path / "r.stl"), n_r=20, n_theta=40)
    v = stl.mesh.Mesh.from_file(str(tmp_path / "r.stl")).vectors.reshape(-1, 3)
    z = np.asarray(G.reflector_surface(jnp.asarray(v[:, 0]), jnp.asarray(v[:, 1]), params, F))
    np.testing.assert_allclose(v[:, 2], z, atol=1e-4)


def test_po_paraboloid_boresight_capture():
    import reflector_po as P
    F = G.F_OVER_D * G.D
    res = P.run_po_check(G.zero_params(), F, [(0.0, 0.0)], pols=('x',), plot_pol=None, samples_per_lam=3)
    assert 0.8 < res[0]['eff'] < 1.0
    assert abs(res[0]['cx']) < 0.01 and abs(res[0]['cy']) < 0.01
