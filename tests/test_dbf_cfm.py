import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import dbf_cfm as DB  # noqa: E402
import dummy_cst  # noqa: E402
import reflector_go as G  # noqa: E402
from cst_io import ETA0, read_ffs, read_port_map, read_touchstone  # noqa: E402


def test_ffs_roundtrip(tmp_path):
    th, ph = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)
    et, ep = dummy_cst.huygens(np.radians(th), np.radians(ph), 'V', tilt=(0.1, -0.2))
    path = tmp_path / "farfield_f1.3_port42.ffs"
    dummy_cst.write_ffs(path, th, ph, et, ep, 1.3e9, 0.4, 0.45, 0.5, position=(0.1, -0.2, 0.0))
    src = read_ffs(path)
    assert src.port == 42 and src.freqs[0] == 1.3e9
    np.testing.assert_allclose(src.position, [0.1, -0.2, 0.0])
    np.testing.assert_allclose([src.p_rad[0], src.p_acc[0], src.p_stim[0]], [0.4, 0.45, 0.5])
    assert src.e_theta.shape == (1, len(th), len(ph) - 1)              # 360 deg duplicate dropped
    np.testing.assert_allclose(src.e_theta[0], et[:, :-1], atol=1e-6)
    np.testing.assert_allclose(src.e_phi[0], ep[:, :-1], atol=1e-6)
    hdr = read_ffs(path, header_only=True)
    assert hdr.freqs[0] == 1.3e9 and hdr.theta.size == 0


def test_touchstone_roundtrip(tmp_path):
    S = [dummy_cst.dummy_s(10, 8, f) for f in (1.25e9, 1.3e9)]
    path = tmp_path / "x.s160p"
    dummy_cst.write_touchstone(path, [1.25e9, 1.3e9], S)
    ts = read_touchstone(path)
    np.testing.assert_allclose(ts.freqs, [1.25e9, 1.3e9])
    np.testing.assert_allclose(ts.s, np.stack(S), atol=1e-6)


def test_flight_array_class_counts():
    names = DB.flight_array_classes()
    families = Counter(n.split('_')[0] + ('_' + n.split('_')[1] if n.startswith('n1') else '') for n in names)
    assert families == {'corner': 4, 'edge': 16 + 92, 'n1_corner': 4, 'n1_edge': 12 + 88, 'centre': 264}
    # the CST -> reflector rotation flips y: reflector top-left is the CST bottom-left corner
    assert names[(G.N_ELEM_Y - 1) * G.N_ELEM_X + 0] == 'corner_BL'
    assert names[0] == 'corner_TL'
    assert set(names) == {r['eep_class'] for r in read_port_map(os.path.join(os.path.dirname(HERE), 'cst', 'port_map.csv'))
                          if r['eep_class']}


class _OneFile:
    def __init__(self, src):
        self.src = src

    def get(self, port, f_hz):
        return self.src, self.src.freq_index(f_hz)


def test_po_gain_matches_silver_formula(tmp_path):
    """Single Huygens feed at the focus of the plain paraboloid, no blockage: the PO boresight gain
    must match the aperture-efficiency formula cot^2(t0/2) |int sqrt(G_f) tan(t/2) dt|^2."""
    f = 1300e6
    q = 2.5                                                        # power cos^q feed, ~-10 dB at the rim
    th, ph = np.arange(0, 180.01, 1.0), np.arange(0, 360.01, 2.0)
    et, ep = dummy_cst.huygens(np.radians(th), np.radians(ph), 'H', q=q, phase_curv=0.0)
    p_rad = dummy_cst.sphere_power(np.radians(th), np.radians(ph[:-1]), et[:, :-1], ep[:, :-1])
    path = tmp_path / "farfield_f1.3_port1.ffs"
    dummy_cst.write_ffs(path, th, ph, et, ep, f, p_rad, p_rad, p_rad)
    src = read_ffs(path)

    F = G.F_OVER_D * G.D
    params = G.zero_params()
    cls = [{'port_H': 1, 'port_V': 1, 'eep_class': 'feed'}]
    tab_H, tab_V, grid, axes = DB.pattern_table(_OneFile(src), cls, np.zeros((1, 3)), f, False, 1.0)
    A = DB.steering_matrices(params, F, f, np.array([[0.0, 0.0, -1.0]]), np.zeros((1, 3)), np.array([0]),
                             tab_H, tab_V, grid, axes, samples_per_lam=4, blockage=False, say=lambda *a: None)
    g_po = 2 * np.pi / ETA0 * abs(A['H_co'][0, 0])**2

    t0 = 2 * np.arctan(G.D / (4 * F))
    t = np.linspace(0, t0, 4001)
    gf = 2 * (q + 1) * np.cos(t)**q                                # cos^q power pattern, normalized gain
    eta = (1 / np.tan(t0 / 2))**2 * (getattr(np, "trapezoid", None) or np.trapz)(np.sqrt(gf) * np.tan(t / 2), t)**2
    g_silver = 4 * np.pi * np.pi * (G.D / 2)**2 / (G.C0 / f)**2 * eta
    print(f"PO {10 * np.log10(g_po):.3f} dBi, Silver {10 * np.log10(g_silver):.3f} dBi (eta {eta:.4f})")
    assert abs(10 * np.log10(g_po / g_silver)) < 0.1
    assert abs(A['H_x'][0, 0]) < 1e-3 * abs(A['H_co'][0, 0])         # Huygens feed: no cross-pol


def test_time_convention_detected(tmp_path):
    out = dummy_cst.make_dataset(str(tmp_path), [1300e6], step_deg=3.0, convention='-jwt')
    pm = read_port_map(dummy_cst.PORT_MAP)
    classes = [r for r in pm if r['eep_class']]
    cst_pos = np.array([DB.cst_element_centre(c['col'], c['row'], 10, 8, (0, 0, 0)) for c in classes])
    eeps = DB.EEPIndex(out, {c[f'port_{p}'] for c in classes for p in ('H', 'V')})
    assert DB.phase_reference_check(eeps, classes, cst_pos, 1300e6, 'auto', say=lambda *a: None) is True


def test_end_to_end_dummy(tmp_path):
    data = dummy_cst.make_dataset(str(tmp_path / "ffs"), [1300e6], step_deg=3.0)
    out = tmp_path / "out"
    DB.main(['--design', os.path.join(os.path.dirname(HERE), 'designs', 'design_fD0.38.json'),
             '--ffs-dir', data, '--touchstone', os.path.join(data, 'dummy.s160p'), '--out', str(out),
             '--az-step', '3', '--el-step', '7.5', '--samples-per-lam', '3'])
    r = np.load(out / "dbf_cfm_1300.0MHz.npz")
    K = len(r['beam_az'])
    assert r['W_H'].shape == (480, K) and r['A_H'].shape == (K, 480)
    np.testing.assert_allclose(np.linalg.norm(r['W_H'], axis=0), 1.0, rtol=1e-5)        # unit-norm CFM
    resp = np.einsum('kn,nk->k', r['A_H'], r['W_H'])                                     # real, = |a_k|
    np.testing.assert_allclose(resp.imag, 0.0, atol=1e-4 * np.abs(resp).max())
    lam = G.C0 / 1300e6
    eff0 = 10**(r['gain_H_dBi'][np.argmin(r['beam_az']**2 + r['beam_el']**2)] / 10) / (4 * np.pi**2 * (G.D / 2)**2 / lam**2)
    assert 0.5 < eff0 < 1.0
    text = (out / "summary.txt").read_text()
    assert "e^{+jwt}" in text and "peak phasors" in text
    for name in ("gain_map_H_1300MHz.png", "weights_H_1300MHz.png", "cuts_H_1300MHz.png", "gain_vs_el.png"):
        assert (out / name).exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
