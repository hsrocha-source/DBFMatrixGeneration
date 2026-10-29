# -*- coding: utf-8 -*-
"""Synthetic CST exports with the structure of the real data, for testing dbf_cfm.py without CST:
one .ffs farfield source per (port, frequency) for the 17 x 2 ports of cst/port_map.csv, and a
160-port Touchstone file whose S-parameters are consistent with the farfield powers.

The patterns follow the ideal dense-array embedded element: power ~ cos(theta), realized gain
4 pi d^2 / lambda^2 at broadside, radiating pi (d / lambda)^2 of the stimulated power; the rest is
coupled into the other ports. Edge classes get a small tilt so that the classes differ.

    python tests/dummy_cst.py OUT_DIR [--freqs-mhz 1225 1300 1375] [--step 2] [--convention +jwt]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import reflector_go as G  # noqa: E402
from cst_io import ETA0, read_port_map  # noqa: E402
from dbf_cfm import cst_element_centre  # noqa: E402

PORT_MAP = os.path.join(os.path.dirname(HERE), 'cst', 'port_map.csv')


def huygens(theta, phi, pol, q=1.0, tilt=(0.0, 0.0), phase_curv=0.4):
    """E_theta, E_phi on the (theta, phi) grid of an x- (H) or y- (V) polarized Huygens-like
    element radiating toward +z: cos^(q/2) amplitude, optional tilt and a mild phase pattern."""
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    ct = np.cos(TH)
    f = np.where(ct > 0, np.clip(ct, 0, None)**(q / 2), 0.0)
    f = f * (1 + np.sin(TH) * (tilt[0] * np.cos(PH) + tilt[1] * np.sin(PH))) * np.exp(1j * phase_curv * np.sin(TH)**2)
    if pol == 'H':
        return f * np.cos(PH), -f * np.sin(PH)
    return f * np.sin(PH), f * np.cos(PH)


def sphere_power(theta, phi, e_theta, e_phi):
    """(1 / 2 eta) * sphere integral of |E|^2; phi grid without the 360 deg duplicate."""
    p = np.abs(e_theta)**2 + np.abs(e_phi)**2
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return trapz(p.sum(axis=1) * (2 * np.pi / len(phi)) * np.sin(theta), theta) / (2 * ETA0)


def write_ffs(path, theta_deg, phi_deg, e_theta, e_phi, freq_hz, p_rad, p_acc, p_stim, position=(0, 0, 0)):
    """CST farfield source (version 3.0 layout). phi_deg must include 360 like CST exports do."""
    hdr = ["// CST Farfield Source File", " ", "// Version:", "3.0", " ", "// Data Type", "Farfield", " ",
           "// #Frequencies", "1", " ", "// Position", " ".join(f"{v:.6e}" for v in position), " ",
           "// zAxis", "0.000000e+00 0.000000e+00 1.000000e+00", " ",
           "// xAxis", "1.000000e+00 0.000000e+00 0.000000e+00", " ",
           "// Radiated/Accepted/Stimulated Power , Frequency",
           f"{p_rad:.8e}", f"{p_acc:.8e}", f"{p_stim:.8e}", f"{freq_hz:.8e}", " ", " ",
           "// >> Total #phi samples, total #theta samples", f"{len(phi_deg)} {len(theta_deg)}", " ",
           "// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi): "]
    PH, TH = np.meshgrid(phi_deg, theta_deg, indexing='ij')            # phi outer, theta inner
    et, ep = e_theta.T, e_phi.T
    data = np.column_stack([PH.ravel(), TH.ravel(), et.real.ravel(), et.imag.ravel(), ep.real.ravel(), ep.imag.ravel()])
    with open(path, 'w') as fh:
        fh.write("\n".join(hdr) + "\n")
        np.savetxt(fh, data, fmt=['%8.3f', '%8.3f', '%15.7e', '%15.7e', '%15.7e', '%15.7e'])


def dummy_s(n_cols, n_rows, f_hz):
    """Reciprocal S-matrix: -16.5 dB match, coupling decaying with distance, weak cross-pol."""
    k = 2 * np.pi * f_hz / G.C0
    col, row = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    xy = np.column_stack([col.ravel(), row.ravel()]) * G.ELEMENT_PITCH
    r = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    rp = np.where(r > 0, r / G.ELEMENT_PITCH, 1.0)
    co = np.where(r > 0, 0.05 / rp**1.5 * np.exp(-1j * k * r), 0.15 * np.exp(0.3j))
    cx = np.where(r > 0, 0.01 / rp**2 * np.exp(-1j * k * r), 0.005)
    n = 2 * len(xy)
    S = np.zeros((n, n), complex)
    S[0::2, 0::2] = co                                       # ports 2e-1 (H), 2e (V)
    S[1::2, 1::2] = co
    S[0::2, 1::2] = cx
    S[1::2, 0::2] = cx
    return S


def write_touchstone(path, freqs_hz, s_list, z0=50.0):
    with open(path, 'w') as fh:
        fh.write("! dummy CST S-parameters\n")
        fh.write(f"# HZ S RI R {z0:g}\n")
        for f, S in zip(freqs_hz, s_list):
            for m, srow in enumerate(S):
                vals = np.column_stack([srow.real, srow.imag]).ravel()
                for j in range(0, len(vals), 8):            # 4 complex values per line
                    lead = f"{f:.6e} " if (m == 0 and j == 0) else " " * 14
                    fh.write(lead + " ".join(f"{v:.6e}" for v in vals[j:j + 8]) + "\n")


def _tilt(name):
    """Pattern tilt toward the array interior, stronger for edge than for n1 elements."""
    a = 0.25 if not name.startswith('n1_') else 0.12
    tag = name.split('_')[-1] if name != 'centre' else ''
    return (a * (('L' in tag) - ('R' in tag)), a * (('B' in tag) - ('T' in tag)))


def make_dataset(out_dir, freqs_hz, step_deg=2.0, convention='+jwt', cst_center=(0.0, 0.0, 0.0),
                 position=(0.0, 0.0, 0.0)):
    os.makedirs(out_dir, exist_ok=True)
    pm = read_port_map(PORT_MAP)
    n_cols, n_rows = max(r['col'] for r in pm), max(r['row'] for r in pm)
    theta_deg = np.arange(0.0, 180.0 + 1e-9, step_deg)
    phi_deg = np.arange(0.0, 360.0 + 1e-9, step_deg)
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    r_hat = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)

    s_list = [dummy_s(n_cols, n_rows, f) for f in freqs_hz]
    p_stim = 0.5
    for f, S in zip(freqs_hz, s_list):
        k = 2 * np.pi * f / G.C0
        for r in pm:
            if not r['eep_class']:
                continue
            centre = cst_element_centre(r['col'], r['row'], n_cols, n_rows, cst_center)
            shift = np.exp(1j * k * (r_hat @ (centre - np.asarray(position))))     # e^{+jwt}: e^{+jk r.d}
            for pol in ('H', 'V'):
                port = r[f'port_{pol}']
                et, ep = huygens(theta, phi, pol, tilt=_tilt(r['eep_class']))
                et, ep = et * shift, ep * shift
                p_rad = p_stim * np.pi * (G.ELEMENT_PITCH * f / G.C0)**2
                # coupling into the other ports carries the rest: scale this column's off-diagonal terms
                col = S[:, port - 1]
                off = np.arange(len(col)) != port - 1
                col[off] *= np.sqrt((1 - p_rad / p_stim - abs(col[port - 1])**2) / np.sum(np.abs(col[off])**2))
                scale = np.sqrt(p_rad / sphere_power(theta, phi[:-1], et[:, :-1], ep[:, :-1]))
                et, ep = et * scale, ep * scale
                if convention == '-jwt':
                    et, ep = np.conj(et), np.conj(ep)
                write_ffs(os.path.join(out_dir, f'farfield_f{f / 1e9:g}_port{port}.ffs'), theta_deg, phi_deg,
                          et, ep, f, p_rad, p_stim * (1 - abs(S[port - 1, port - 1])**2), p_stim, position)
    write_touchstone(os.path.join(out_dir, f'dummy.s{2 * n_cols * n_rows}p'), freqs_hz, s_list)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir')
    ap.add_argument('--freqs-mhz', type=float, nargs='+', default=[1225.0, 1300.0, 1375.0])
    ap.add_argument('--step', type=float, default=2.0, help="theta / phi step [deg]")
    ap.add_argument('--convention', choices=('+jwt', '-jwt'), default='+jwt')
    a = ap.parse_args()
    make_dataset(a.out_dir, [f * 1e6 for f in a.freqs_mhz], a.step, a.convention)
    print(f"dummy CST data written to {a.out_dir}")
