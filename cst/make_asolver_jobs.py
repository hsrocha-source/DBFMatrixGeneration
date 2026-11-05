# -*- coding: utf-8 -*-
"""Prepare the CST asymptotic-solver comparison (cst/asolver_secondary_patterns.bas).

Writes into --out (put that folder next to the CST reflector project):
  sources/eep_<class>_<pol>_f<GHz>.ffs   one farfield source per EEP class, polarization and frequency:
                                         the CST pattern de-embedded to its own element centre and
                                         normalized to 1 W stimulated power, still in CST's own time
                                         and field conventions, so CST can import it as is
  jobs.csv                               one row per A-solver run: flight-array element, polarization,
                                         frequency, source position in the CST reflector project
                                         (vertex at the origin, opening toward +z), source file
  manifest.json                          geometry, conventions and normalization for reading the results

The CST array position and time convention are found from the patterns exactly as in dbf_cfm.py.

    python cst/make_asolver_jobs.py --ffs-dir EEP_ffs --out asolver
    python cst/make_asolver_jobs.py --ffs-dir EEP_ffs --out asolver --freqs-ghz 1.225 1.3 1.375 --pols H
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dbf_cfm as DB  # noqa: E402
import reflector_go as G  # noqa: E402
from cst_io import read_port_map, write_ffs  # noqa: E402

THETA_AXIS = (0.0, 0.0, -1.0)   # where the element's +z (broadside) points in the reflector project
PHI_AXIS = (1.0, 0.0, 0.0)      # where its +x (H polarization) points: (x, y, z) -> (x, -y, -z)


def parse_elements(spec, n_total):
    """'all' or a list like '0-9,236,470-479' -> sorted element indices."""
    if spec == 'all':
        return list(range(n_total))
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    if not out or min(out) < 0 or max(out) >= n_total:
        raise ValueError(f"element indices must be within 0 ... {n_total - 1}")
    return sorted(out)


def write_local_source(path, src, fi, delta, conj, k):
    """Source file with the phase reference moved by delta (CST frame) to the element centre and the
    fields scaled to 1 W stimulated power, in the file's own conventions."""
    _, r_hat, _ = DB.field_cartesian(src, fi)
    shift = np.exp((1j if conj else -1j) * k * (r_hat @ delta))        # e^{-jk r.d} in e^{+jwt}
    scale = shift / np.sqrt(src.p_stim[fi])
    et, ep = src.e_theta[fi] * scale, src.e_phi[fi] * scale
    close = lambda a: np.concatenate([a, a[:, :1]], axis=1)              # repeat phi = 0 at 360 deg
    write_ffs(path, np.degrees(src.theta), np.append(np.degrees(src.phi), 360.0), close(et), close(ep),
              src.freqs_raw[fi], src.p_rad[fi] / src.p_stim[fi], src.p_acc[fi] / src.p_stim[fi], 1.0,
              position=(0.0, 0.0, 0.0), x_axis=src.x_axis, z_axis=src.z_axis)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ffs-dir', required=True, help="folder with the CST .ffs farfield sources (EEP_ffs)")
    ap.add_argument('--out', default='asolver')
    ap.add_argument('--design', default=os.path.join(os.path.dirname(HERE), 'designs', 'design_fD0.38.json'))
    ap.add_argument('--port-map', default=os.path.join(HERE, 'port_map.csv'))
    ap.add_argument('--freqs-ghz', type=float, nargs='+', default=[1.3])
    ap.add_argument('--pols', nargs='+', choices=('H', 'V'), default=['H', 'V'])
    ap.add_argument('--elements', default='all', help="'all' or e.g. '0-9,236,470-479' (n = j * 10 + i)")
    ap.add_argument('--cst-array-center', nargs='+', default=['auto'], metavar='auto|X Y Z')
    ap.add_argument('--cst-array-z', type=float, default=0.0)
    ap.add_argument('--time-convention', choices=('auto', '+jwt', '-jwt'), default='auto')
    args = ap.parse_args(argv)
    centre_arg = None if args.cst_array_center == ['auto'] else [float(v) for v in args.cst_array_center]

    params, F = G.load_params(args.design)
    z_vertex = float(G.reflector_surface(0.0, 0.0, params, F))       # focus frame: focus / array plane at z = 0
    array_z = -z_vertex                                                # in the CST project: vertex at z = 0
    pm = read_port_map(args.port_map)
    classes = [r for r in pm if r['eep_class']]
    n_cols, n_rows = max(r['col'] for r in pm), max(r['row'] for r in pm)
    layout = np.array([DB.cst_element_centre(c['col'], c['row'], n_cols, n_rows, (0.0, 0.0, 0.0)) for c in classes])
    eeps = DB.EEPIndex(args.ffs_dir, {c[f'port_{p}'] for c in classes for p in ('H', 'V')})
    freqs = eeps.freqs([c[f'port_{p}'] for c in classes for p in ('H', 'V')])
    pick = [freqs[np.argmin(np.abs(freqs - f * 1e9))] for f in args.freqs_ghz]
    for f, want in zip(pick, args.freqs_ghz):
        if abs(f - want * 1e9) > 1e6:
            raise ValueError(f"no EEPs at {want} GHz (available: {', '.join(f'{v / 1e9:g}' for v in freqs)})")

    print(f"Locating the simulated array at {pick[len(pick) // 2] / 1e6:.0f} MHz ...")
    conj, centre = DB.locate_array(eeps, classes, layout, pick[len(pick) // 2], centre_arg, args.cst_array_z,
                                   args.time_convention, print)
    cst_pos = layout + centre

    os.makedirs(os.path.join(args.out, 'sources'), exist_ok=True)
    src_name = {}
    for f in pick:
        k = 2 * np.pi * f / G.C0
        for c, cls in enumerate(classes):
            for pol in args.pols:
                src, fi = eeps.get(cls[f'port_{pol}'], f)
                name = f"eep_{cls['eep_class']}_{pol}_f{f / 1e9:g}GHz.ffs"
                write_local_source(os.path.join(args.out, 'sources', name), src, fi, cst_pos[c] - src.position, conj, k)
                src_name[(cls['eep_class'], pol, f)] = name
    print(f"wrote {len(src_name)} source files to {os.path.join(args.out, 'sources')}")

    elem_names = DB.flight_array_classes()
    pos = np.asarray(G.element_positions(G.array_frame(params, F))[0], float)
    elements = parse_elements(args.elements, len(elem_names))
    port_of = {c['eep_class']: c for c in classes}
    rows, job = [], 0
    for f in pick:
        for pol in args.pols:
            for n in elements:
                cls = elem_names[n]
                rows.append([job, n, n % G.N_ELEM_X, n // G.N_ELEM_X, cls, pol, port_of[cls][f'port_{pol}'],
                             f"{f / 1e9:.6g}", f"{pos[n, 0]:.6f}", f"{pos[n, 1]:.6f}", f"{array_z + pos[n, 2]:.6f}",
                             src_name[(cls, pol, f)]])
                job += 1
    with open(os.path.join(args.out, 'jobs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\r\n')
        w.writerow(['job', 'element', 'i', 'j', 'eep_class', 'pol', 'port', 'freq_ghz', 'x_m', 'y_m', 'z_m', 'source_file'])
        w.writerows(rows)

    manifest = {
        'design': os.path.abspath(args.design), 'D': G.D, 'F': F,
        'cst_frame': 'reflector vertex at the origin, opening toward +z; x azimuth (10 elements), y elevation (48)',
        'array_plane_z_m': array_z, 'vertex_z_in_focus_frame_m': z_vertex,
        'source_theta_axis': THETA_AXIS, 'source_phi_axis': PHI_AXIS,
        'element_index': 'n = j * 10 + i, i along x, j along y, as in dbf_cfm.py',
        'cst_model_array_center_m': [float(v) for v in centre], 'cst_time_convention_conjugated': bool(conj),
        'source_normalization': '1 W stimulated power, phase reference at the element centre, '
                                'CST time and field (peak/RMS) conventions kept',
        'beam_convention': 'beam (az, el) = plane wave propagating along (sin az, sin el, -cos); '
                           'in the CST reflector project the beam points to (-sin az, -sin el, +cos)',
        'freqs_ghz': [f / 1e9 for f in pick], 'pols': args.pols, 'n_jobs': len(rows),
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(rows)} jobs to {os.path.join(args.out, 'jobs.csv')}; array plane at z = {array_z:.4f} m "
          f"(rim at z = {float(G.reflector_surface(G.D / 2, 0.0, params, F)) - z_vertex:.4f} m)")


if __name__ == "__main__":
    main()
