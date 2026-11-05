import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'cst'))
import dbf_cfm as DB  # noqa: E402
import dummy_cst  # noqa: E402
import make_asolver_jobs as MJ  # noqa: E402
import reflector_go as G  # noqa: E402
from cst_io import read_ffs, read_port_map, write_ffs  # noqa: E402


@pytest.mark.parametrize('convention', ['+jwt', '-jwt'])
def test_sources_match_pipeline_patterns(tmp_path, convention):
    centre = np.array([4.5, 3.5, 0.0]) * G.ELEMENT_PITCH
    ffs = dummy_cst.make_dataset(str(tmp_path / 'ffs'), [1300e6], step_deg=3.0, convention=convention, cst_center=centre)
    out = tmp_path / 'asolver'
    MJ.main(['--ffs-dir', ffs, '--out', str(out), '--elements', '0-3,236'])

    man = json.loads((out / 'manifest.json').read_text())
    assert man['cst_time_convention_conjugated'] == (convention == '-jwt')
    rows = list(csv.DictReader(open(out / 'jobs.csv', newline='')))
    assert len(rows) == 5 * 2 and len(list((out / 'sources').glob('*.ffs'))) == 17 * 2

    params, F = G.load_params(os.path.join(ROOT, 'designs', 'design_fD0.38.json'))
    pos = np.asarray(G.element_positions(G.array_frame(params, F))[0])
    for r in rows:
        n = int(r['element'])
        np.testing.assert_allclose([float(r['x_m']), float(r['y_m'])], pos[n, :2], atol=1e-6)
        assert abs(float(r['z_m']) - man['array_plane_z_m']) < 1e-6 and abs(man['array_plane_z_m'] - 5.6062) < 1e-3
        assert r['eep_class'] == DB.flight_array_classes()[n]

    # each written source, read back and brought to e^{+jwt}, equals the pipeline's de-embedded pattern
    pm = read_port_map(dummy_cst.PORT_MAP)
    classes = [c for c in pm if c['eep_class']]
    layout = np.array([DB.cst_element_centre(c['col'], c['row'], 10, 8, (0, 0, 0)) for c in classes])
    eeps = DB.EEPIndex(ffs, {c[f'port_{p}'] for c in classes for p in ('H', 'V')})
    cst_pos = layout + np.array(man['cst_model_array_center_m'])
    tab_H, tab_V, grid, _ = DB.pattern_table(eeps, classes, cst_pos, 1300e6, man['cst_time_convention_conjugated'], 1.0)
    n_grid = grid[2] * grid[5]
    for c, cls in enumerate(classes):
        for pol, tab in (('H', tab_H), ('V', tab_V)):
            src = read_ffs(out / 'sources' / f"eep_{cls['eep_class']}_{pol}_f1.3GHz.ffs")
            assert src.p_stim[0] == 1.0 and np.allclose(src.position, 0.0)
            e, _, _ = DB.field_cartesian(src, 0)
            if man['cst_time_convention_conjugated']:
                e = np.conj(e)
            ref = np.asarray(tab[c * n_grid:(c + 1) * n_grid]).reshape(e.shape)
            np.testing.assert_allclose(e, ref, atol=2e-6 * np.abs(ref).max())


def test_parse_elements():
    assert MJ.parse_elements('0-2,10,478-479', 480) == [0, 1, 2, 10, 478, 479]
    assert len(MJ.parse_elements('all', 480)) == 480
    with pytest.raises(ValueError):
        MJ.parse_elements('470-480', 480)


def test_trimmed_result_file_parses(tmp_path):
    """The macro keeps theta <= 20 deg of the exported result; such a file must still read."""
    th, ph = np.round(np.arange(0, 20.0001, 0.2), 6), np.arange(0, 360.0001, 1.0)
    et = np.exp(1j * np.radians(th))[:, None] * np.cos(np.radians(ph))[None, :]
    write_ffs(tmp_path / 'job0.ffs', th, ph, et, 0 * et, 1.3e9, 0.4, 0.45, 1.0)
    src = read_ffs(tmp_path / 'job0.ffs')
    assert src.e_theta.shape == (1, len(th), len(ph) - 1)
    assert np.isnan(src.integrated_power(0))                  # partial sphere: no power check
