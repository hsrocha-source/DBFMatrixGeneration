# DBFMatrixGeneration
Private repository for DBF matrix generation work: a shaped 15 m L-band reflector (f/D = 0.38)
with a 48 × 10 digital feed array, receive beams over ±15° elevation × ±3° azimuth.

| file | contents |
|---|---|
| `reflector_go.py` | geometry and conventions, Zernike-shaped surface, GO ray tracing, surface optimisation, GO reports (capture, −3 dB coverage, arrival angles), design JSON I/O, STL export |
| `reflector_po.py` | occasional physical-optics check: plane-wave PO currents, field at the element centres, capture / cross-pol / centroid per beam |
| `docs/dbf_matrix_procedure.md` | procedure for building the DBF matrix from CST embedded element patterns + PO, and where it is (and is not) valid |
| `tests/` | `pytest` regression tests |

## Quick start

```bash
pip install -r requirements.txt

python reflector_go.py --out design_fD0.38.json           # optimise (5000 epochs) and report
python reflector_go.py --params design_fD0.38.json --stl shaped_reflector.stl
python reflector_po.py --params design_fD0.38.json         # PO check at 1315 MHz
python reflector_po.py --params design_fD0.38.json --freq-mhz 1375 --samples-per-lam 6 --az3db
python -m pytest -q tests
```

In Colab or a notebook, import the modules instead:

```python
import reflector_go as rg, reflector_po as rp
params, F = rg.load_params("design_fD0.38.json")
x_ap, y_ap = rg.aperture_rays(60)
rg.print_capture_table(params, F, x_ap, y_ap)
rp.run_po_check(params, F, rg.scan_test_angles())
```

If you already have optimised coefficients from the old notebook, save them once with
`rg.save_params(params, 0.38 * rg.D, "design_fD0.38.json")`. The parametrisation is unchanged
(`z4` … `z22`, same Noll definitions and normalisation).
