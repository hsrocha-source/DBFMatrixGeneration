# DBFMatrixGeneration
Private repository for DBF matrix generation work: a shaped 15 m L-band reflector (f/D = 0.38)
with a 48 × 10 digital feed array, receive beams over ±15° elevation × ±3° azimuth.

| file | contents |
|---|---|
| `reflector_go.py` | geometry and conventions, Zernike-shaped surface, GO ray tracing, surface optimisation, GO reports (capture, −3 dB coverage, arrival angles), design JSON I/O, STL export |
| `reflector_po.py` | occasional physical-optics check: plane-wave PO currents, field at the element centres, capture / cross-pol / centroid per beam |
| `dbf_cfm.py` | first DBF matrix: CST embedded element patterns (.ffs) + PO on the reflector (GPU via JAX) -> steering matrices and conjugate-field-matched weights per frequency and polarization |
| `cst_io.py` | readers for CST farfield sources (.ffs), Touchstone (.sNp) and the port map; `python cst_io.py FILE` inspects a file |
| `cst/` | CST export macro and the port map of the 10 x 8 EEP model |
| `designs/` | optimised reflector design (JSON) and its STL |
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

## DBF matrix (conjugate-field match)

```bash
pip install -U "jax[cuda12]"                               # GPU; plain "jax" runs on the CPU
python cst_io.py EEP_ffs/farfield_f1.3_port71.ffs model.s160p   # check one file of each kind first
python dbf_cfm.py --design designs/design_fD0.38.json --ffs-dir EEP_ffs \
    --touchstone model.s160p --out dbf_out
```

* `--cst-array-center X Y Z`: CST coordinates [m] of the centre of the simulated 10 x 8 array
  (default 0 0 0). Used to de-embed each pattern to its own element; the summary's phase-centre
  table shows whether it is right (offsets of a few mm are fine).
* Beam grid: `--az-max 3 --az-step 0.5 --el-max 15 --el-step 0.1` (defaults, 3913 beams).
* `--samples-per-lam 4` (PO surface sampling), `--chunk`, `--beam-block` (GPU memory).
* The S-parameters only feed the power-balance report.

Outputs per frequency, `dbf_out/dbf_cfm_<f>MHz.npz`:

| key | shape | meaning |
|---|---|---|
| `W_H`, `W_V` | (480, K) | unit-norm CFM weights, beam k = column k; beam output y_k = sum_n W[n, k] v_n |
| `A_H`, `A_V` | (K, 480) | co-pol steering vectors at the beam directions [V for 1 W stimulated] |
| `gain_H_dBi`, `gain_V_dBi` | (K,) | CFM beam gain (uncorrelated, equal receiver noise) |
| `xpol_H_dB`, `xpol_V_dB` | (K,) | cross-pol / co-pol response of each beam |
| `n99_H`, `n99_V` | (K,) | number of elements holding 99 % of the beam power |
| `beam_az`, `beam_el` | (K,) | beam directions [deg], k = i_el * n_az + i_az, conventions of `reflector_go.py` |
| `element_xyz`, `element_class` | (480, 3), (480,) | flight-array element n = j * 10 + i: position (reflector frame) and EEP class used |

Plus `summary.txt` (all checks) and PNG plots (gain maps, weights, pattern cuts, gain vs el).
Test without CST data: `python tests/dummy_cst.py dummy_ffs` writes a synthetic data set.

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
