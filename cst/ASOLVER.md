# CST asymptotic-solver comparison

Computes the secondary pattern of every flight-array element through the reflector with CST's
asymptotic solver, as an independent check of the PO steering matrices in `dbf_cfm.py`.

## Running

1. `python cst/make_asolver_jobs.py --ffs-dir EEP_ffs --out asolver`
   (defaults: 1.3 GHz, H and V, all 480 elements = 960 jobs; `--freqs-ghz`, `--pols`,
   `--elements 0-9,236` to change). Copy the `asolver` folder next to `shaped_reflector.cst`.
   The Colab notebook (`DBF_colab.ipynb`, section 9) does this step too and puts the folder and a
   `.zip` of it in `DBF_generation/asolver_<date>_<time>` on Drive, with the macro included;
   rename it to `asolver` next to the CST project.
2. In `shaped_reflector.cst`: units m / GHz, reflector vertex at the origin opening toward +z, PEC,
   asymptotic solver settings chosen once in the GUI. The array plane is at
   `array_plane_z_m` of `manifest.json` (5.6062 m for the committed design).
3. Run `cst/asolver_secondary_patterns.bas` with `MAX_JOBS = 2` first, check `asolver/results`,
   then with `MAX_JOBS = 0`. Finished jobs are skipped, so it can be stopped and restarted.

The macro has not been run here (no CST). If a call fails in your CST version, compare it with the
History List entry CST writes when you do the same step by hand: defining a farfield source
(`Setpos`, `Settheta` = where the source's +z points, `Setphi` = where its +x points), starting the
asymptotic solver (`ASolver.Start`), and the farfield export (`Step` / `Step2` = theta / phi step).

## Files

| file | contents |
|---|---|
| `jobs.csv` | `job, element, i, j, eep_class, pol, port, freq_ghz, x_m, y_m, z_m, source_file`; element n = j * 10 + i as in `dbf_cfm.py` |
| `sources/eep_<class>_<pol>_f<GHz>.ffs` | CST pattern of that class, phase reference moved to its element centre, 1 W stimulated power, CST's own conventions |
| `manifest.json` | design, array-plane z, source orientation, CST array centre and time convention found from the patterns, normalization |
| `results/job<N>.ffs` | A-solver far field of job N: CST farfield-source format, theta from +z, phase reference at the vertex (CST origin), theta <= 20 deg kept |
| `results/log.txt` | run time per job |

## Reading the results (for the comparison script)

For job N (element n, polarization p, frequency f), the steering-matrix entry for beam (az, el) is
the co-polar component of the result field in the beam direction, brought to the conventions of
`dbf_cfm.py`:

* beam direction in the CST project: u = -d_in = (-sin az, -sin el, +sqrt(1 - sin^2 az - sin^2 el)),
  i.e. theta = arccos(u_z), phi = atan2(u_y, u_x);
* co-polar vector: `pol_vectors(d_in)` of `dbf_cfm.py` (x-like for H, y-like for V);
* phase reference: `dbf_cfm.py` uses the focus, CST the vertex, so multiply by
  exp(-j k u_z array_plane_z_m) (e^{+jwt}; conjugate the CST fields first if
  `cst_time_convention_conjugated` is true);
* amplitude: the sources carry 1 W stimulated power; check once whether CST rescales the result
  (compare one element's peak with `A_H` of the matching `dbf_cfm_<f>MHz.npz`).

Stacking all elements gives A (beams x 480); `cfm_weights` in `dbf_cfm.py` then gives the same
weights, gains and plots as the GPU run.
