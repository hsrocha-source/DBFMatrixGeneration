# CST port map

`port_map.csv` lists the 80 elements of the CST model (10 columns × 8 rows, 160 ports) and marks
the 17 elements whose embedded element patterns were exported.

## Conventions (as in the CST model)

* View from +z looking toward −z; the array radiates toward +z. Column 1 is the left (−x) side,
  row 1 the bottom (−y) side.
* Each element has two ports: odd = H, even = V. Element `e` → ports `2e−1` (H) and `2e` (V).
  Elements are numbered row by row from the bottom-left, 10 per row:
  `col = (e−1) mod 10 + 1`, `row = (e−1) div 10 + 1`.
  This numbering reproduces all 17 port pairs given for the exported elements.
* `x_index` / `y_index` are 0-based (`col−1`, `row−1`).
* `eep_class`: `corner_*`, `edge_*`, `n1_*` (one element in from that edge or corner), `centre`;
  B / T / L / R = bottom / top / left / right. Empty means the port was not excited.
* Which physical direction H and V correspond to (x or y) is not recorded here. Add it once confirmed.

## Simulated elements

| class | ports (H,V) | col | row |
|---|---|---|---|
| corner_BL | 1, 2 | 1 | 1 |
| edge_B | 11, 12 | 6 | 1 |
| corner_BR | 19, 20 | 10 | 1 |
| n1_corner_BL | 23, 24 | 2 | 2 |
| n1_edge_B | 31, 32 | 6 | 2 |
| n1_corner_BR | 37, 38 | 9 | 2 |
| edge_L | 61, 62 | 1 | 4 |
| centre | 71, 72 | 6 | 4 |
| n1_edge_L | 83, 84 | 2 | 5 |
| n1_edge_R | 97, 98 | 9 | 5 |
| edge_R | 99, 100 | 10 | 5 |
| n1_corner_TL | 123, 124 | 2 | 7 |
| n1_edge_T | 131, 132 | 6 | 7 |
| n1_corner_TR | 137, 138 | 9 | 7 |
| corner_TL | 141, 142 | 1 | 8 |
| edge_T | 151, 152 | 6 | 8 |
| corner_TR | 159, 160 | 10 | 8 |

## Relation to the 48 × 10 flight array

* The CST model has 10 columns like the real array (azimuth, x) but only 8 rows instead of 48
  (elevation, y). Its patterns are used for the full array with the class assignment of
  `docs/dbf_matrix_procedure.md` §4: the real array's left/right long sides take the `edge_L/R` and
  `n1_edge_L/R` patterns, its bottom/top short sides the `*_B/T` patterns, and its interior the
  `centre` pattern.
* With 8 rows, the `centre`, `edge_L/R` and `n1_edge_L/R` elements are only 3–4 rows from the
  top/bottom edges. In the real array they stand for elements up to ~23 rows away. Their patterns
  therefore still carry some short-edge truncation in the elevation plane. Compare `centre` with
  `n1_edge_B/T` in that plane to see how large that effect is.
* When rotating into the reflector frame ((x, y, z) → (x, −y, −z), doc §3), bottom and top swap:
  CST row 1 becomes the reflector-frame top row.
