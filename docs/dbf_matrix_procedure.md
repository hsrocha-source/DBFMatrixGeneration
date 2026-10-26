# DBF matrix generation for the shaped-reflector SAR: procedure and validity

Scope: one 48 × 10 feed array (pitch 0.0926 m) in the focal region of the shaped 15 m reflector
(f/D = 0.38), receive beams over ±15° elevation × ±3° azimuth, CST embedded element patterns for
17 representative elements at 1225 … 1375 MHz in 15 MHz steps.

## Bottom line

The chain you propose is the right one, and it is the standard way reflector-fed digital feed
arrays are designed:

> CST embedded element patterns → physical optics through the reflector → one secondary pattern
> per element (= receive steering vectors) → DBF weights.

It is valid under three conditions:

1. **The CST patterns are embedded** (all other ports terminated), **realized**, share one power
   normalization and have a known phase reference (§3).
2. **The "inversion" is not a plain inverse of the steering matrix.** For this system that inverse
   is ill-posed: the matrix has roughly 200 useful dimensions out of 480 (§6.1). Compute the weights as
   conjugate-field-matched (maximum-SNR) beams, then add only the constraints you need (nulls at
   ambiguity directions, sidelobe shaping) with a small or regularized solve (§6.2).
3. **The 17-element shortcut** and **the bulk delay between frequency samples** are accounted for
   (§4, §7).

The remaining assumptions hold comfortably here: reciprocity, element far field at the reflector,
PO for a 65λ reflector, and no feed–reflector multiple bounces. §8 lists each one with its numbers.

---

## 1. Geometry, numbers, conventions

Same conventions as `reflector_go.py` / `reflector_po.py`:

* Focus at the origin. Reflector $z = (x^2+y^2)/4F - F + \sum_j c_j Z_j(x,y)$, $D = 15$ m, $F = 5.70$ m.
* Array in the plane $z = 0$, facing the reflector ($-z$). Element $n = j\,N_x + i$, with
  $i = 0..9$ along $x$ (azimuth) and $j = 0..47$ along $y$ (elevation). Centre
  $\mathbf r_n = \big((i-4.5)\,d,\ (j-23.5)\,d,\ 0\big)$, $d = 0.0926$ m.
* Beam $(az, el)$: the incoming plane wave *propagates* along
  $\hat d = (\sin az,\ \sin el,\ -\sqrt{1-\sin^2 az-\sin^2 el})$.
  The target, and the transmit direction used below, is $\hat u = -\hat d$.
  So a positive az in the code means the target is on the $-x$ side. Keep this convention in GO,
  PO and DBF, or mirror beams get swapped.
* Time convention $e^{+j\omega t}$, outgoing waves $e^{-jkr}$.

| | 1225 MHz | 1300 MHz | 1375 MHz |
|---|---|---|---|
| λ | 0.245 m | 0.231 m | 0.218 m |
| d/λ | 0.38 | 0.40 | 0.42 |
| D/λ | 61 | 65 | 69 |
| F/λ | 23 | 25 | 26 |
| beamwidth of a uniform 15 m aperture (1.02 λ/D) | 0.95° | 0.90° | 0.85° |
| direction sampling λ/2D (§5.4) | 0.47° | 0.44° | 0.42° |

Other numbers used below:

* Element-to-reflector distance: 5.3 … 10.2 m (23 … 47 λ).
* The reflector rim is seen 67° off the array normal from the array centre, and up to 72° from the
  elevation-edge elements. Your "matched to ±70°" covers practically the whole reflector.
* Array blockage: 0.93 × 4.45 m = 4.1 m² of 177 m² aperture (2.3 %).

## 2. What the DBF matrix is

For one frequency and one receive polarization:

* Each element has its own receiver. A point target in direction $\hat u$ with complex amplitude $s$
  gives at the 480 element outputs

  $$\mathbf v = \mathbf a(\hat u)\, s + \mathbf n .$$

* $\mathbf a(\hat u)$ is the **steering vector**: $a_n(\hat u)$ is the output of element $n$ for a
  unit plane wave from $\hat u$ with the system polarization $\hat p$. Everything is in it: element
  pattern, mutual coupling, reflector, blockage.
* Beam $k$ is a weighted sum $y_k = \sum_n W_{nk}\, v_n$, i.e. $\mathbf y = \mathbf W^T \mathbf v$.
  **$\mathbf W$ (480 × K) is the DBF matrix**, one column $\mathbf w_k$ per beam direction $\hat u_k$.
* Sample the steering vectors on $M$ directions $\hat u_m$ and stack them as rows:
  $\mathbf A$ (M × 480), $A_{mn} = a_n(\hat u_m)$. The beam patterns on those directions are then
  just a matrix product:

  $$\mathbf B = \mathbf A\,\mathbf W, \qquad B_{mk} = \text{response of beam } k \text{ toward } \hat u_m .$$

* With uncorrelated receiver noise of equal power $\sigma^2$, beam $k$ has output noise
  $\sigma^2 \lVert \mathbf w_k\rVert^2$ and signal $|B_{kk}|^2 |s|^2$, so

  $$\mathrm{SNR}_k = \frac{|B_{kk}|^2\,|s|^2}{\sigma^2\,\lVert\mathbf w_k\rVert^2}.$$

  This one line decides between the weight options in §6: a big weight vector means a noisy beam.

So the job is: get $\mathbf A$ right (§3–5), then choose $\mathbf W$ (§6), per frequency (§7).

## 3. Step 1: embedded element patterns from CST

Driving port $n$ with all other ports terminated in $Z_0$ gives the **embedded element pattern**
(EEP) $\mathbf e_n(\hat r)$. It includes the coupling to the neighbours, which is exactly what the
receive steering vector needs (§5.1). Things to get right in the export:

1. **Complex fields, not gain.** Export $E_\theta, E_\phi$ with phase (a farfield source `.ffs`,
   or ASCII with phase columns).
2. **One normalization for all elements, to stimulated power** (realized pattern). The receivers see
   the port mismatch, so it belongs in $\mathbf a$. Check per element:
   $\frac{1}{2\eta}\oint |\mathbf e_n|^2\, d\Omega \,/\, P_\text{stim} = 1 - \sum_m |S_{mn}|^2$
   (minus ohmic loss). This checks the normalization and the coupling energy balance against the
   S-parameters in one go.
3. **Phase reference at each element's own centre.** The PO step places pattern $n$ at $\mathbf r_n$.
   If CST referenced the field to the global origin, de-embed it:
   $\mathbf e_n^\text{local}(\hat r) = \mathbf e_n^\text{global}(\hat r)\, e^{-jk\,\hat r\cdot\mathbf r_n}$.
   Verify the sign once by exporting one element with both origins. If the ratio comes out
   conjugated, CST's export uses the opposite time convention: conjugate all fields.
4. **Rotate into the reflector frame; never mirror.** In CST the array probably radiates toward
   $+z_c$; here it faces $-z$. Use a 180° rotation, e.g. about $x$:
   $(x, y, z)_c \to (x, -y, -z)$. Apply the same rotation to the field vectors and to the element
   positions (element row $j$ then becomes row $47-j$). A mirror instead of a rotation flips
   handedness and scrambles the polarization ports.
5. **Angular sampling of 1° or finer**, at least over the front hemisphere. Interior EEPs are smooth.
   Edge and corner EEPs ripple with a period of about λ/L ≈ 3° in elevation, from interference with the
   wave diffracted at the far end of the 4.4 m array. Interpolate Cartesian components
   $(e_x, e_y, e_z)$, not $(E_\theta, E_\phi)$: spherical components are singular at $\theta = 0$,
   and $\theta = 0$ points straight at the reflector vertex.

## 4. Step 2: from 17 simulated elements to 480 patterns

Each element takes the pattern of the simulated element of its class on the same side, placed at its
own position $\mathbf r_n$. This is why the phase reference must be local.

| class | pattern used | count |
|---|---|---|
| corners | itself (4 simulated) | 4 |
| perimeter, short (10-element) sides | mid-edge element of that side | 2 × 8 |
| perimeter, long (48-element) sides | mid-edge element of that side | 2 × 46 |
| ring-2 corners | itself (4 simulated) | 4 |
| ring 2, short sides | ring-2 mid-edge element of that side | 2 × 6 |
| ring 2, long sides | ring-2 mid-edge element of that side | 2 × 44 |
| interior, i = 2..7, j = 2..45 | centre element | 6 × 44 = 264 |
| **total** | | **480** |

All four corners, edges, etc. were simulated, so no mirroring is needed. Use the pairs as a check
instead: if the element is mirror-symmetric, opposite corners must be mirror images of each other.
How well they agree is the numerical noise floor of the CST results.

Where this is approximate, and how to tell:

* **The array is only 10 elements wide.** No element is more than 4 positions from a long edge, and
  the 88 "interior" elements in columns i = 2 and 7 are third from it. Their true EEP lies between
  the ring-2 and centre patterns. The difference shows mostly in the azimuth plane at wide angles,
  which is where the reflector rim is seen (60°–72°).
  *Check:* compare the ring-2 mid-edge EEP with the centre EEP inside the reflector cone
  (θ ≤ 72°). If they differ by less than about 0.5 dB / 5°, ring 3 is closer still and the
  assignment is fine. If not, simulate one ring-3 element (e.g. i = 2, j = 24).
* **Which beams care.** Edge-element patterns only matter for beams whose focal spot reaches the
  array edge: the scan extremes (±15° el, ±3° az), i.e. the critical ones. For conjugate-matched
  beams (§6.2a) a relative steering-vector error ε costs gain only to second order,
  ≈ var(ε): 0.5 dB / 5° random error ≈ 0.05 dB. For nulls the same error limits the null depth (§6.2c).
* **Terminations.** The EEPs belong to $Z_0$ terminations. If the LNA input impedance is far from
  $Z_0$, the EEPs change. That can be corrected with the full S-matrix, but only with all ports' patterns.

## 5. Step 3: steering vectors with PO (the "projection")

### 5.1 Why transmit patterns give the receive steering vector

For a passive, reciprocal antenna (no ferrites in the antenna path), the signal at port $n$ for a
unit plane wave from $\hat u$ with polarization $\hat p$ is

$$a_n(\hat u) = C\;\hat p\cdot\mathbf E^\text{sec}_n(\hat u),$$

where $\mathbf E^\text{sec}_n$ is the far field of the complete antenna (array + reflector) when port
$n$ transmits and all others are terminated: the *secondary* embedded element pattern. $C$ is the
same for all elements, so it drops out of the beam shapes (keep it for absolute gain). Isolated
element patterns would miss the coupling the receivers actually see. That is why the patterns
must be embedded ones.

### 5.2 Transmit-mode PO, per element and frequency

1. Field of element $n$ on reflector point $\mathbf p$. The element far field is valid there (§8):

   $$\mathbf E^\text{inc}_n(\mathbf p) = \mathbf e_n(\hat R)\,\frac{e^{-jkR}}{R}, \qquad
   \mathbf H^\text{inc}_n = \frac{1}{\eta}\,\hat R\times\mathbf E^\text{inc}_n, \qquad
   \mathbf R = \mathbf p - \mathbf r_n .$$

2. PO current on the lit side: $\mathbf J_n = 2\,\hat n\times\mathbf H^\text{inc}_n$.
3. Secondary far field toward $\hat u$ (common factor $e^{-jkr}/r$ dropped):

   $$\mathbf E^\text{sec}_n(\hat u) = -\frac{jk\eta}{4\pi}\int_S
   \big[\mathbf J_n - (\mathbf J_n\cdot\hat u)\,\hat u\big]\; e^{\,jk\,\hat u\cdot\mathbf p}\; dS .$$

   Blockage: leave out the reflector points whose ray toward $\hat u$ passes through the array
   rectangle (the transmit-mode twin of the shadow mask in the GO/PO code). The element's own direct
   radiation toward $\hat u$ is its back lobe, negligible with a ground plane.
4. $A_{mn} = \hat p\cdot\mathbf E^\text{sec}_n(\hat u_m)$.

### 5.3 The same thing in receive mode (one matrix product)

Equivalently, let $\mathbf J^\text{pw}_m$ be the PO current induced by the unit plane wave of
direction $m$. `reflector_po.py` already computes that current, shadow mask included. Then

$$A_{mn} \;\propto\; \int_S \mathbf J^\text{pw}_m(\mathbf p)\cdot\mathbf E^\text{inc}_n(\mathbf p)\; dS .$$

On a surface grid of $N_s$ points this is a single product, (M × 3N_s) · (3N_s × 480). The two forms
agree exactly at the specular points that dominate the integral, and to PO accuracy elsewhere. Use
whichever is faster.

Note that the current `reflector_po.py` check (field at the element centres) is **not** a steering
vector. It treats the elements as point probes, which ignores the element pattern and the coupling.
It is a focal-field check of the reflector design only.

### 5.4 Direction grid

* The secondary patterns are band-limited by the reflector size, so samples every λ/D in
  direction cosines are enough to interpolate them. Use λ/2D (≈ 0.42° at 1375 MHz), so that the
  synthesized patterns are also under control between samples.
* Cover the FOV plus a margin, and every direction whose response you want to control: range-ambiguity
  elevations, which can lie outside ±15°.
* ±3° × ±15° at λ/2D is about 15 × 70 ≈ 1000 directions per frequency.

## 6. Step 4: from A to W (the "inversion")

### 6.1 Why a plain inverse is not valid here

"Plain inversion" means picking K = 480 beam directions and setting $\mathbf W = \mathbf A^{-1}$:
each beam is 1 at its own direction and 0 at the other 479. Its least-squares version,
$\mathbf W = (\mathbf A^H\mathbf A)^{-1}\mathbf A^H$ with M > 480, fails the same way:

* **A has far fewer useful dimensions than 480.** A reflector of diameter D can form only about
  (FOV area in direction cosines) × (aperture area) / λ² independent beams inside the FOV. For
  ±3° × ±15° that is ≈ 160 at 1225 MHz, 180 at 1300 MHz and 200 at 1375 MHz. The array needs 480
  elements to cover the moving focal spot with a pitch fine enough for the steep rim angles, not
  because there are 480 independent beams. The remaining singular values of $\mathbf A$ are tiny,
  so $\mathbf A^H\mathbf A$ is singular for practical purposes.
* **Beams closer than a beamwidth cannot be orthogonal.** A 10 × 48 grid over the FOV spaces the
  beams 0.64°–0.67°, inside the ≈ 0.9° beamwidth. A null at the neighbour's centre needs
  super-directive weights: a huge $\lVert\mathbf w\rVert$, so huge noise gain and extreme sensitivity
  to model errors.

A quick scalar-PO toy model shows the size of this. It uses the plain f/D = 0.38 paraboloid at
1300 MHz, point elements with the cos^q pattern from `arrival_report` (−8 dB at 70°), the array
shadow, no polarization and no coupling:

| quantity (toy model, 1300 MHz) | value |
|---|---|
| singular values within 10 / 20 / 30 dB of the largest (952 directions at λ/2D) | 170 / 212 / 239 of 480 |
| degrees-of-freedom estimate above | 180 |
| condition number, square 480 × 480 (beams every 0.64°–0.67°) | ≈ 10¹⁷ |
| SNR loss vs. conjugate match, exact inverse | ≈ 300 dB (numerically meaningless) |
| SNR loss, Tikhonov α = (0.1 σ_max)² | 5.4 dB, and the neighbour response only improves from −4 to −6 dB |
| SNR loss, conjugate match + one null 1° away | 0.06 dB |
| SNR loss, conjugate match + four nulls at el = ±5°, ±10° | < 0.01 dB |

The shaped surface changes details, not the picture: the degrees-of-freedom count depends only on
D, λ and the FOV. Nulls are cheap where they make sense (a beamwidth or more away). Forcing every
beam to be orthogonal to its 0.6° neighbours is what breaks.

### 6.2 What to compute instead

These are still "inversions", but of small, well-conditioned problems.

**(a) Conjugate-field match (CFM): the baseline.**

$$\mathbf w_k = \frac{\mathbf a^*(\hat u_k)}{\lVert\mathbf a(\hat u_k)\rVert^2}$$

* Unit response at $\hat u_k$ and the maximum possible SNR for uncorrelated, equal receiver noise.
  No inversion at all.
* It conjugates whatever focal field the shaped reflector produces, so it corrects all phase
  aberrations automatically. Focus quality does not matter, only the captured power:
  $\mathrm{SNR}_k \propto \lVert\mathbf a(\hat u_k)\rVert^2 = \sum_n |a_n(\hat u_k)|^2$.
  This is why spill / capture was the right objective for the GO shaping, and why dropping the
  centroid and spread terms from the loss was fine.
* Pattern: about that of a uniformly illuminated aperture. Toy model at 1300 MHz, unshaped
  paraboloid: HPBW 0.92° and first sidelobe −18.4 dB at boresight; 1.56° and −16.6 dB at el = 15°,
  where the array catches less of the spot. A uniform 15 m aperture gives 0.90° and −17.6 dB.

**(b) Maximum SNR with a noise covariance.**

$$\mathbf w_k \propto \big(\mathbf R_n^{-1}\,\mathbf a(\hat u_k)\big)^*, \qquad
\mathbf R_n = E[\mathbf n\,\mathbf n^H].$$

$\mathbf R_n$ comes from the LNA noise parameters with the array S-matrix (noise coupling between
channels), plus the Earth's thermal noise seen through the antenna. With $\mathbf R_n = \sigma^2\mathbf I$
it is (a). Start with (a) and add this later.

**(c) Constrained beams (LCMV): nulls where they matter.** Stack the look direction and the null
directions, $\mathbf C = [\,\mathbf a(\hat u_k),\ \mathbf a(\hat u_{\text{null},1}),\ \dots]$
(480 × L), with responses $\mathbf g = (1, 0, \dots, 0)$. The lowest-noise weights meeting
$\mathbf C^T\mathbf w = \mathbf g$ are

$$\mathbf w_k = \mathbf C^*\,\big(\mathbf C^T\mathbf C^*\big)^{-1}\,\mathbf g .$$

* Only an L × L inverse, with L = 1 + number of nulls. With L = 1 this is (a).
* Typical nulls: the range-ambiguous elevations for each SCORE look angle, the nadir return.
* Cheap in SNR while the nulls are at least about one beamwidth from $\hat u_k$ (toy: < 0.1 dB).
* **Null depth is set by the model accuracy**, roughly 20 log₁₀ of the relative steering-vector
  error: 0.5 dB / 5° gives about −20 dB. Deeper nulls need a better model, or wider nulls
  (constrain a few neighbouring directions together).

**(d) Regularized least-squares pattern synthesis**, when you need a pattern *shape*: a sidelobe
envelope, a flat top, or a beamwidth kept constant over frequency. With $\mathbf d_k$ the desired
response on the M directions (optionally weighted per direction):

$$\mathbf w_k = \big(\mathbf A^H\mathbf A + \alpha\,\mathbf I\big)^{-1}\mathbf A^H\,\mathbf d_k .$$

* α trades pattern fit against noise. Choose it from a gain budget, e.g. the smallest α for which the
  SNR loss against CFM, $10\log_{10}\!\big(\lVert\mathbf w_k\rVert^2\lVert\mathbf a_k\rVert^2/|B_{kk}|^2\big)$,
  stays ≤ 0.5 dB.
* Do it through one SVD per frequency, $\mathbf A = \mathbf U\boldsymbol\Sigma\mathbf V^H$:
  $\mathbf w_k = \mathbf V\,\mathrm{diag}\!\big(\sigma_i/(\sigma_i^2+\alpha)\big)\,\mathbf U^H\mathbf d_k$.
  The singular values also show the effective rank directly.

**Recommended order:** CFM for all beams first. It cannot be beaten on SNR, and it validates the
whole chain (§9). Then LCMV nulls at the ambiguity directions. Use (d) only if the SAR
ambiguity / sidelobe budget asks for a shape that CFM + nulls cannot give.

### 6.3 SCORE beam grid and sparsity

* The elevation beam follows the echo, so you need weights on a fine elevation grid. The weights
  vary smoothly with look angle as the focal spot slides across the array. A 0.1° grid keeps the
  pointing loss between grid points below ≈ 0.04 dB (≈ 12 (δ/θ₃)² dB, δ = half a step), or
  interpolate the weights between coarser beams.
* Each beam only uses the elements under its focal spot. Keeping the strongest elements that hold
  99 % of $\sum_n |a_n|^2$ costs 0.04 dB, which gives a sparse $\mathbf W$ for the on-board processor.

## 7. Step 5: frequency

* Everything above is per frequency: $\mathbf W_q$ for $f_q = 1225 + 15q$ MHz, q = 0..10, and per
  polarization channel.
* **Bulk delay.** With the phase reference at the focus, every steering vector carries a common delay
  of ≈ 2F/c = 38 ns (aperture plane → reflector → focal plane). Samples every 15 MHz can only follow
  delays below 1/(2 · 15 MHz) = 33 ns, so the raw phases of $\mathbf W_q$ are ambiguous between
  frequency samples. Remove $e^{-j2\pi f\cdot 2F/c}$ (or each beam's mean delay) before interpolating
  in frequency, and put it back afterwards. What remains, the delay differences between elements
  under the focal spot, is a few wavelengths at most: far inside the 10 m path range that 15 MHz
  sampling resolves.
* **Narrowband or wideband DBF.** After removing the bulk delay, look at how much the phase of each
  strong weight changes across 1225–1375 MHz. If it stays within about ±10° (≈ 0.1 dB loss at the
  band edges), one weight set per beam plus one true-time delay per beam is enough. Otherwise use
  sub-band or FIR beamforming with $\mathbf W_q$ as the frequency samples. The focal spot scales with
  λ, so amplitudes also change by roughly 10 % across the band.
* **Polarimetry.** If the system is polarimetric, also compute the cross-pol matrix
  ($\hat p_\times\cdot\mathbf E^\text{sec}_n$) and check, or constrain as in (c), each beam's
  cross-pol response.

## 8. Validity checklist

| assumption | holds? | why / numbers |
|---|---|---|
| reciprocity: receive steering vector = transmit secondary EEP | yes | passive, reciprocal antenna; EEPs with Z₀ terminations; LNAs close to Z₀ |
| element far field used at the reflector | yes | R = 5.3–10.2 m (23–47 λ). An embedded element radiates mainly from a few cells (L ≈ 0.3–0.5 m, 2L²/λ ≈ 0.8–2 m). Weak long-range coupling (edge waves along the 4.4 m array) gets placed at the element, but it is weak |
| PO on the reflector | yes, main beam and near sidelobes | D = 61–69 λ, F = 23–26 λ. Edge diffraction (not in PO) matters for far sidelobes. Surface sampling λ/4–λ/6 as in `reflector_po.py` |
| feed–reflector multiple reflections | negligible | reflection back into an element ≈ G_e λ/(4πF) ≈ −43 dB (G_e ≈ 4πd²/λ² ≈ 2) |
| array blockage | include it | 2.3 % of the aperture: ≈ −0.2 dB and raised sidelobes. Mask as in §5.2 |
| struts / supports | not modelled | add them to the blockage mask if they exist |
| 17 → 480 patterns | approximate | §4. Check ring 2 against centre. Second-order gain effect for CFM; limits null depth |
| plain inversion of A | **no** | §6.1 |
| CFM / LCMV / regularized LS | yes | §6.2 |
| 11 frequencies, 15 MHz step | yes, after removing the bulk delay | §7 |
| direction grid | λ/2D in direction cosines | §5.4 |
| GO-optimized surface as input | yes | GO only shaped the surface. A must come from PO: GO has no finite spot size and no diffraction |
| model vs. hardware | needs calibration | W assumes perfectly calibrated receivers. Internal calibration fixes channel gain/phase; in-orbit calibration fixes the beam model. CFM tolerates model errors (≈ 0.05 dB for 0.5 dB / 5°); nulls do not (§6.2c) |

## 9. Checks to run once A exists

1. **Energy.** The CFM beam power $\sum_n|a_n(\hat u)|^2$ over the FOV should track the GO capture
   weighted by the element pattern (the `weighted` column of `arrival_report`) within about 1 dB.
   A large mismatch points to a normalization or phase-reference error.
2. **Pointing.** The CFM beam peak should sit on $\hat u_k$ within a small fraction of a beamwidth.
   Scanned beams can peak slightly off (toy: 0.04° at el = 15°). CFM maximizes the gain *at*
   $\hat u_k$, which is what the SAR needs.
3. **Symmetry.** Beams at (±az, ±el) should be mirror images if array and reflector are symmetric.
4. **Rank.** The singular values of A per frequency should show about 160–200 significant values.
5. **Independent PO check.** For one element at one frequency, compare the in-house PO secondary
   pattern with CST's own reflector solver fed by the same farfield source.
6. **Sensitivity.** Perturb A randomly (e.g. 0.5 dB / 5°), keep the weights computed from the
   unperturbed A, and look at the gain loss and the null depth.
