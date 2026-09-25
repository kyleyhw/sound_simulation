# Pose-robust sensing: rigid pose error, refinement, autofocus and jitter training (follow-up to 6.6.1-6.6.2) — 2026-09-25

**Headline.** The 6.6.1 error model (every speaker and mic displaced
independently) overstates the damage. A device with a known internal layout
is misplaced by one rigid transform per placement. Under that model, at a
realistic careful hand placement (σ_t = 2.5 cm per axis, σ_θ = 2°), the
aligned U-Net falls from **0.362** to **0.203**. Refining each pose against
the empty-box model with 3 unknowns, then polishing each device within one
cell, brings it back to **0.324** (−0.039 ± 0.004 vs exact poses, z = −10;
+0.222 vs the no-audio prior, z = 24). That recovers **76 %** of the loss,
against 66 % for the 6.6.2 per-device joint fit (paired +0.015 ± 0.003,
z = 4.5). Translation alone is almost free to fix: with σ_t = 2.5 cm and no
rotation, the rigid fit reaches 0.350, within 0.012 ± 0.002 (z = −5) of exact
poses. **Rotation is what costs.** Image-domain autofocus (cross-pose
coherence, the SAR sharpness metric) is a **negative result**: it adds only
+0.017 (z ≈ 4) at the realistic rigid errors, nothing under independent
errors, and does not move the poses. Training the U-Net on pose-jittered
images gains **+0.01 to +0.08** (+0.044 at the realistic rigid level)
without any test-time search. But it costs 0.034 at exact poses (z = −8) and
is 0.015-0.025 *worse* than the exact-pose U-Net once the poses are refined
at any rigid level.

**Verdict for a real laptop.** Treat pose error as rigid. Measure the
orientation carefully, because a rigid translation is recovered almost
exactly. Before imaging, refine every placement against the known-room
model: the rigid fit plus a one-cell per-device polish, or the 6.6.2 joint
fit, which is within 0.015 of it. Then feed the images to the U-Net trained
on *exact* poses. Use the jitter-trained U-Net only when no refinement can
run, for example on a tight on-device budget. Do not use autofocus. All of
this assumes the outer room is known (here the empty box; in a real room,
the room twin from the wall echoes). With 500 held-out rooms, K = 4, nothing
was tuned on held-out rooms (one caveat on the polish variant, below).

Figures and numbers: `pose_robust_2026_09_25_artifacts/` (`results.json`,
`pose_robust_iou.png`, `pose_error_vs_iou.png`, `examples_rigid_1_2.png`).
Maths: `docs/imaging.md` §10.

## What was built

| file | content |
|---|---|
| `src/acoustic_system/imaging/pose_robust.py` | rigid error model (`perturb_poses_rigid`), rigid candidate grid, candidate scoring (empty-box misfit: L2, Huber, quiet-start; per-pose back-projection and carving focus images), rigid least squares, autofocus (coherence / entropy, prior and data terms, coordinate ascent), rigid + per-device polish |
| `scripts/eval_pose_robust.py` | stages `dev`, `augdata`, `augtrain`, `held`, `polish`, `score`, cached under `$TMPDIR/acoustic_imaging_models_cache/pose_robust` |
| `tests/imaging/test_pose_robust.py` | 7 tests (~1 s): rigidity, candidate grid, Huber, exact recovery of a translated pose, zero misfit at the truth, autofocus aligning shifted images |
| `docs/imaging.md` §10 | the maths (appended) |

No existing file was changed except the append to `docs/imaging.md`.

## The error model and its magnitudes

**Rigid model.** For each pose (one source cell and the 2-mic pair) draw
t ~ N(0, σ_t² I) in cells and θ ~ N(0, σ_θ²) in degrees. Rotate the pose
about its centroid, translate it, and round to cells (clipped to the
interior). The recordings stay at the true cells. The empty-box background
and every travel time use the assumed cells, as in 6.6.1. Rounding keeps
each inter-device distance to within a cell (unit-tested).

**Magnitudes at 2.5 cm per cell** (`simulation/units.py` LAPTOP_ROOM):

| label | σ_t | σ_θ | stands for |
|---|---|---|---|
| rigid (0.5, 1°) | 0.5 cell = 1.25 cm | 1° | placement on measured tape marks, or a phone VIO pose (ARKit/ARCore absolute errors of 1.5-4 cm [16]) |
| rigid (1, 2°) | 1 cell = 2.5 cm | 2° | careful hand placement, squared to a mark: the realistic default |
| rigid (2, 5°) | 2 cells = 5 cm | 5° | casual placement |
| rigid (1, 0°) / (0, 3°) | — | — | translation only / rotation only (decomposition) |
| indep σ = 0.5, 1, 2 | — | — | the 6.6.1 model (same random draws; its numbers are reproduced exactly) |

**The lever-arm caveat.** The v2 generator places the source and the mic
pair independently (median source-mic distance 29 cells = 73 cm), so the
RMS device distance from the pose centroid is 16.3 cells. A rotation of 2°
therefore moves a device by 0.57 cells RMS here, about three times more
than on a 20-30 cm laptop. The rotation results are pessimistic for a real
laptop, and the translation results carry over.

**Device error before correction** (mean Euclidean error per device):
0.65 / 1.37 / 2.80 cells for the three rigid levels, against 0.59 / 1.26 /
2.50 for independent σ = 0.5 / 1 / 2. At a similar device error, rigid
error hurts less than independent error. With the exact-pose U-Net, rigid
(1, 2°) (1.37 cells) costs −0.159 ± 0.008. Independent σ = 1 (1.26 cells)
costs −0.218 ± 0.009. A pure translation of the same size (1.26 cells)
costs only −0.090 ± 0.006. What breaks the chain is a change of the pose's
*internal* geometry (source-mic distances). Rotation plus rounding does
that, independent jitter does it maximally, and an integer translation does
not do it at all. Under a translation the direct wave is reproduced exactly
by the empty-box model away from the walls, so only the wall echoes and the
travel times are wrong.
## Literature, and what each line implies here

| line of work | key references | idea | implication for this setup |
|---|---|---|---|
| SAR / SAS autofocus (image-domain) | Fienup 2000 [1]; Fienup & Miller 2003 [2]; Kragh & Kharbouch 2006 [3]; Wahl et al. 1994 (PGA) [4]; Gerg & Monga 2020 [5] | Estimate the platform-position (phase) errors from the data alone by maximising a sharpness metric (sum of a convex point function of intensity, e.g. $\sum I^2$) or minimising the image entropy. PGA instead estimates the phase-error gradient from strong isolated scatterers. | SAR has hundreds of pulses per aperture, each contributing a small, coherent, phase-only perturbation, and a scene of many point-like scatterers. Here K = 4 poses give four bistatic, heavily smeared ellipse images, and a pose error changes the *background subtraction* (the direct wave leaks into the residual), not just a phase. The quadratic sharpness of the sum of unit-norm pose images is exactly their pairwise coherence, so it is cheap to evaluate over a candidate grid. PGA needs isolated dominant point scatterers, which a room of extended p = 0 obstacles does not have, so it was not tried. |
| SAS micronavigation (data-domain) | Bellettini & Pinto 2002 [6] | Displaced-phase-centre / redundant-phase-centre: correlate overlapping pings to estimate sway and yaw ping to ping, down to microns. | Needs overlapping, redundant apertures between consecutive pings. Four independent hand placements share no redundant phase centres, so there is nothing to correlate directly; the closest analogue is fitting each pose's data to a model (the empty-box fit below). |
| Echo-based room geometry and self-localisation | Dokmanić et al. 2013 [7]; Kreković, Dokmanić & Vetterli 2016 (EchoSLAM) [8]; Evers & Naylor 2018 (acoustic SLAM) [9] | First-order wall echoes (image sources) constrain the room and the device jointly; EchoSLAM alternates room and trajectory estimates for a co-located source/mic and proves convergence; acoustic SLAM tracks the observer pose probabilistically together with the map. | The outer box is known here, so its echoes are the "map" that pins the pose, and pose-then-map alternation is what `pose_refine.py` already does with the empty box. The key lesson of EchoSLAM is that *rigid* constraints make the joint problem well posed: one pose = 3 unknowns, not 2(1 + M). The 6.6.2 oracle check (93 % of devices exact with the true map in the model) is the EchoSLAM regime; the imagers' maps are too poor to close the loop. |
| Microphone-array self-calibration | Plinge et al. 2016 [10]; Kowalk, Doclo & Bitzer 2022 [11] | Calibration methods are classified by what they measure (ToA, TDoA, DoA, diffuse-field coherence). When each node is an array of *known internal geometry*, only its position and orientation are estimated. Learned DoA estimators trained on one geometry degrade on another unless the geometry is an input. | Model the error of a device with known geometry as a rigid transform per placement and search 3 DOF per pose; this is the "known sub-array geometry" case. It also motivates the rigid error model of this report. |
| Robust estimation | Huber 1964 [12]; Fischler & Bolles 1981 (RANSAC) [13] | Down-weight or reject samples the model cannot explain. | In the empty-box fit the scattered arrivals are outliers for the pose model. A Huber loss and the quiet-start cost of `pose_refine.py` were tried as the robust variants; RANSAC over time windows was not (the scattered onset is only ~4 lags after the direct arrival, so there is no clean inlier window to sample). |
| Training with perturbed nuisance parameters | Tobin et al. 2017 (domain randomisation) [14]; position-uncertainty injection, e.g. Jiang & Choueiri 2026 [15] | Randomise the nuisance in simulation so the network learns to be invariant to it, instead of estimating it. | Cheapest possible fix: image each training room at randomly perturbed poses (both error models, three magnitudes) and train the same U-Net on exact + perturbed images. No test-time search. |
| Pose accuracy that is realistic | ARKit/ARCore/T265/ZED 2 benchmark (Kim et al. 2022) [16]; Apple ARKit world-tracking notes [17] | Phone VIO absolute pose errors of about 1.5-4 cm indoors. | Sets the translation scale: 1.25 cm (measured marks) to 5 cm (casual placement) per axis, i.e. 0.5-2 cells at 2.5 cm/cell. |

References

1. J. R. Fienup, "Synthetic-aperture radar autofocus by maximizing sharpness," *Opt. Lett.* 25(4), 221-223 (2000). https://opg.optica.org/ol/abstract.cfm?uri=ol-25-4-221
2. J. R. Fienup and J. J. Miller, "Aberration correction by maximizing generalized sharpness metrics," *JOSA A* 20(4), 609-620 (2003). https://labsites.rochester.edu/fienup/wp-content/uploads/2019/07/JOSAA03_GenSharpness.pdf
3. T. J. Kragh and A. A. Kharbouch, "Monotonic iterative algorithm for minimum-entropy autofocus," ASAP Workshop (2006). https://www.researchgate.net/publication/228804600_Monotonic_iterative_algorithm_for_minimum-entropy_autofocus
4. D. E. Wahl, P. H. Eichel, D. C. Ghiglia, C. V. Jakowatz, "Phase gradient autofocus — a robust tool for high resolution SAR phase correction," *IEEE Trans. AES* 30(3), 827-835 (1994). https://ui.adsabs.harvard.edu/abs/1994ITAES..30..827W
5. I. Gerg and V. Monga, "Deep Autofocus for Synthetic Aperture Sonar," arXiv:2010.15687 (2020; withdrawn 2021). https://arxiv.org/abs/2010.15687
6. A. Bellettini and M. A. Pinto, "Theoretical accuracy of synthetic aperture sonar micronavigation using a displaced phase-center antenna," *IEEE J. Oceanic Eng.* 27(4), 780-789 (2002). https://ieeexplore.ieee.org/document/1134178/
7. I. Dokmanić, R. Parhizkar, A. Walther, Y. M. Lu, M. Vetterli, "Acoustic echoes reveal room shape," *PNAS* 110(30), 12186-12191 (2013). https://www.pnas.org/doi/abs/10.1073/pnas.1221464110
8. M. Kreković, I. Dokmanić, M. Vetterli, "EchoSLAM: Simultaneous localization and mapping with acoustic echoes," ICASSP 2016, 11-15. https://infoscience.epfl.ch/entities/publication/a8fe5625-3872-4848-a0bb-1340ca06cf68
9. C. Evers and P. A. Naylor, "Acoustic SLAM," *IEEE/ACM TASLP* 26(9) (2018). https://dl.acm.org/doi/10.1109/TASLP.2018.2828321
10. A. Plinge, F. Jacob, R. Haeb-Umbach, G. A. Fink, "Acoustic microphone geometry calibration: an overview and experimental evaluation of state-of-the-art algorithms," *IEEE SPM* 33(4), 14-29 (2016). https://www.semanticscholar.org/paper/Acoustic-Microphone-Geometry-Calibration:-An-and-of-Plinge-Jacob/3213ea56b0637fb53ff4c924e2a5a4d7a64f4edf (code: https://github.com/Plinge/audiogeocal)
11. U. Kowalk, S. Doclo, J. Bitzer, "Geometry-aware DoA estimation using a deep neural network with mixed-data input features," arXiv:2212.04788 (2022). https://arxiv.org/abs/2212.04788
12. P. J. Huber, "Robust estimation of a location parameter," *Ann. Math. Statist.* 35(1), 73-101 (1964). https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full
13. M. A. Fischler and R. C. Bolles, "Random sample consensus," *Comm. ACM* 24(6), 381-395 (1981). https://dl.acm.org/doi/10.1145/358669.358692
14. J. Tobin et al., "Domain randomization for transferring deep neural networks from simulation to the real world," IROS 2017. https://arxiv.org/abs/1703.06907
15. H. Jiang and E. Choueiri, "Neighbor-consistent neural filters for robust personal sound zones under localization uncertainty," arXiv:2605.21891 (2026). https://arxiv.org/abs/2605.21891
16. P. Kim, J. Kim, M. Song, Y. Lee, M. Jung, H.-G. Kim, "A benchmark comparison of four off-the-shelf proprietary visual-inertial odometry systems," *Sensors* 22(24), 9873 (2022). https://doi.org/10.3390/s22249873
17. Apple, "Understanding world tracking" (ARKit documentation). https://developer.apple.com/documentation/arkit/understanding-world-tracking

## Methods tried

Each correction searches, per pose, the same rigid candidate set: every
integer translation within ⌈2σ_t⌉ cells and rotations of {−2, −1, 0, 1, 2}·σ_θ
about the assumed centroid, rounded and de-duplicated (22 / 96 / 349
candidates per pose on average at the three rigid levels). For the
independent conditions the rotation step is the angle that moves a device
by σ at the RMS lever arm. Each candidate is scored once: one engine run per
distinct source cell gives its empty-box recording, then its misfit and its
two focus images.

1. **Rigid least squares (3 DOF per pose)**, from microphone-array
   calibration with known sub-array geometry [10] and the rigidity that
   makes EchoSLAM well posed [8]. The robust variants (Huber, and the
   quiet-start cost of `pose_refine.py`, which treat scatter as outliers
   [12]) tied with plain L2 on validation rooms, so L2 was used.
2. **Autofocus**, from SAR/SAS [1-4]. It maximises the coherence
   ‖Σ_k Ĩ_k‖² of unit-norm, zero-mean per-pose carving images (the
   quadratic sharpness metric) minus 0.2 × the Gaussian prior penalty, by
   coordinate ascent. It uses no model of the obstacles.
3. **Autofocus + rigid**: the same ascent plus 3 × the log misfit
   (back-projection and carving features).
4. **Rigid + polish**: rigid least squares, then the 6.6.2 per-device
   joint fit within one cell of each refined device. This absorbs the
   rotation grid step and the rounding.
5. **Pose-jitter augmentation** [14, 15]. The same U-Net is retrained for
   6 epochs on the exact images of training rooms 500-3999 plus four copies
   imaged at poses perturbed by a random member of {independent σ = 0.5, 1,
   2; rigid (0.5, 1°), (1, 2°), (2, 5°)}. Its threshold comes from
   validation rooms perturbed by the same mixture.
6. **Baselines**: no correction, and the 6.6.2 per-device joint least
   squares (window ⌈2σ_dev⌉, with σ_dev the per-coordinate device error
   SD of the model).

Every correction is scored with both networks.

**Settings (validation rooms 0-39 of the training archive, conditions
rigid (1, 2°), rigid (2, 5°) and independent σ = 1; `dev.json`).** The
mean U-Net IoU over the three conditions:

| variant | mean IoU (3 dev conditions) |
|---|---|
| no correction | 0.172 |
| rigid L2 / Huber / quiet-start | **0.224** / 0.224 / 0.222 |
| rigid L2 with a MAP prior penalty | 0.182 |
| autofocus, best (carving, coherence, prior 0.2) | 0.181 |
| autofocus, entropy metric, best | 0.166 |
| autofocus + rigid, best (both channels, L2, weight 3) | **0.235** |

**Protocol caveat (the polish variant).** Methods 1-3 and 5 and all
settings were fixed before any held-out room was scored. Method 4 was added
*after* a preliminary held-out score showed that the rigid fit lost to the
per-device joint fit under rotation. Its only parameter (polish radius 1)
was set a priori, not tuned. On 30 validation rooms it did *not* beat the
joint fit (0.279 vs 0.282 at (1, 2°); 0.247 vs 0.258 at (2, 5°)). Its
held-out edge over the joint fit (+0.015, z = 4.5 and +0.008, z = 2.0) is
therefore reported as "on par with the joint fit", not as a tuned win.

## Results (held-out, 500 rooms, K = 4, filled mask)

Exact poses: U-Net **0.362 ± 0.008**; jitter-trained U-Net 0.328 ± 0.007
(−0.034 ± 0.004, z = −8.1); no-audio prior 0.101 ± 0.003. Thresholds come
from validation rooms: U-Net 0.239 (exact poses), jitter U-Net 0.160
(perturbed validation rooms).

**Headline** (U-Net IoU; the recovered fraction of the IoU lost to pose
error is in brackets):

| condition | no correction | jitter U-Net, no correction | autofocus | rigid LS | autofocus + rigid | joint LS (6.6.2) | **rigid + polish** |
|---|---|---|---|---|---|---|---|
| rigid (0.5, 1°) | 0.280 | 0.290 | 0.284 (5 %) | 0.319 (47 %) | 0.322 (51 %) | 0.318 (47 %) | **0.329 (60 %)** |
| rigid (1, 2°) | 0.203 | 0.247 | 0.220 (11 %) | 0.295 (58 %) | 0.296 (59 %) | 0.308 (66 %) | **0.324 (76 %)** |
| rigid (2, 5°) | 0.143 | 0.184 | 0.160 (8 %) | 0.245 (47 %) | 0.244 (46 %) | 0.281 (63 %) | **0.290 (67 %)** |
| rigid (1, 0°) translation only | 0.272 | 0.284 | 0.287 (16 %) | **0.350 (86 %)** | **0.350 (86 %)** | 0.320 (53 %) | 0.323 (57 %) |
| rigid (0, 3°) rotation only | 0.296 | 0.295 | 0.300 (7 %) | 0.322 (39 %) | **0.328 (48 %)** | 0.315 (28 %) | 0.325 (44 %) |
| indep σ = 0.5 | 0.235 | 0.271 | 0.236 (1 %) | 0.244 (7 %) | 0.246 (9 %) | **0.334 (78 %)** | 0.319 (66 %) |
| indep σ = 1 | 0.144 | 0.224 | 0.146 (1 %) | 0.156 (6 %) | 0.158 (7 %) | **0.314 (78 %)** | 0.267 (57 %) |
| indep σ = 2 | 0.105 | 0.171 | 0.105 (0 %) | 0.111 (2 %) | 0.107 (1 %) | **0.293 (73 %)** | 0.179 (29 %) |

Selected paired differences (U-Net, per room, mean ± SE (z); `results.json`
→ `pairs`):

| comparison | rigid (0.5, 1°) | rigid (1, 2°) | rigid (2, 5°) | rigid (1, 0°) | indep σ = 1 |
|---|---|---|---|---|---|
| rigid + polish − joint LS | +0.011 ± 0.002 (4.4) | +0.015 ± 0.003 (4.5) | +0.008 ± 0.004 (2.0) | +0.003 ± 0.003 (1.2) | −0.047 ± 0.005 (−9.1) |
| rigid LS − joint LS | +0.000 ± 0.005 (0.0) | −0.013 ± 0.006 (−2.3) | −0.036 ± 0.006 (−5.8) | +0.030 ± 0.004 (8.4) | −0.158 ± 0.009 (−17.8) |
| autofocus + rigid − rigid LS | +0.003 ± 0.002 (1.3) | +0.001 ± 0.002 (0.4) | −0.001 ± 0.003 (−0.4) | −0.000 ± 0.001 (−0.2) | +0.002 ± 0.002 (0.9) |
| autofocus − no correction | +0.004 ± 0.003 (1.5) | +0.017 ± 0.004 (4.3) | +0.017 ± 0.004 (4.0) | +0.014 ± 0.004 (3.9) | +0.002 ± 0.002 (0.8) |
| jitter U-Net − U-Net, no correction | +0.010 ± 0.005 (1.9) | +0.044 ± 0.005 (8.1) | +0.041 ± 0.004 (9.3) | +0.012 ± 0.004 (2.7) | +0.080 ± 0.006 (12.6) |
| jitter U-Net − U-Net, after joint LS | −0.019 ± 0.004 (−4.6) | −0.018 ± 0.004 (−4.3) | −0.017 ± 0.004 (−4.1) | −0.018 ± 0.004 (−4.4) | −0.015 ± 0.004 (−3.5) |
| jitter U-Net − U-Net, after rigid + polish | −0.022 ± 0.004 (−5.4) | −0.025 ± 0.004 (−6.0) | −0.019 ± 0.004 (−4.4) | −0.022 ± 0.004 (−5.3) | −0.002 ± 0.004 (−0.6) |

**Threshold control.** Part of the jitter net's gain without correction
could come from its lower, perturbation-tuned threshold. The exact-pose
U-Net with that same kind of threshold (chosen on the perturbed validation
rooms, 0.209) scores 0.206 / 0.153 / 0.152 at rigid (1, 2°) / rigid (2, 5°)
/ indep σ = 1, against 0.203 / 0.143 / 0.144 at its own threshold and 0.247
/ 0.184 / 0.224 for the jitter net. So the threshold explains at most a
quarter of the gain.

**Full table** (every condition and correction; device error = mean
Euclidean error per device after correction; "Δ vs no correction" for the
jitter U-Net is against the exact-pose U-Net without correction, and the
U-Net's own gain over no correction is in brackets; `rigid_a_b` is
σ_t = a cells, σ_θ = b°, `indep_s` is σ = s, and `perturbed` means no
correction):

| condition | correction | device error (cells) | on exact cell | U-Net IoU | Δ vs exact (z) | Δ vs prior (z) | jitter U-Net IoU | Δ vs exact (z) | Δ vs no correction (z) |
|---|---|---|---|---|---|---|---|---|---|
| rigid_0.5_1 | perturbed | 0.65 | 42% | 0.280 ± 0.008 | -0.082 ± 0.006 (-13.5) | +0.179 ± 0.008 (22.2) | 0.290 | -0.072 ± 0.006 (-12.1) | +0.010 ± 0.005 (1.9) (U-Net: —) |
| rigid_0.5_1 | joint | 1.04 | 46% | 0.318 ± 0.008 | -0.044 ± 0.004 (-11.7) | +0.217 ± 0.009 (23.0) | 0.299 | -0.063 ± 0.005 (-13.8) | +0.019 ± 0.006 (2.9) (U-Net: +0.038 ± 0.007 (5.8)) |
| rigid_0.5_1 | rigid | 0.60 | 56% | 0.319 ± 0.008 | -0.044 ± 0.004 (-10.2) | +0.217 ± 0.009 (24.3) | 0.306 | -0.056 ± 0.005 (-11.5) | +0.025 ± 0.006 (4.3) (U-Net: +0.038 ± 0.005 (7.2)) |
| rigid_0.5_1 | autofocus | 0.65 | 42% | 0.284 ± 0.008 | -0.078 ± 0.006 (-13.3) | +0.183 ± 0.008 (22.3) | 0.290 | -0.072 ± 0.006 (-12.1) | +0.009 ± 0.005 (1.7) (U-Net: +0.004 ± 0.003 (1.5)) |
| rigid_0.5_1 | autofocus+rigid | 0.53 | 59% | 0.322 ± 0.008 | -0.040 ± 0.004 (-9.6) | +0.220 ± 0.009 (24.8) | 0.308 | -0.054 ± 0.005 (-11.0) | +0.028 ± 0.006 (4.8) (U-Net: +0.041 ± 0.005 (8.5)) |
| rigid_0.5_1 | rigid+polish | 0.79 | 52% | 0.329 ± 0.008 | -0.033 ± 0.003 (-9.7) | +0.228 ± 0.009 (24.4) | 0.307 | -0.055 ± 0.004 (-12.3) | +0.027 ± 0.006 (4.3) (U-Net: +0.049 ± 0.006 (7.7)) |
| rigid_1_2 | perturbed | 1.37 | 13% | 0.203 ± 0.006 | -0.159 ± 0.008 (-20.1) | +0.101 ± 0.006 (16.9) | 0.247 | -0.115 ± 0.007 (-16.4) | +0.044 ± 0.005 (8.1) (U-Net: —) |
| rigid_1_2 | joint | 1.56 | 43% | 0.308 ± 0.008 | -0.054 ± 0.004 (-12.5) | +0.207 ± 0.010 (21.7) | 0.290 | -0.072 ± 0.005 (-14.5) | +0.088 ± 0.007 (11.7) (U-Net: +0.106 ± 0.008 (12.5)) |
| rigid_1_2 | rigid | 1.20 | 41% | 0.295 ± 0.008 | -0.067 ± 0.005 (-12.5) | +0.194 ± 0.009 (22.4) | 0.288 | -0.074 ± 0.005 (-13.5) | +0.086 ± 0.007 (12.5) (U-Net: +0.093 ± 0.007 (13.4)) |
| rigid_1_2 | autofocus | 1.36 | 14% | 0.220 ± 0.007 | -0.142 ± 0.007 (-19.0) | +0.119 ± 0.007 (17.8) | 0.253 | -0.109 ± 0.007 (-15.8) | +0.051 ± 0.006 (8.8) (U-Net: +0.017 ± 0.004 (4.3)) |
| rigid_1_2 | autofocus+rigid | 1.11 | 43% | 0.296 ± 0.008 | -0.066 ± 0.005 (-12.1) | +0.195 ± 0.009 (22.4) | 0.291 | -0.071 ± 0.005 (-13.0) | +0.088 ± 0.007 (13.0) (U-Net: +0.094 ± 0.007 (13.4)) |
| rigid_1_2 | rigid+polish | 1.24 | 47% | 0.324 ± 0.008 | -0.039 ± 0.004 (-10.1) | +0.222 ± 0.009 (23.9) | 0.299 | -0.063 ± 0.005 (-13.2) | +0.096 ± 0.007 (13.0) (U-Net: +0.121 ± 0.008 (14.9)) |
| rigid_2_5 | perturbed | 2.80 | 4% | 0.143 ± 0.005 | -0.219 ± 0.009 (-24.7) | +0.041 ± 0.004 (9.8) | 0.184 | -0.178 ± 0.008 (-21.3) | +0.041 ± 0.004 (9.3) (U-Net: —) |
| rigid_2_5 | joint | 2.80 | 38% | 0.281 ± 0.008 | -0.081 ± 0.005 (-16.5) | +0.180 ± 0.009 (19.3) | 0.264 | -0.098 ± 0.005 (-18.5) | +0.122 ± 0.009 (14.2) (U-Net: +0.138 ± 0.009 (15.1)) |
| rigid_2_5 | rigid | 2.65 | 27% | 0.245 ± 0.007 | -0.117 ± 0.006 (-18.4) | +0.144 ± 0.008 (18.3) | 0.249 | -0.113 ± 0.006 (-18.1) | +0.106 ± 0.008 (14.0) (U-Net: +0.102 ± 0.008 (13.5)) |
| rigid_2_5 | autofocus | 2.89 | 5% | 0.160 ± 0.006 | -0.202 ± 0.009 (-23.7) | +0.059 ± 0.005 (11.0) | 0.195 | -0.167 ± 0.008 (-20.6) | +0.053 ± 0.006 (9.5) (U-Net: +0.017 ± 0.004 (4.0)) |
| rigid_2_5 | autofocus+rigid | 2.59 | 26% | 0.244 ± 0.007 | -0.118 ± 0.007 (-17.9) | +0.142 ± 0.008 (18.5) | 0.247 | -0.115 ± 0.007 (-17.5) | +0.105 ± 0.007 (14.1) (U-Net: +0.101 ± 0.007 (13.9)) |
| rigid_2_5 | rigid+polish | 2.57 | 38% | 0.290 ± 0.008 | -0.072 ± 0.005 (-15.7) | +0.188 ± 0.009 (20.2) | 0.271 | -0.091 ± 0.005 (-16.9) | +0.128 ± 0.008 (15.7) (U-Net: +0.147 ± 0.009 (16.2)) |
| rigid_1_0 | perturbed | 1.26 | 15% | 0.272 ± 0.007 | -0.090 ± 0.006 (-14.4) | +0.171 ± 0.008 (21.9) | 0.284 | -0.078 ± 0.007 (-11.9) | +0.012 ± 0.004 (2.7) (U-Net: —) |
| rigid_1_0 | joint | 1.11 | 47% | 0.320 ± 0.008 | -0.042 ± 0.004 (-11.5) | +0.218 ± 0.010 (22.9) | 0.302 | -0.060 ± 0.005 (-12.7) | +0.030 ± 0.006 (4.6) (U-Net: +0.048 ± 0.007 (7.1)) |
| rigid_1_0 | rigid | 0.79 | 64% | 0.350 ± 0.008 | -0.012 ± 0.002 (-5.1) | +0.249 ± 0.009 (26.7) | 0.320 | -0.042 ± 0.004 (-9.8) | +0.048 ± 0.006 (7.8) (U-Net: +0.078 ± 0.006 (12.3)) |
| rigid_1_0 | autofocus | 1.20 | 20% | 0.287 ± 0.008 | -0.076 ± 0.006 (-13.7) | +0.185 ± 0.008 (22.3) | 0.287 | -0.075 ± 0.006 (-12.0) | +0.015 ± 0.005 (2.9) (U-Net: +0.014 ± 0.004 (3.9)) |
| rigid_1_0 | autofocus+rigid | 0.75 | 65% | 0.350 ± 0.008 | -0.012 ± 0.002 (-5.2) | +0.248 ± 0.009 (26.6) | 0.318 | -0.044 ± 0.004 (-10.0) | +0.046 ± 0.006 (7.5) (U-Net: +0.077 ± 0.006 (12.3)) |
| rigid_1_0 | rigid+polish | 1.06 | 49% | 0.323 ± 0.008 | -0.039 ± 0.004 (-10.3) | +0.222 ± 0.009 (23.6) | 0.301 | -0.061 ± 0.005 (-13.1) | +0.029 ± 0.006 (4.6) (U-Net: +0.051 ± 0.007 (7.4)) |
| rigid_0_3 | perturbed | 0.51 | 62% | 0.296 ± 0.008 | -0.066 ± 0.005 (-12.2) | +0.194 ± 0.008 (23.2) | 0.295 | -0.067 ± 0.005 (-12.8) | -0.001 ± 0.005 (-0.2) (U-Net: —) |
| rigid_0_3 | joint | 1.08 | 46% | 0.315 ± 0.008 | -0.047 ± 0.004 (-11.3) | +0.213 ± 0.009 (22.6) | 0.299 | -0.063 ± 0.005 (-13.3) | +0.003 ± 0.006 (0.5) (U-Net: +0.019 ± 0.006 (3.3)) |
| rigid_0_3 | rigid | 0.80 | 54% | 0.322 ± 0.008 | -0.040 ± 0.004 (-10.0) | +0.221 ± 0.009 (24.3) | 0.304 | -0.058 ± 0.005 (-11.3) | +0.008 ± 0.006 (1.5) (U-Net: +0.026 ± 0.005 (5.2)) |
| rigid_0_3 | autofocus | 0.52 | 63% | 0.300 ± 0.008 | -0.062 ± 0.005 (-12.2) | +0.199 ± 0.008 (23.5) | 0.293 | -0.069 ± 0.005 (-12.8) | -0.003 ± 0.005 (-0.5) (U-Net: +0.005 ± 0.004 (1.1)) |
| rigid_0_3 | autofocus+rigid | 0.64 | 60% | 0.328 ± 0.008 | -0.034 ± 0.004 (-8.4) | +0.226 ± 0.009 (24.9) | 0.309 | -0.053 ± 0.005 (-11.2) | +0.013 ± 0.006 (2.3) (U-Net: +0.032 ± 0.005 (6.4)) |
| rigid_0_3 | rigid+polish | 0.97 | 49% | 0.325 ± 0.008 | -0.037 ± 0.004 (-10.3) | +0.223 ± 0.009 (23.5) | 0.303 | -0.059 ± 0.005 (-13.0) | +0.007 ± 0.006 (1.2) (U-Net: +0.029 ± 0.006 (4.9)) |
| indep_0.5 | perturbed | 0.59 | 46% | 0.235 ± 0.007 | -0.127 ± 0.007 (-18.1) | +0.134 ± 0.008 (16.9) | 0.271 | -0.091 ± 0.006 (-14.0) | +0.036 ± 0.006 (6.5) (U-Net: —) |
| indep_0.5 | joint | 0.55 | 58% | 0.334 ± 0.008 | -0.028 ± 0.003 (-8.3) | +0.233 ± 0.009 (25.1) | 0.313 | -0.049 ± 0.005 (-10.5) | +0.078 ± 0.007 (11.4) (U-Net: +0.099 ± 0.007 (14.0)) |
| indep_0.5 | rigid | 1.00 | 30% | 0.244 ± 0.007 | -0.118 ± 0.007 (-18.0) | +0.143 ± 0.008 (18.1) | 0.269 | -0.093 ± 0.006 (-14.9) | +0.034 ± 0.006 (6.0) (U-Net: +0.009 ± 0.005 (1.9)) |
| indep_0.5 | autofocus | 0.61 | 45% | 0.236 ± 0.008 | -0.126 ± 0.007 (-18.4) | +0.134 ± 0.008 (16.8) | 0.269 | -0.093 ± 0.006 (-14.6) | +0.034 ± 0.006 (6.2) (U-Net: +0.001 ± 0.003 (0.3)) |
| indep_0.5 | autofocus+rigid | 0.87 | 35% | 0.246 ± 0.008 | -0.116 ± 0.007 (-17.7) | +0.145 ± 0.008 (17.8) | 0.269 | -0.093 ± 0.006 (-15.1) | +0.034 ± 0.006 (6.1) (U-Net: +0.011 ± 0.005 (2.3)) |
| indep_0.5 | rigid+polish | 0.91 | 48% | 0.319 ± 0.008 | -0.043 ± 0.004 (-11.4) | +0.218 ± 0.009 (23.3) | 0.303 | -0.059 ± 0.005 (-13.0) | +0.068 ± 0.007 (9.9) (U-Net: +0.084 ± 0.007 (11.6)) |
| indep_1 | perturbed | 1.26 | 14% | 0.144 ± 0.005 | -0.218 ± 0.009 (-24.7) | +0.042 ± 0.005 (8.4) | 0.224 | -0.139 ± 0.007 (-19.0) | +0.080 ± 0.006 (12.6) (U-Net: —) |
| indep_1 | joint | 1.07 | 47% | 0.314 ± 0.008 | -0.048 ± 0.004 (-11.4) | +0.213 ± 0.009 (22.8) | 0.299 | -0.063 ± 0.005 (-13.1) | +0.155 ± 0.008 (19.0) (U-Net: +0.171 ± 0.009 (18.8)) |
| indep_1 | rigid | 1.96 | 11% | 0.156 ± 0.006 | -0.206 ± 0.009 (-23.5) | +0.054 ± 0.006 (9.7) | 0.218 | -0.144 ± 0.007 (-19.5) | +0.074 ± 0.007 (10.7) (U-Net: +0.012 ± 0.004 (2.8)) |
| indep_1 | autofocus | 1.28 | 14% | 0.146 ± 0.005 | -0.217 ± 0.009 (-24.2) | +0.044 ± 0.005 (8.6) | 0.222 | -0.140 ± 0.007 (-19.0) | +0.078 ± 0.006 (12.2) (U-Net: +0.002 ± 0.002 (0.8)) |
| indep_1 | autofocus+rigid | 1.81 | 12% | 0.158 ± 0.006 | -0.204 ± 0.009 (-23.4) | +0.057 ± 0.006 (10.1) | 0.221 | -0.141 ± 0.008 (-18.7) | +0.078 ± 0.007 (11.2) (U-Net: +0.014 ± 0.004 (3.4)) |
| indep_1 | rigid+polish | 1.74 | 29% | 0.267 ± 0.008 | -0.095 ± 0.005 (-17.4) | +0.166 ± 0.009 (19.5) | 0.265 | -0.097 ± 0.006 (-17.2) | +0.121 ± 0.008 (15.1) (U-Net: +0.124 ± 0.008 (15.4)) |
| indep_2 | perturbed | 2.50 | 4% | 0.105 ± 0.005 | -0.257 ± 0.009 (-27.5) | +0.004 ± 0.004 (1.0) | 0.171 | -0.191 ± 0.009 (-22.1) | +0.065 ± 0.006 (11.3) (U-Net: —) |
| indep_2 | joint | 2.18 | 38% | 0.293 ± 0.008 | -0.069 ± 0.005 (-14.8) | +0.192 ± 0.009 (20.4) | 0.277 | -0.085 ± 0.005 (-16.8) | +0.172 ± 0.009 (19.5) (U-Net: +0.188 ± 0.010 (19.4)) |
| indep_2 | rigid | 3.95 | 4% | 0.111 ± 0.004 | -0.252 ± 0.009 (-27.1) | +0.009 ± 0.003 (2.7) | 0.150 | -0.213 ± 0.009 (-24.3) | +0.044 ± 0.006 (7.0) (U-Net: +0.005 ± 0.004 (1.5)) |
| indep_2 | autofocus | 2.53 | 4% | 0.105 ± 0.005 | -0.257 ± 0.009 (-27.5) | +0.003 ± 0.004 (0.9) | 0.168 | -0.194 ± 0.009 (-22.3) | +0.062 ± 0.006 (11.1) (U-Net: -0.001 ± 0.001 (-0.5)) |
| indep_2 | autofocus+rigid | 3.70 | 4% | 0.107 ± 0.005 | -0.255 ± 0.009 (-27.6) | +0.005 ± 0.004 (1.5) | 0.157 | -0.205 ± 0.009 (-23.2) | +0.051 ± 0.006 (8.1) (U-Net: +0.001 ± 0.004 (0.4)) |
| indep_2 | rigid+polish | 3.82 | 10% | 0.179 ± 0.006 | -0.183 ± 0.008 (-23.0) | +0.077 ± 0.007 (11.7) | 0.197 | -0.165 ± 0.008 (-21.4) | +0.092 ± 0.007 (12.7) (U-Net: +0.073 ± 0.007 (10.8)) |
wrote tests/reports/pose_robust_2026_09_25_artifacts/results.json

## Findings

1. **Rigid errors: refine, then image with the exact-pose net.** At the
   three realistic rigid levels, rigid + polish recovers 60 / 76 / 67 % of
   the loss and is on par with or slightly above the 6.6.2 per-device joint
   fit (+0.008 to +0.015). The rigid fit alone is best when the error really
   is a translation: 86 % recovered, and 0.350 against 0.362. It is worse
   than the joint fit once rotations matter (−0.036 at (2, 5°)). The
   rotation grid (steps of σ_θ) and the rounding leave residual per-device
   errors that the one-cell polish removes.
2. **Poses are still not recovered, only made consistent with the data.**
   After rigid + polish at (1, 2°), the mean device error is 1.24 cells
   (1.37 before) and 47 % of devices sit on their exact cell (13 % before).
   The joint fit is similar (1.56 cells, 43 %). The IoU recovers much more
   than the poses do, because the fit finds cells whose empty-box response
   matches the data (the 6.6.2 finding holds for rigid errors too). Only
   pure translations are pinned: 64 % of devices exact, and IoU within
   0.012 of exact.
3. **Independent errors need the per-device fit.** A rigid search cannot
   represent them (it recovers ≤ 7 %). Rigid + polish recovers 57-66 % at
   σ ≤ 1 but only 29 % at σ = 2, where the one-cell polish is too narrow.
   The joint fit remains best there (73-78 %).
4. **Autofocus is a negative result.** Cross-pose coherence of the
   per-pose images lifts IoU by at most +0.017 (z ≈ 4) and never moves the
   poses (device error unchanged, for example 1.37 → 1.36 cells). Added to
   the rigid fit it changes nothing (|Δ| ≤ 0.003, except +0.006 at rotation
   only). The entropy metric is worse. The reason: with K = 4 bistatic
   poses, a pose error mainly corrupts the *background subtraction* (the
   mis-modelled direct wave leaks into the residual right at the onset that
   carving keys on). It does not coherently shift a sharp image that the
   other poses could align to. The four ellipse images are too smooth and
   too pose-specific to carry a focus signal. SAR autofocus works because
   hundreds of pulses see the same point scatterers, with the error a pure
   phase.
5. **Jitter training helps only when nothing is corrected.** It gives +0.04
   at realistic rigid errors and +0.04 to +0.08 at independent errors, at
   zero test-time cost (+0.01 at the smallest rigid level). It gives up
   0.034 at exact poses and 0.015-0.025 after the joint or rigid + polish
   refinement at every rigid level: the network learns to trust the fragile carving
   channel less, which is the wrong trade once the poses are fixed. A
   pipeline that always refines should keep the exact-pose network.
6. **Robust losses gave nothing.** Huber and the quiet-start cost tied with
   L2 on validation rooms (0.224 / 0.222 / 0.224). With the recording's
   scatter onset only ~4 lags after the direct arrival (6.6.2), there is no
   clean inlier window for a robust or RANSAC-style fit to exploit.

## Compute (per room of 4 poses, one numba thread, 4-core machine shared with another job)

| method | rigid (0.5, 1°) | rigid (1, 2°) | rigid (2, 5°) | engine runs per pose |
|---|---|---|---|---|
| imaging one variant (all four imagers) | 0.2 s | 0.2 s | 0.2 s | 2 (incident + time reversal) |
| candidate scoring (shared by rigid LS, autofocus, autofocus + rigid) | 0.3 s | 1.0 s | 3.5 s | one per distinct candidate source cell (≤ 22 / 96 / 349) |
| joint LS (6.6.2), window 2 / 3 / 5 | 0.5 s | 1.0 s | 2.4 s | (2R + 1)² = 25 / 49 / 121 |
| rigid + polish (rigid LS without focus images, 9-run polish, imaging) | 0.5 s | 1.0 s | 2.7 s | distinct sources + 9 |
| autofocus ascent itself | < 10 ms | < 10 ms | < 50 ms | 0 |
| jitter U-Net | 0 at test time | | | 0; 23 min of one core to image 14 000 perturbed training rooms, 16 min to train |

At laptop scale (a 3-6 m room is 120-240 cells a side) an engine run has
4-14× more cells per step than on the 64² grid, and more steps, so the refinement would cost seconds to
tens of seconds per placement on one core. The GPU engine brings that down.

## Limitations

1. **Simulation, noise-free, known outer box.** The refinement fits the
   empty-box model. A real room needs the room twin (walls from the echoes,
   `room_params.py` and the Lab page), and its own errors will add to the
   pose errors.
2. **v2 geometry.** The source and the mics of a pose are far apart, so
   rotation errors are about 3× larger in device displacement than on a
   laptop. The per-coordinate rounding also breaks exact rigidity.
3. **Matched jitter distribution.** The jitter net was trained on the same
   six error settings it was tested on (as a mixture, with independent
   draws). A real error distribution that differs would erode its gain.
4. **Coarse rotation grid.** Rotations are searched in steps of σ_θ. A
   finer grid (or continuous optimisation of θ) would probably narrow the
   gap to the polish variant at a proportional cost. It was not tried.
5. **Multiple comparisons.** About 90 paired tests are reported. The
   conclusions rest on |z| ≥ 4. The polish-vs-joint edge at (2, 5°) (z = 2.0)
   and the rotation-only autofocus + rigid gain (z = 2.6) do not survive a
   correction.

## Reproduce

```
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python scripts/eval_pose_robust.py --stages dev                 # ~10 min
uv run python scripts/eval_pose_robust.py --stages augdata             # 23 min on 2 processes (--chunks)
uv run python scripts/eval_pose_robust.py --stages augtrain            # 16 min
uv run python scripts/eval_pose_robust.py --stages held,polish \
    --conditions rigid_1_2,rigid_2_5,indep_2,rigid_0_3                  # ~2 h (split across processes)
uv run python scripts/eval_pose_robust.py --stages held,polish \
    --conditions indep_1,rigid_0.5_1,indep_0.5,rigid_1_0
uv run python scripts/eval_pose_robust.py --stages score \
    --out-dir tests/reports/pose_robust_2026_09_25_artifacts           # 5 min
uv run pytest tests/imaging/test_pose_robust.py                         # 7 tests, ~1 s
```

This needs the `eval_imaging_models.py` cache (per-pose images, priors, the
6.3.1 U-Net `unet_s0` and, for the independent conditions, the 6.6.2
`pose_s*.npz` files, whose perturbed and joint-refined images are reused
after checking that the random draws match). Total wall time was about 3.5 h
on two threads.
