# Architecture Decision Records
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document records the foundational architectural decisions, theoretical justifications, empirical evidence, and rejected alternatives for the system.

---

## D01. Pluggable, Benchmark-First Matcher Architecture

Decision: No single matcher is declared the universal solution. Candidate matchers run within a unified benchmark harness, and winner arbitration is determined per terrain and illumination stratum.

Evidence: Classical descriptors excel on high-contrast equatorial topography; learned matchers remain robust under large illumination shifts; crater-geometry graphs operate reliably in polar crater networks. No single methodology dominates all lunar regimes.

Alternatives Rejected:
- Hardcoded single-matcher pipeline: Fragile under extreme illumination and scale variations.
- Late ad-hoc selection: Obscures stratum-level failure modes and prevents objective arbitration.

---

## D02. SIFT (M0) as Fixed Baseline and Universal Fallback

Decision: SIFT executes across all pairs to establish a performance floor and guarantee a fallback output.

Evidence: Empirical studies on Chandrayaan-2 IIRS-to-WAC pairs confirm sub-pixel RMSE across 200 strips in textured regions. Furthermore, SIFT consistently outperforms ORB and unweighted feature concatenations on lunar panchromatic data.

Alternatives Rejected:
- ORB as baseline: Exhibits the lowest inlier survival rate on low-contrast regolith.
- IntFeat (SIFT+ORB concatenation): Fails to improve registration accuracy while increasing descriptor noise.

---

## D03. SuperPoint with LightGlue (M2) as Primary Learned Matcher

Decision: SuperPoint feature extraction paired with LightGlue attention-based matching serves as learned matcher M2.

Evidence: LightGlue provides adaptive depth/width computation (~17 ms per pair), positional encoding robustness, open Apache-2.0 licensing, and integrated support in planetary matching pipelines.

Alternatives Rejected:
- SuperGlue: Superseded by LightGlue; heavier compute overhead and rigid inference layers.
- LoFTR: Detector-free architecture trained on terrestrial terrestrial scenes without planetary illumination modeling; lacks explicit point covariance.

---

## D04. DEGENSAC and MAGSAC++ for Robust Geometric Verification

Decision: DEGENSAC or MAGSAC++ replaces standard RANSAC for geometric model estimation.

Evidence: Standard RANSAC degrades silently on planar or near-planar lunar mare basins. DEGENSAC explicitly guards against dominant-plane degeneracy by testing two-point homography samples against planar hypothesis tests.

Alternatives Rejected:
- Standard RANSAC: Prone to degenerate planar overfitting.
- Orientation-consistency filtering (DOC/MGEO): Discards valid correspondences due to non-linear shadows.

---

## D05. Hierarchical Geometric Model Ladder

Decision: Models are evaluated hierarchically in ascending degrees of freedom: Similarity -> Affine -> Homography -> Local Tile Models.

Evidence: Planetary terrain relief causes global homographies to overfit on low-distortion scenes, while simple affine warps fail completely in mountainous topography. The ladder selects the lowest degree of freedom model satisfying inlier residual tolerances.

Alternatives Rejected:
- Fixed global homography: High parameter count overfits on low-overlap or sparse-feature pairs.
- Fixed affine transformation: Fails under high-relief topographic relief.

---

## D06. Dual-Stage Spatial Uniformity Enforcement

Decision: Spatial distribution is controlled at two pipeline stages: keypoint-level ANMS (SSC) prior to feature description (for M0/M1), and grid-based density capping post-matching (for all matchers).

Evidence: Concentrated feature clusters along high-contrast crater rims starve flat terrain of correspondences, distorting global transformation estimates. Dual-stage filtering ensures wide geometric baseline support across the entire field of view.

Alternatives Rejected:
- Pre-match ANMS only: Ineffective for learned matchers with internal keypoint sampling.
- Post-match grid capping only: Fails to prevent detector starvation in lower-contrast image tiles.

---

## D07. Phase-Correlation Sub-Pixel Refinement with Tukey Apodization

Decision: Layer 5 executes local patch phase correlation or normalized cross-correlation with Gaussian pyramids, 2D paraboloid peak interpolation, and Tukey window apodization.

Evidence: Apodization benchmarks demonstrate that Tukey and Gaussian windows minimize spectral leakage and deliver peak localization precision below 0.1 pixels. Blackman windows consistently degrade peak sharpness in lunar imagery.

Alternatives Rejected:
- Omission of refinement: Raw feature detection yields integer or coarse sub-pixel coordinates exceeding target tolerances.
- Blackman apodization: Demonstrably broadens correlation peaks and degrades sub-pixel accuracy.

---

## D08. Multi-Octave Scale-Space Extension for RIFT2 (M1)

Decision: Matcher M1 incorporates RIFT2 with a multi-octave log-Gabor scale-space extension.

Evidence: Standard RIFT phase congruency is rotation-invariant but scale-sensitive. In cross-sensor pairs (such as OHRC vs TMC-2 with GSD ratios up to 17x), scale-space search is required to guarantee convergence.

Alternatives Rejected:
- Unmodified RIFT: Fails when resolution disparity exceeds 1.5x.
- HOPC (Histogram of Oriented Phase Congruency): Relies heavily on accurate a-priori spacecraft geolocation metadata.

---

## D09. Dedicated Parallel Module for IIRS Hyperspectral Registration

Decision: IIRS registration operates as a dedicated module with mandatory radiometric correction and dedicated spectral band selection.

Evidence: Hyperspectral bands (0.8 to 5.0 um at 80 m GSD) exhibit non-linear emissivity and photometric phase variations that violate panchromatic intensity assumptions. Photometric phase-angle correction is mandatory before correspondence search.

Alternatives Rejected:
- Merging IIRS into panchromatic pipeline: Causes gradient collapse and false feature correspondences.

---

## D10. Deterministic Normalization over Generative Radiometric Synthesis

Decision: Radiometric normalization relies on percentile clipping, CLAHE, and statistical moment transfer rather than Generative Adversarial Networks (GANs).

Evidence: Deep generative synthesis introduces non-verifiable geometric hallucinations and artifacts near high-contrast crater rims. Classical moment transfer normalizes dynamic ranges predictably without spatial hallucination.

Alternatives Rejected:
- CycleGAN/cGAN normalization: Unstable training dynamics and risk of hallucinated topographic landmarks.

---

## D11. Tile-Wise Local Model Partitioning for High Latitudes

Decision: For pairs at latitudes beyond +/- 55 degrees or with extreme topographic relief, the pipeline falls back to tile-wise local transformations.

Evidence: Lunar surface curvature and oblique pushbroom angles cause single global projective homographies to diverge at extreme latitudes. Tile-wise local planar approximations preserve sub-pixel accuracy across large swaths.

Alternatives Rejected:
- Single global homography at polar latitudes: Diverges exponentially near lunar poles.

---

## D12. Geometric Checkpoint Metrics over Structural Image Similarity

Decision: Primary evaluation relies on ground-truth RMSE, inlier counts, inlier ratios, and spatial coverage. Image-wide SSIM and PSNR serve only as qualitative diagnostic checks.

Evidence: High SSIM scores can occur on misaligned images in flat maria, while legitimate photometric differences penalize SSIM despite perfect geometric alignment. Independent ground-truth point residuals provide the only scientifically rigorous evaluation.

Alternatives Rejected:
- SSIM/PSNR as primary acceptance gates: Poorly correlated with true sub-pixel coordinate alignment.

---

## D13. Crater Density Gating for Crater-Geometry Matcher (M3)

Decision: Matcher M3 executes only when detected crater density exceeds tau_c in both images and terrain is classified as highland or polar.

Evidence: Graph-matching algorithms (CNSFM) require sufficient node density to construct invariant topological descriptors. In sparse maria, graph construction fails and wastes computational cycles.

Alternatives Rejected:
- Unconditional M3 execution: Generates degenerate graph solutions and false positive correspondences in maria.

---

## D14. LNIFT Evaluation for Classical Fallback Promotion

Decision: LNIFT is included in the benchmark registry alongside RIFT2 to evaluate runtime and success rate advantages on cross-sensor pairs.

Evidence: Literature indicates LNIFT achieves orders-of-magnitude faster descriptor extraction than RIFT with high matching success. Benchmark evaluation on real Chandrayaan-2 pairs governs its promotion to primary classical fallback.

---

## D15. Placement of Matcher Selection Model at Layer 1.5

Decision: The Matcher Selection Model (MSM) is situated at Layer 1.5 (Stage 4.5), immediately after preprocessing and prior to matcher dispatch.

Evidence: Exhaustive execution of all matchers incurs 4x to 10x compute overhead. Placing the model after L1 allows leveraging patch-level texture contrast, gradient magnitudes, and shadow mask statistics to predict the optimal matcher before initiating heavy compute.

Alternatives Rejected:
- Post-match arbitration only: Incurs full computational cost across all matchers.
- Pre-L1 selection: Lacks localized texture and shadow metrics necessary for accurate routing.

---

## D16. LightGBM Gradient-Boosted Decision Trees for Meta-Classification

Decision: LightGBM multi-class classification is selected as the meta-selection algorithm for MSM.

Evidence: Tabular planetary metadata and localized image features are modeled efficiently with GBDTs on medium sample sets (N = 50-500). Inference executes in under 5 ms without GPU dependencies.

Alternatives Rejected:
- Deep neural networks: High sample complexity and risk of overfitting on specialized planetary calibration sets.
- Fixed heuristic rule trees: Incapable of learning complex empirical trade-offs between solar angle, GSD, and texture.

---

## D17. Dual-Threshold Confidence Routing and Safe-Mode Fallback

Decision: The MSM employs a dual-threshold routing policy: high confidence (>= 0.65) executes the predicted winner; medium confidence (0.40 to 0.65) executes the top two candidates; low confidence (< 0.40) triggers safe mode (runs all matchers).

Evidence: Planetary mission pipelines cannot tolerate silent failures. Dual-threshold routing achieves greater than 50% execution time reduction while bounding RMSE degradation within 0.10 pixels.

Alternatives Rejected:
- Strict argmax single-matcher routing: Risks catastrophic failure when model confidence is distributed evenly across multiple matchers.

---

## D18. Strict Geographic Cell Disjointness for Meta-Model Training

Decision: MSM cross-validation and training splits must strictly group samples by 10-degree geographic cells, ensuring zero spatial overlap with test sets.

Evidence: Regional geological formations share distinctive regolith texture and crater distributions. Random pair-wise splitting allows the classifier to memorize geographic cells, causing optimistic evaluation bias.

Alternatives Rejected:
- Random pair train/test split: Violates spatial cross-validation hygiene in geospatial machine learning.

---

## D19. Formal Acceptance Criteria for Production Model Activation

Decision: The MSM remains inactive in production until satisfying all 8 formal Acceptance Criteria (AC1 through AC8) on held-out test splits.

Evidence: Production registration must ensure non-regression against the baseline before automated routing is permitted to bypass candidate algorithms.

---

## Architecture Evolution and Revision History

### MSM Integration (Layer 1.5 / Stage 4.5)
The Matcher Selection Model was introduced to resolve the computational bottleneck of exhaustive multi-matcher execution:
- Added `src/selector/` containing feature extraction (`features.py`) and LightGBM inference (`model.py`).
- Added 13-dimensional feature vector combining geometric, illumination, and texture metrics.
- Added dual-threshold routing logic with automated safe-mode fallback.
- Added hard-rule gates: crater density gating for M3, GPU availability checks for M2, and dedicated routing for IIRS.
- Added 8 formal acceptance criteria (AC1 through AC8) in `VALIDATION.md`.
- Added regression tests T13 through T16 in the validation suite.
- Enhanced preprocessing to record patch texture contrast, mean gradient magnitudes, and active tile counts into `meta.json`.
- Established strict geographic cell disjointness (`geo_cell`) across all training and validation splits.
