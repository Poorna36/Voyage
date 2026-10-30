# Voyage --- Hybrid Quantum Research & Integration Build Spec

> **Goal:** Extend the existing Voyage lunar cross-sensor
> image-registration project with a modular, benchmark-first
> quantum/hybrid research layer. Integrate useful modules into the real
> registration workflow only after they pass validation. Preserve the
> existing classical pipeline as the trusted baseline and fallback.

## 1. Instructions to the AI IDE Agent

1.  **Inspect before editing.** Read the repository, current branch,
    docs, tests, API routes, frontend and configuration. Identify the
    actual working pipeline and existing quantum code. Do not assume
    documentation accurately describes implementation.
2.  Produce a short audit first: current architecture, runnable
    commands, existing modules, broken/incomplete features, dependencies
    and integration points. Then implement in small, testable stages.
3.  Do not rewrite or replace working registration code unnecessarily.
    Do not delete existing features, data, documentation or tests. Avoid
    broad refactors.
4.  Do not fabricate benchmark data, model results, hardware execution,
    speedups or completed integrations. Missing results must be marked
    `not_run`; unsupported experiments must be marked `skipped` with a
    reason.
5.  Keep the classical pipeline operational. With quantum features
    disabled, behaviour and outputs must match the current classical
    baseline within existing numerical tolerances.
6.  Use real lunar image-pair data for research claims. Synthetic data
    may be used for unit tests and controlled transformations, but must
    be labelled synthetic.
7.  Implement simulator support first. Add real quantum hardware
    backends only when credentials, access and a tested integration are
    available. Clearly distinguish hardware, simulator, quantum-inspired
    and purely classical methods.
8.  Pin compatible dependency versions and isolate optional quantum
    dependencies so the core application can still run without them.
9.  Add tests, documentation and a reproducible experiment for every
    implemented module. Do not claim a module is integrated merely
    because an API or dashboard page exists.
10. Before making destructive changes, changing public API contracts, or
    pushing commits, ask for approval.

## 2. Project Context

Voyage registers cross-sensor lunar imagery, particularly Chandrayaan-2
OHRC, TMC-2 and IIRS imagery against suitable lunar reference imagery
such as LRO products. The system should support feature extraction,
candidate matching, correspondence verification, geometric
transformation estimation, refinement and geospatial export.

Coordinate conventions must follow the existing project documentation.
Confirm them during the audit and do not silently change them.

## 3. Target Architecture

Keep the existing classical registration pipeline. Add a modular
research layer with three modes:

-   `classical`: existing pipeline only.
-   `hybrid`: classical pipeline plus explicitly enabled, validated
    quantum/hybrid modules.
-   `experimental`: individual research experiments; no automatic
    promotion into production.

Proposed flow:

1.  Validate inputs and metadata; preprocess images.
2.  Extract classical and optional experimental features.
3.  Select matcher using pair-level characteristics (optional QSVC).
4.  Generate candidate correspondences.
5.  Estimate match reliability (optional VQC/QNN or graph model).
6.  Select geometrically compatible correspondences (optional QUBO
    solver).
7.  Run robust geometric verification (retain the existing trusted
    method, e.g. MAGSAC++ if present and correctly implemented).
8.  Optimise transformation parameters (optional QPSO).
9.  Perform existing subpixel refinement and export.
10. Record metrics, configuration, provenance and limitations.

Do not force every experimental model into every run. Each module must
be independently switchable and have a safe fallback.

## 4. Research Modules and Minimum Specifications

### R1 --- QAOA and Quantum Annealing: Correspondence Selection

**Priority: first.**

-   Input: candidate matches with source/reference coordinates,
    descriptor score and confidence.
-   Build a compatibility graph using relative distance, orientation,
    scale and applicable geometric consistency.
-   Define binary variables `x_i` for selecting candidate matches.
-   Build a documented QUBO that rewards good matches and penalises
    incompatible pairs and constraint violations.
-   Solvers: greedy baseline, exact solver for small instances,
    simulated annealing, optional simulated quantum annealing, QAOA
    simulator. Add quantum annealing hardware only if available.
-   Return selected match IDs, objective value, solver metadata and
    feasibility/constraint diagnostics.
-   Compare solution quality and downstream registration against all
    baselines. Never call SQA a quantum hardware run.

### R2 --- Quantum Kernel + QSVC: Intelligent Matcher Selection

**Priority: first.**

-   Predict the best matcher or matcher suitability for an image
    pair/region.
-   Start with a compact, meaningful feature vector: GSD/scale
    difference, texture, contrast, entropy, feature density,
    illumination difference and overlap estimate where available.
-   Train labels from actual matcher outcomes on real image pairs. Do
    not use random synthetic labels as evidence.
-   Implement classical baselines (at least RBF-SVM and one tree-based
    model) and QSVC using a documented quantum feature map/kernel.
-   Use geographically separated train/validation/test splits; fit
    preprocessing only on training data.
-   Output recommendation, calibrated confidence if supported,
    alternatives and model/version metadata.

### R3 --- QPSO: Geometric Parameter Optimisation

**Priority: first.**

-   Optimise a clearly defined transform: begin with similarity or
    affine parameters; support homography only with suitable constraints
    and validation.
-   Initialise from the existing robust estimator rather than searching
    an unconstrained parameter space.
-   Define bounded parameters and a robust objective combining
    reprojection error, spatial coverage/consistency and optional
    photometric or regional-information terms.
-   Implement standard PSO as the mandatory baseline and QPSO as the
    quantum-inspired method.
-   Compare accuracy, convergence, stability and total runtime. Label
    QPSO as quantum-inspired classical optimisation, not quantum
    hardware.

### R4 --- VQC / QNN: Match Reliability

**Priority: second.**

-   Predict whether a candidate correspondence is a geometrically valid
    match.
-   Inputs may include descriptor distance/ratio, mutual-match flag,
    local scale/orientation differences, texture, matcher identity and
    spatial-consistency features.
-   Implement a small VQC/QNN (e.g. angle encoding and a compact
    parameterised circuit) and classical baselines such as logistic
    regression, MLP and a tree-based model.
-   Evaluate precision, recall, PR-AUC, false-positive rate and
    probability calibration, not accuracy alone.
-   Ensure training labels come from verified ground truth or carefully
    defined geometric validation.

### R5 --- QCNN + FRQI / NEQR: Patch Feature Research

**Priority: second; experimental.**

-   Begin with small patches around craters, rims, boulders, shadows,
    ridges and textured terrain. Do not encode full-resolution images
    initially.
-   Implement small FRQI/NEQR demonstrations and a hybrid quantum patch
    model only where feasible.
-   Compare with a classical CNN or classical patch descriptor on
    identical data.
-   Record image/state preparation cost, qubit count, circuit depth,
    measurements, inference time and feature quality.
-   Do not claim exponential image-processing savings from compact state
    representations without accounting for loading and readout.

### R6 --- Quantum Graph Neural Network: Correspondence Graph Learning

**Priority: second; experimental.**

-   Nodes represent candidate correspondences; node features include
    coordinates, descriptor quality and confidence.
-   Edges represent pairwise geometric compatibility and relative
    spatial relationships.
-   Start with small graphs and a hybrid model (classical graph
    construction/aggregation with a small quantum embedding or circuit
    component).
-   Compare against hand-designed graph filtering and a classical GNN
    (GCN, GraphSAGE or GAT).
-   Evaluate node-level inlier prediction and downstream registration
    quality. Do not scale circuit size before a small experiment
    demonstrates value.

### R7 --- QFT + Grover: Algorithm Experiments

**Priority: exploratory; separate sandbox.**

-   QFT: compare a tiny encoded feature/patch experiment with classical
    FFT. Include encoding and measurement/readout costs.
-   Grover: define a small, explicit correspondence-search
    predicate/oracle and compare against exhaustive and indexed
    classical search.
-   Report oracle construction, circuit depth, qubits, shots, total
    execution and post-processing. Iteration-count reduction alone is
    not end-to-end speedup.
-   These experiments must not be required by the production
    registration pipeline.

### R8 --- QPCA + Quantum Clustering: IIRS Spectral Research

**Priority: exploratory; separate track.**

-   Use calibrated/valid IIRS spectral vectors after invalid/noisy band
    handling and documented normalisation.
-   Compare QPCA with classical PCA/incremental PCA; compare quantum
    clustering with k-means and classical kernel clustering.
-   Evaluate explained variance, reconstruction error, cluster
    quality/stability and whether resulting features improve
    registration.
-   Do not assign mineralogical or geological meaning to clusters
    without independent scientific validation.

## 5. Dataset and Evaluation Protocol

Create a dataset manifest; do not hardcode machine-specific paths. Each
pair should record:

-   Pair ID, source/reference identifiers and sensor types.
-   Acquisition and relevant sensor metadata, GSD and scale information
    where available.
-   Image paths/checksums, overlap/valid-region information and
    preprocessing version.
-   Ground-control points or independently verified correspondences,
    annotation source and uncertainty.
-   Geographic region identifier and train/validation/test split.

Include OHRC--reference, TMC-2--reference and IIRS--suitable-reference
cases where data is available. Keep same-sensor or synthetic transformed
pairs as separate controls, not substitutes for cross-sensor testing.

**Prevent geographic leakage:** split by lunar region/scene, not random
adjacent patches. Keep a held-out test set untouched until model
selection is complete.

Mandatory registration metrics where applicable:

-   RMSE and median absolute reprojection error.
-   Percentage of points below 0.5 px and 1 px.
-   Verified inlier count and inlier ratio.
-   Spatial coverage/grid occupancy.
-   End-to-end runtime and memory.
-   For learning: macro-F1, precision/recall, PR-AUC and calibration.
-   For quantum circuits: qubit count, circuit depth, shots/evaluations,
    backend and state/data preparation cost.

Every experiment must compare with a relevant strong classical baseline
and report both isolated module performance and complete-pipeline
impact. Use multiple seeds where stochastic algorithms are involved.
Record negative results.

## 6. Experiment Tracking and Result Integrity

Create a consistent experiment runner and machine-readable result
format. Every run must record:

-   Experiment ID, timestamp, git commit, configuration and random seed.
-   Dataset manifest/version and split.
-   Algorithm, parameters and baseline.
-   Backend type: classical, quantum simulator, quantum-inspired, or
    quantum hardware.
-   Runtime, resource use and relevant circuit details.
-   Metrics, warnings, status (`completed`, `failed`, `not_run`,
    `skipped`) and skip reason.
-   Output artifact paths and reproducibility instructions.

Never return invented default benchmark values. If no benchmark exists,
return an explicit empty/not-run state. Do not count skipped samples as
successful or perfect results. Do not describe classical exhaustive work
as quantum speedup.

## 7. Integration, API and UI

First inspect existing API/frontend conventions and extend them rather
than creating a parallel application.

-   Expose module execution and experiment status through the existing
    backend conventions.
-   Validate inputs and configuration; use background jobs for long
    experiments if the project already supports them.
-   Return structured results with status, metrics, backend and
    warnings.
-   Keep experimental APIs clearly labelled and separate from production
    pipeline endpoints.
-   Add dashboard controls for mode, module selection, run status and
    baseline comparison only if consistent with the existing UI.
-   Display simulator/hardware status and clearly label quantum-inspired
    algorithms.
-   Do not display placeholder metrics as real results.

## 8. Configuration

Use the project's existing configuration pattern if one exists.
Otherwise provide a small configuration structure equivalent to:

``` yaml
mode: classical  # classical | hybrid | experimental

quantum:
  enabled: false
  backend: simulator
  modules:
    qaoa_correspondence: false
    quantum_annealing: false
    qsvc_matcher_selection: false
    qpso_geometry: false
    vqc_reliability: false
    qcnn_patch_features: false
    quantum_graph: false
    qft_grover: false
    qpca_clustering: false

experiments:
  seed: 42
  save_artifacts: true
  require_real_data_for_claims: true
```

Do not silently enable expensive or hardware-dependent modules. Validate
unknown options and unsupported backends.

## 9. Tests and Acceptance Criteria

For every module add:

1.  Unit tests for input validation, mathematical formulation and output
    schema.
2.  Small deterministic tests with known expected behaviour.
3.  Integration tests proving it connects to the intended pipeline
    stage.
4.  Failure/fallback tests when dependencies or backends are
    unavailable.
5.  Reproducibility tests where feasible.

Overall acceptance:

-   Existing classical tests and workflows continue to work.
-   Quantum disabled produces the established classical behaviour.
-   Each implemented module has a real runnable experiment and relevant
    baseline.
-   No fabricated metrics or misleading speedup claims.
-   Real-data evaluation uses geographically separated splits.
-   API/UI status reflects actual execution state.
-   Documentation explains setup, execution, outputs, limitations and
    how to reproduce results.
-   Only modules that pass documented gates can be enabled in hybrid
    mode.

## 10. Implementation Order

Work in this order and report progress after each stage:

1.  Repository audit and baseline verification.
2.  Dataset manifest, experiment runner and result schema.
3.  QUBO correspondence selection: greedy, exact-small, annealing/SQA,
    then QAOA simulator.
4.  QSVC matcher selection with classical baselines.
5.  QPSO geometry with standard PSO baseline.
6.  VQC/QNN reliability.
7.  QCNN/FRQI/NEQR and quantum graph prototypes.
8.  QFT/Grover and IIRS QPCA/clustering experiments.
9.  Selectively integrate validated modules; run full regression and
    benchmark suite.
10. Update docs and provide a final implementation report.

## 11. Required Final Report from the Agent

At completion, provide:

-   What was already present versus newly implemented.
-   Files added/changed and why.
-   Commands to install, test and run each experiment.
-   Tests actually executed and their real results.
-   Experiments completed, failed, skipped or not run.
-   Dataset limitations and whether real lunar data was used.
-   Backend used for each quantum experiment.
-   Baseline comparisons and measured outcomes.
-   Known limitations and recommended next steps.

**Do not claim all eight directions are complete unless each has a
working implementation, tests, reproducible experiment and documented
evaluation.**
