# Voyage — Quantum Integration Build Specification

**Project:** Voyage — Cross-sensor lunar image registration  
**Goal:** Add a focused, testable hybrid quantum layer to the existing Voyage pipeline without replacing its proven classical image-processing and geometric-registration stages.

**Scope to build:** 
1. Intelligent matcher selector
2. QNN for candidate-match reliability
3. QCNN (the “QCN” direction) for small-patch feature extraction
4. IBM-compatible local quantum simulator/emulator
5. QAOA-based correspondence optimisation

Do not implement other quantum algorithms or turn this into a separate demo application.

---

## 1. Non-negotiable instructions for the AI IDE agent

1. **Audit before coding.** Inspect the actual repository, branch, docs, tests, API, frontend and end-to-end registration path. Verify what exists; do not infer implementation from documentation or old claims.
2. First provide a short audit containing: runnable baseline commands, pipeline stages, available matchers, current quantum code, integration points, dependencies and blockers.
3. Preserve the existing classical pipeline and its outputs. Do not rewrite unrelated modules or delete existing features.
4. Implement in phases. After each phase, run relevant tests and report the real result before moving on.
5. Do not fabricate datasets, labels, metrics, benchmark results, simulator runs or quantum speedups. Use explicit states such as `not_run`, `failed` and `skipped` with a reason.
6. Quantum functionality must be optional. `quantum.enabled: false` must use the existing classical path and must not require quantum packages to be installed.
7. Start with local simulation. No IBM account, API token, cloud job or real QPU is required. If fake IBM backends are used, clearly label them as simulated device snapshots—not hardware execution.
8. Check compatibility between the installed Qiskit SDK, Qiskit Aer and Qiskit Machine Learning before pinning versions. Pin a tested compatible set; isolate optional dependencies.
9. Do not create a parallel backend/frontend if Voyage already has one. Follow the project's established design and API conventions.
10. Ask before destructive changes, changing public API contracts, removing existing functionality or pushing commits.

## 2. Intended architecture

Keep classical image processing and geometric registration as the foundation. Quantum components are optional modules called at selected pipeline stages.

```text
Source lunar image + reference image + metadata
                       |
              Existing preprocessing
                       |
             Intelligent Matcher Selector
                       |
          Existing classical feature matcher(s)
                       |
              Candidate correspondences
                       |
          +------------+-------------+
          |                          |
     QCNN patch features        Classical patch/
     (optional feature)         descriptor features
          |                          |
          +------------+-------------+
                       |
              Candidate match features
                       |
                QNN reliability
                       |
          QAOA correspondence selection
                       |
       Existing robust geometric verification
                       |
        Existing transform + subpixel refinement
                       |
           Registered output + evaluation
```

**Responsibilities must remain distinct:**

- **Matcher selector:** chooses which available matcher to run for an image pair/region.
- **QCNN:** extracts a compact feature representation from small local image patches. It does not replace the full image feature detector.
- **QNN:** estimates whether a proposed correspondence is likely to be valid.
- **QAOA:** selects a mutually compatible subset of candidate correspondences.
- **Classical geometry:** estimates and refines the final image transformation.

The QNN and QCNN are separate models with different inputs and outputs. Do not combine them into one vague “quantum AI” module.

## 3. Component specifications

### A. Intelligent matcher selector

**Purpose:** Select the most suitable matcher that actually exists in Voyage for a given image pair or region.

**Inputs (use available fields only):**
- Source/reference sensor pair.
- GSD or scale difference, if metadata exists.
- Texture, contrast and entropy.
- Feature density and descriptor statistics.
- Illumination difference and overlap estimate, where available.
- Optional quick quality probes from candidate matchers, if inexpensive.

**Behaviour:**
1. Inspect and enumerate the real matchers implemented in the repository. Do not invent matcher names or assume all example matchers are available.
2. Initially implement a transparent rule-based selector using measurable pair characteristics and benchmark evidence.
3. In parallel, collect results from each available matcher on the same training pairs.
4. Once sufficient real labelled data exists, support a learned pair-level selector. Train labels from observed downstream registration outcomes, not hand-invented labels.
5. Allow a user override and keep a configurable existing matcher as fallback.
6. Return selected matcher, selection reason/features, confidence only when calibrated, fallback used and model/config version.

**Evaluation:** Compare fixed-best-matcher, rule-based selector and learned selector using downstream registration error, success rate, spatial coverage and total runtime. Split by lunar scene/region to prevent leakage.

### B. QNN — candidate match reliability

**Purpose:** Estimate the probability that an individual candidate correspondence is geometrically valid.

**Input:** A compact numeric feature vector for each match, using available measurements such as:
- Descriptor distance and ratio.
- Mutual-match indicator.
- Local scale and orientation differences.
- Local texture/contrast.
- Matcher identity.
- Spatial-consistency features relative to neighbouring matches or an initial transform.

**Model:**
- Implement one small hybrid QNN using Qiskit Machine Learning.
- Choose `EstimatorQNN` or `SamplerQNN` based on the pinned, tested package versions and the desired output. Do not mix incompatible examples from different Qiskit releases.
- Use classical scaling/feature selection before encoding.
- Begin with a shallow circuit and small feature/qubit count; make the count configurable and record it.
- Train with labels from verified correspondences or carefully validated geometric ground truth.
- Output a reliability score/probability and model metadata. Only call it a probability when calibration has been evaluated.

**Baselines:** Logistic regression and a small tree-based model at minimum.

**Metrics:** Precision, recall, PR-AUC, false-positive rate, Brier score/calibration error and downstream registration impact. Do not use accuracy alone.

### C. QCNN — small-patch quantum feature extraction

**Purpose:** Experiment with learned quantum features for small lunar terrain patches that may help distinguish local structures across sensors.

**Potential patches:** Crater rims, crater interiors, boulders, shadow boundaries, ridges, rough terrain and distinctive texture regions.

**Constraints:**
- Do not encode entire lunar images or large image tiles into quantum circuits.
- Begin with tiny, normalised patches (for example 2×2 or 4×4), or compress classical patch features before encoding.
- Keep the initial circuit small (target roughly 4–8 qubits where practical); document any change to this budget and why it is needed.
- Implement a genuine quantum convolution/pooling-style circuit, based on a documented QCNN design. A generic fully connected VQC must not be labelled a QCNN.
- Read out a compact feature vector from circuit measurements/observables. Make it usable as an optional feature alongside Voyage's existing classical descriptors.
- Compare QCNN features with a classical patch descriptor or small CNN using exactly the same train/validation/test scene splits.
- Test cross-sensor usefulness: train/evaluate without leaking patches from the same lunar scene across splits.
- Record patch preparation/encoding time, qubits, circuit depth, shots, execution time and feature dimension.

**Ablations:** classical features only; QCNN features only where meaningful; classical + QCNN features.

**Do not claim** that compact quantum encoding makes full-image processing exponentially cheaper. State preparation, measurements and readout are part of the cost.

### D. IBM-compatible local quantum simulator/emulator

**Purpose:** Provide one local backend for QNN, QCNN and QAOA experiments without cloud credentials.

**Default implementation:**
- Use Qiskit Aer `AerSimulator` for local circuit simulation, if compatible with the pinned Qiskit stack.
- Optionally support an IBM fake backend/noise snapshot for local testing of device-like noise and transpilation behaviour.
- Expose a small backend interface shared by all quantum components.
- Support ideal simulation first; noisy simulation is optional and must specify its noise model.
- Make simulator mode explicit in every result: `ideal_simulator`, `noisy_simulator` or `fake_ibm_backend`.
- Record backend name/type, simulator method, noise model, shots, qubits, circuit depth and transpilation settings.

**Important wording:** This is IBM-compatible local quantum simulation/emulation. It is not a real IBM QPU and does not require an IBM Quantum account. Real hardware execution is outside this build scope.

### E. QAOA — overall quantum correspondence optimisation

**Purpose:** Select a high-quality, geometrically compatible subset of candidate correspondences before the existing robust geometric estimator.

**Formulation:**
- Each candidate match \(i\) has binary selection variable \(x_i \in \{0,1\}\).
- Give a positive reward \(w_i\) for a high-quality candidate. This can use QNN reliability when available, or classical match confidence otherwise.
- Estimate pairwise incompatibility \(c_{ij}\) from relative distance, orientation, scale and geometric consistency.
- Define and document a QUBO such as:

\[
\min_x E(x) = -\sum_i w_i x_i
               + \lambda\sum_{i<j} c_{ij}x_ix_j
               + P(x)
\]

where \(P(x)\) represents any necessary conflict/assignment penalties, such as preventing duplicate use of the same source or reference feature. Define the sign, scale and meaning of every term; test the formulation against hand-computable examples.

**Implementation:**
1. Build a compatibility/conflict graph from candidate matches.
2. Filter and cap candidates before circuit construction. Start with small instances (for example up to 8–16 variables, subject to simulation cost); make the limit configurable and document it.
3. Implement one QAOA solver on the local simulator.
4. Decode sampled bitstrings, recompute their classical QUBO objective and reject invalid solutions.
5. Return selected match IDs, objective, feasibility/constraint diagnostics, backend and runtime.
6. Pass selected correspondences into Voyage's existing RANSAC/MAGSAC++ or other verified geometric estimator. Do not replace it.
7. If QAOA is disabled, unavailable or fails, use the existing classical correspondence filtering path and explicitly report the fallback.

**Baselines:** Greedy compatibility selection and an exact classical solver for small instances. These are required to validate the QUBO and QAOA output.

**Metrics:** QUBO objective, constraint violations, gap to exact optimum on small instances, selected count, downstream inlier ratio, spatial coverage, registration error and total runtime including graph construction, circuit simulation, sampling and decoding.

Do not implement separate quantum annealing, QPSO, Grover or another optimisation algorithm in this scope. Do not claim quantum advantage from fewer iterations or circuit measurements alone.

## 4. End-to-end execution modes

Provide these modes through the existing project configuration/API if practical:

- `classical`: existing pipeline only; no quantum dependency required.
- `hybrid`: selected quantum components enabled, with safe classical fallbacks.
- `experimental`: allows an individual QCNN/QNN/QAOA experiment to run and save results without promoting it into normal registration.

Each component needs its own enable flag. Do not enable expensive circuits by default. The user must be able to override the matcher selection and disable QNN, QCNN or QAOA independently.

## 5. Configuration target

Follow existing configuration conventions. If none exist, implement an equivalent validated configuration:

```yaml
mode: classical

quantum:
  enabled: false
  backend: aer_simulator
  shots: 1024
  noise_model: null

  matcher_selector:
    enabled: false
    strategy: rules       # rules | trained
    allow_manual_override: true
    fallback_matcher: null  # resolve to an actually available matcher

  qnn:
    enabled: false
    purpose: match_reliability
    max_qubits: 8

  qcnn:
    enabled: false
    purpose: patch_features
    max_qubits: 8
    patch_size: 4

  optimisation:
    enabled: false
    algorithm: qaoa
    max_candidate_matches: 16
    reps: 1
```

Validate unknown options, unavailable matchers, incompatible dependencies and unsupported backends. Never silently pretend that a fallback quantum computation ran.

## 6. Data and experiment protocol

Create or extend a dataset manifest. Do not hardcode personal machine paths.

For each pair/patch record, where available:
- Pair and patch IDs; source/reference image IDs.
- Sensor pair (OHRC, TMC-2, IIRS, reference sensor).
- Geographic region/scene ID.
- Acquisition metadata and GSD.
- Image path/checksum and preprocessing version.
- Ground-truth source, correspondence/label and annotation confidence.
- Train/validation/test split.

Use real lunar pairs for scientific claims. Synthetic transformations may be used for unit tests and controlled sensitivity tests, clearly labelled synthetic.

**Split by geographic scene/region**, not random crops from the same scene. Hold out regions for the final test set.

Required comparisons:
1. Classical baseline.
2. Baseline + matcher selector.
3. Baseline + QNN reliability.
4. Baseline + QCNN patch features.
5. Baseline + QAOA correspondence selection.
6. Combined hybrid pipeline.

Track registration RMSE and median error, percentage below 0.5 px and 1 px, verified inlier count/ratio, spatial coverage, total runtime and memory. Track model-specific and circuit-specific metrics in addition.

## 7. Experiment records and result integrity

Every run must save a machine-readable record with:
- Experiment ID, timestamp, git commit and configuration.
- Dataset manifest/version and geographic split.
- Module/model name and version.
- Backend type, simulator/noise configuration, shots and circuit resource metadata.
- Random seed, runtime and resource usage.
- Metrics, warnings, status (`completed`, `failed`, `not_run`, `skipped`) and reason.
- Artifact paths and reproducibility command.

No fabricated default benchmark metrics. If no run exists, return `not_run` or an empty result. Skipped samples must not count as successful solutions.

## 8. API and dashboard integration

Inspect the current backend and UI and extend their existing patterns. Do not build a separate app.

Expose, as supported by the existing architecture:
- Execution mode and component enable flags.
- Actual selected matcher and selection reason.
- QNN per-match reliability outputs when the QNN ran.
- QCNN feature status/dimension and patch-level outputs when it ran.
- QAOA run status, backend, candidate/selected counts, objective, constraint status and runtime.
- Fallback reason and warnings.

Clearly distinguish real computed outputs, cached outputs, simulation results and unavailable/not-run states. Long-running jobs should use the project's existing background-task mechanism if present.

## 9. Tests and acceptance gates

**Unit tests**
- Feature preprocessing and encoding shape/range.
- QNN and QCNN input/output dimensions.
- QUBO construction, objective calculation and bitstring decoding.
- Matcher selector decision, unknown matcher and manual override.
- Backend creation, simulator execution and unavailable dependency handling.

**Integration tests**
- Matcher selector calls an actual registered matcher.
- QCNN feature output reaches the intended descriptor/feature stage.
- QNN scores reach correspondence ranking/filtering.
- QAOA-selected correspondences reach the existing geometric verification stage.
- Each module can be independently disabled.
- Full classical mode does not import or require optional quantum libraries and preserves baseline behaviour within established tolerances.

**Promotion gate:** A quantum module may be enabled in normal hybrid runs only after it has a reproducible experiment, a relevant classical baseline, held-out geographic evaluation, documented limitations and a safe fallback. A circuit that runs successfully is not, by itself, evidence of registration improvement.

## 10. Implementation order

1. Audit repository and reproduce the existing classical pipeline and tests.
2. Identify real matchers and add the transparent rule-based matcher selector plus benchmark logging.
3. Set up the shared local simulator backend and execute a small known circuit.
4. Implement QUBO formulation, exact-small-instance validation and greedy baseline.
5. Implement QAOA and connect decoded matches to existing geometric verification.
6. Implement QNN match reliability and compare with classical baselines.
7. Implement the small-patch QCNN, test encoding/resource constraints and compare with classical patch features.
8. Connect and ablate the complete hybrid workflow.
9. Add API/UI controls, tests, setup instructions and research documentation.
10. Run regression tests and final classical-versus-hybrid benchmarks.

Do not skip the audit or baseline stages. If data is not available for training/evaluation, implement the interface and unit tests but mark model training and performance evaluation as blocked—not completed.

## 11. Definition of done and final report

The work is complete only when:
- Existing classical registration still runs.
- Matcher selector chooses only real available matchers and has an override/fallback.
- QNN and QCNN have distinct working implementations and tested input/output contracts.
- The local IBM-compatible simulator executes circuits and is labelled accurately.
- QAOA produces decoded correspondence selections that reach existing geometric verification.
- All components are independently switchable.
- Experiments, metrics and limitations are recorded without fabricated results.
- The full hybrid workflow has integration tests.

At completion, report:
1. What was already present versus newly implemented.
2. Files created/changed and integration points.
3. Exact install, run and test commands.
4. Tests actually executed and their results.
5. Which simulator/backend was used; explicitly state if no real QPU was used.
6. Dataset, sensors and geographic splits used.
7. Classical-versus-hybrid results, including total runtime.
8. Incomplete work, blockers, limitations and next steps.

**Out of scope:** standalone QSVC, quantum annealing, QPSO, quantum graph neural networks, QFT, Grover, QPCA, quantum clustering and real IBM hardware execution.

## Official technical references

- Qiskit Machine Learning QNNs (`EstimatorQNN`, `SamplerQNN`, neural-network classifiers): https://github.com/qiskit-community/qiskit-machine-learning
- Qiskit Machine Learning QCNN tutorial: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html
- IBM Quantum local simulators (`AerSimulator`, fake backends): https://qiskit.qotlabs.org/docs/guides/local-simulators
- IBM Quantum QAOA tutorial: https://qiskit.qotlabs.org/docs/tutorials/quantum-approximate-optimization-algorithm
