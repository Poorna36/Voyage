# 06 — Roadmap and Tasks

## Phase 1: Core Subsystem Architecture & Simulation Foundations (Complete)
- [x] Config schema in `configs/quantum.yaml`
- [x] Backend abstraction in `quantum/backends.py` (Q6)
- [x] Record serialization schema in `quantum/records.py`
- [x] Service facade in `quantum/service.py`
- [x] Unit test suites for backends and records

## Phase 2: Quantum Algorithms Implementation (Complete)
- [x] Q0b QUBO consistency filter in `quantum/matcher_qubo.py`
- [x] Q1 QAOA spatial selection in `quantum/selection_qaoa.py`
- [x] Q2 Quantum kernel classifier and router in `quantum/kernel_classifier.py`
- [x] Q3 Grover consensus hypotheses search in `quantum/grover_consensus.py`
- [x] Q4 2D QFT phase correlation in `quantum/qft_phase_corr.py`
- [x] Q5 FRQI/NEQR and QHED edge detection in `quantum/image_encoding.py`
- [x] Q7 Post-quantum cryptographic signing in `quantum/pqc.py`

## Phase 3: REST API Bridge & Frontend Quantum Lab (Complete)
- [x] FastAPI route group `/api/quantum/*` in `api/routes/quantum.py`
- [x] Registered router in `api/server.py`
- [x] React types in `sih-dashboard/src/types/quantum.ts`
- [x] Client service in `sih-dashboard/src/services/quantumApi.ts`
- [x] Hook `useQuantumRun.ts`
- [x] Quantum Lab UI components and workbench integration
