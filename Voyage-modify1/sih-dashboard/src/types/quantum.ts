// sih-dashboard/src/types/quantum.ts

export type ModuleId =
  | 'Q0_filter'
  | 'Q1_selection'
  | 'Q2_router'
  | 'Q2_inlier'
  | 'Q3_consensus'
  | 'Q4_phase_corr'
  | 'Q5_encoding'
  | 'Q7_signing';

export type BackendKind = 'simulator' | 'annealer' | 'hardware';

export interface BackendInfo {
  id: string;
  kind: BackendKind;
  max_qubits: number;
  available: boolean;
  reason?: string | null;
}

export interface ModuleStatusInfo {
  id: ModuleId;
  name: string;
  mode: 'off' | 'shadow' | 'active';
  available: boolean;
  promoted: boolean;
}

export interface QuantumStatusResponse {
  enabled: boolean;
  global_mode: 'off' | 'shadow' | 'active';
  limits: {
    max_qubits: number;
    max_runtime_s: number;
  };
  modules: ModuleStatusInfo[];
  backends: BackendInfo[];
  versions: {
    voyage_quantum: string;
    numpy: string;
  };
}

export interface QuantumRunRecord {
  run_id: string;
  pair_id: string;
  module: ModuleId | string;
  mode: 'off' | 'shadow' | 'active';
  backend: string;
  seed: number;
  problem: Record<string, any>;
  metrics: Record<string, any>;
  classical_twin: {
    name: string;
    runtime_s?: number;
    [key: string]: any;
  };
  circuit?: {
    qubits?: number;
    depth?: number;
    gate_counts?: Record<string, number>;
    gates_sequence?: Array<{ gate: string; target?: number; targets?: number[]; phi?: number; param?: number }>;
    [key: string]: any;
  } | null;
  runtime_s: number;
  fallback: {
    used: boolean;
    reason?: string | null;
  };
  versions: Record<string, string>;
  created_utc: string;
}

export interface QuantumRunRequest {
  pair_id: string;
  module: ModuleId;
  backend?: string;
  mode?: 'shadow' | 'active';
  seed?: number;
  params?: Record<string, any>;
  inputs?: Record<string, any>;
}

export interface QuantumRunStatus {
  run_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'timeout';
  progress?: number | null;
  record?: QuantumRunRecord | null;
  error?: string | null;
}

export interface CircuitData {
  module: string;
  qubits: number;
  total_gates: number;
  gate_counts: Record<string, number>;
  gate_sequence: Array<{ gate: string; target?: number; targets?: number[]; phi?: number; param?: number }>;
}
