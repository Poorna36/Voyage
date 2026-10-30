// sih-dashboard/src/services/quantumApi.ts
import { API_BASE } from './api';
import type {
  QuantumStatusResponse,
  QuantumRunRequest,
  QuantumRunStatus,
  CircuitData,
  QuantumRunRecord,
} from '../types/quantum';

const BASE_URL = `${API_BASE}/api/quantum`;

export async function fetchQuantumStatus(): Promise<QuantumStatusResponse> {
  const res = await fetch(`${BASE_URL}/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch quantum status: ${res.statusText}`);
  }
  return res.json();
}

export async function startQuantumRun(req: QuantumRunRequest): Promise<{ run_id: string; message: string }> {
  const res = await fetch(`${BASE_URL}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (res.status === 403) {
    const err = await res.json();
    throw new Error(err.detail || 'Promotion gate error: Module cannot run in active mode');
  }
  if (res.status === 409) {
    const err = await res.json();
    throw new Error(err.detail || 'Backend unavailable');
  }
  if (res.status === 404) {
    const err = await res.json();
    throw new Error(err.detail || 'Pair not found in manifest or presets');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Quantum run request failed (${res.status})`);
  }
  return res.json();
}

export async function getQuantumRunStatus(runId: string): Promise<QuantumRunStatus> {
  const res = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) {
    throw new Error(`Failed to query run ${runId}: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchQuantumRunsHistory(params?: {
  pair_id?: string;
  module?: string;
  limit?: number;
}): Promise<QuantumRunRecord[]> {
  const query = new URLSearchParams();
  if (params?.pair_id) query.append('pair_id', params.pair_id);
  if (params?.module) query.append('module', params.module);
  if (params?.limit) query.append('limit', String(params.limit));

  const res = await fetch(`${BASE_URL}/runs?${query.toString()}`);
  if (!res.ok) {
    return [];
  }
  const data = await res.json();
  return data.runs || [];
}

export async function fetchCircuitData(module: string): Promise<CircuitData> {
  const res = await fetch(`${BASE_URL}/circuit/${encodeURIComponent(module)}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch circuit for ${module}`);
  }
  return res.json();
}

export async function fetchLatestBenchmarks(): Promise<any> {
  const res = await fetch(`${BASE_URL}/benchmarks/latest`);
  if (!res.ok) {
    throw new Error('Failed to fetch benchmark matrix');
  }
  return res.json();
}

export async function signProductArtifact(payload: { file_path?: string; payload?: string }): Promise<any> {
  const res = await fetch(`${BASE_URL}/sign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Signing failed');
  }
  return res.json();
}

export async function verifyProductArtifact(payload: {
  signature_b64: string;
  file_path?: string;
  payload?: string;
}): Promise<{ valid: boolean; message: string; algorithm: string }> {
  const res = await fetch(`${BASE_URL}/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Verification failed');
  }
  return res.json();
}
