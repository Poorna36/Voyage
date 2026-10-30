/**
 * sih-dashboard/src/services/api.ts
 * ----------------------------------
 * Typed HTTP client for the SIH26166 Lunar Pipeline Backend API.
 * All methods include automatic fallback: if the backend is offline,
 * they return null and the calling hooks fall back to mock data.
 */

export const API_BASE = (
  import.meta.env.VITE_API_BASE_URL !== undefined && import.meta.env.VITE_API_BASE_URL !== ''
    ? import.meta.env.VITE_API_BASE_URL
    : (import.meta.env.DEV ? 'http://localhost:8000' : '')
).replace(/\/$/, '');
const TIMEOUT_MS = 5000;

// ── Generic fetcher with timeout + error swallowing ──

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.warn(`[API] ${path} responded with ${response.status}`);
      return null;
    }

    return (await response.json()) as T;
  } catch (err) {
    // Network error, timeout, or backend offline — silent fallback
    console.debug(`[API] Backend unavailable for ${path}:`, err instanceof Error ? err.message : err);
    return null;
  }
}


// ── Type Definitions (matching backend Pydantic models) ──

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
}

export interface FootprintSource {
  product_id: string;
  sensor: string;
  gsd_m: number;
  utc?: string;
  footprint_ll?: number[][];
  footprint_shape?: number[];
  solar_incidence_deg?: number;
  solar_azimuth_deg?: number;
}

export interface FootprintReference {
  product_id: string;
  gsd_m: number;
  type: string;
  footprint_ll?: number[][];
}

export interface PairSummary {
  pair_id: string;
  src: FootprintSource;
  ref: FootprintReference;
  overlap_fraction: number;
  terrain_class?: string;
  latitude_center_deg?: number;
  longitude_center_deg?: number;
  crater_density_per_km2?: number;
  split: string;
  created_at: string;
}

export interface DatasetStats {
  total_pairs: number;
  train_pairs: number;
  test_pairs: number;
  skipped_pairs: number;
  sensors: string[];
  terrain_classes: string[];
}

export interface MatchMetrics {
  pair_id: string;
  matcher?: string;
  rmse_px?: number;
  ssim?: number;
  inlier_ratio?: number;
  inlier_count?: number;
  candidate_count?: number;
  spatial_coverage?: number;
  runtime_s?: number;
  terrain_class?: string;
  src_sensor?: string;
  src_gsd_m?: number;
  ref_type?: string;
  ref_gsd_m?: number;
  solar_incidence_deg?: number;
  solar_azimuth_deg?: number;
  latitude_center_deg?: number;
  longitude_center_deg?: number;
  has_ground_truth: boolean;
}

export interface PipelineRunRequest {
  pair_id: string;
  matcher?: string;
  options?: Record<string, unknown>;
}

export interface PipelineResult {
  run_id: string;
  pair_id: string;
  status: string;
  matcher: string;
  stages_completed: string[];
  metrics?: Record<string, unknown>;
  runtime_s: number;
  timestamp: string;
}

export interface MatcherConfig {
  enabled?: boolean;
  [key: string]: unknown;
}

export interface MatchersConfig {
  sift?: MatcherConfig;
  rift2?: MatcherConfig;
  lnift?: MatcherConfig;
  lightglue?: MatcherConfig;
  crater?: MatcherConfig;
  arbitration?: Record<string, unknown>;
  selection?: Record<string, unknown>;
}


// ── API Methods ──

/** Check if backend is online */
export async function checkHealth(): Promise<HealthStatus | null> {
  return apiFetch<HealthStatus>('/api/health');
}

/** List all image pairs from the manifest */
export async function listPairs(): Promise<PairSummary[] | null> {
  return apiFetch<PairSummary[]>('/api/datasets/');
}

/** Get dataset statistics */
export async function getDatasetStats(): Promise<DatasetStats | null> {
  return apiFetch<DatasetStats>('/api/datasets/stats');
}

/** Get details for a specific pair */
export async function getPairDetails(pairId: string): Promise<Record<string, unknown> | null> {
  return apiFetch<Record<string, unknown>>(`/api/datasets/${encodeURIComponent(pairId)}`);
}

/** Get metrics for all pairs */
export async function listMetrics(): Promise<MatchMetrics[] | null> {
  return apiFetch<MatchMetrics[]>('/api/metrics/');
}

/** Get metrics for a specific pair */
export interface BackendSLZDiagnostic {
  slope_deg: number;
  slope_threshold_deg: number;
  slope_pass_rate: number;
  boulder_clearance_m: number;
  boulder_threshold_m: number;
  boulder_pass_rate: number;
  overall_safety_score: number;
  go_no_go: 'GO' | 'NO-GO' | 'MARGINAL';
  terrain_roughness_cm?: number;
  crater_density_km2?: number;
  optimal_landing_site?: {
    lat: number;
    lon: number;
    elevation_m?: number;
    hazard_probability?: number;
  };
}

export interface BackendSpectralData {
  pair_id: string;
  sensor: string;
  band: number;
  probe_coord: [number, number];
  data: Array<{ wavelength: number; reflectance: number }>;
  absorption_trough_wavelength: number;
  absorption_depth: number;
}

export interface BackendKeypointMatch {
  id: number;
  src_xy: [number, number];
  ref_xy: [number, number];
  confidence: number;
  is_inlier: boolean;
  is_shadow_outlier: boolean;
  refined_delta: [number, number];
  refine_sharpness: number;
}

export interface BackendCraterDetail {
  id: string;
  name: string;
  lat: number;
  lon: number;
  height: number;
  diameter_km: number;
  depth_km: number;
  region: string;
  floor_inclination_deg: number;
  wall_slope_deg: number;
  orbit_inclination_deg: number;
  solar_incidence_deg: number;
  solar_azimuth_deg: number;
  water_absorption_depth_pct: number;
  water_ice_concentration_wt_pct: number;
  water_ice_ppm: number;
  psr_status: string;
  subsurface_hydration_level: string;
  surface_temp_kelvin: number;
  frost_index: number;
  spectrometer_band: number;
  description: string;
}

export async function getPairMetrics(pairId: string): Promise<MatchMetrics | null> {
  return apiFetch<MatchMetrics>(`/api/metrics/${encodeURIComponent(pairId)}`);
}

/** Get ground-truth data for a pair */
export async function getGroundTruth(pairId: string): Promise<Record<string, unknown> | null> {
  return apiFetch<Record<string, unknown>>(`/api/metrics/ground-truth/${encodeURIComponent(pairId)}`);
}

/** Run the pipeline on a specific pair */
export async function runPipeline(request: PipelineRunRequest): Promise<PipelineResult | null> {
  return apiFetch<PipelineResult>('/api/pipeline/run', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/** Check pipeline run status */
export async function getPipelineStatus(runId: string): Promise<PipelineResult | null> {
  return apiFetch<PipelineResult>(`/api/pipeline/status/${encodeURIComponent(runId)}`);
}

/** Get all completed pipeline runs */
export async function getPipelineHistory(): Promise<PipelineResult[] | null> {
  return apiFetch<PipelineResult[]>('/api/pipeline/history');
}

/** Get matcher configuration */
export async function getMatchersConfig(): Promise<MatchersConfig | null> {
  return apiFetch<MatchersConfig>('/api/config/matchers');
}

/** Update and persist matcher configuration */
export async function updateMatchersConfig(config: MatchersConfig): Promise<Record<string, unknown> | null> {
  return apiFetch<Record<string, unknown>>('/api/config/matchers', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/** Get all configs merged */
export async function getAllConfigs(): Promise<Record<string, unknown> | null> {
  return apiFetch<Record<string, unknown>>('/api/config/all');
}

/** Get sensor-specific config */
export async function getSensorConfig(sensorName: string): Promise<Record<string, unknown> | null> {
  return apiFetch<Record<string, unknown>>(`/api/config/sensor/${encodeURIComponent(sensorName)}`);
}

/** Get Safe Landing Zone diagnostics for a scene */
export async function getSlzData(sceneId: string): Promise<BackendSLZDiagnostic | null> {
  return apiFetch<BackendSLZDiagnostic>(`/api/science/slz/${encodeURIComponent(sceneId)}`);
}

/** Get IIRS hyperspectral curve and 3.0µm water absorption */
export async function getSpectralData(sceneId: string): Promise<BackendSpectralData | null> {
  return apiFetch<BackendSpectralData>(`/api/science/spectral/${encodeURIComponent(sceneId)}`);
}

/** Get 2D correspondence keypoints */
export async function getKeypointMatches(pairId: string): Promise<BackendKeypointMatch[] | null> {
  return apiFetch<BackendKeypointMatch[]>(`/api/science/keypoints/${encodeURIComponent(pairId)}`);
}

export interface BackendTelemetryDiagnostic {
  pair_id: string;
  rmse_px: number;
  ssim: number;
  inlier_ratio: number;
  inlier_count: number;
  candidate_count: number;
  spatial_coverage: number;
  grid_density_std: number;
  refinement_gain_px: number;
  solar_incidence_deg: number;
  solar_emission_deg: number;
  solar_azimuth_deg: number;
  matcher_winner: string;
  runtime_s: number;
  ladder_level: number;
  utc?: string;
  homography_matrix?: number[][];
  translation_dx_px?: number;
  translation_dy_px?: number;
  translation_dx_m?: number;
  translation_dy_m?: number;
  rotation_deg?: number;
  scale_factor?: number;
  matcher_benchmarks?: Record<
    string,
    {
      rmse_px?: number;
      inlier_ratio?: number;
      inliers?: number;
      candidates?: number;
      status?: string;
      runtime_s?: number;
    }
  >;
}

/** Get authentic co-registration telemetry */
export async function getTelemetryData(pairId: string): Promise<BackendTelemetryDiagnostic | null> {
  return apiFetch<BackendTelemetryDiagnostic>(`/api/science/telemetry/${encodeURIComponent(pairId)}`);
}

export interface CropDataResponse {
  pair_id: string;
  zoom_level: number;
  norm_coord: [number, number];
  effective_gsd_m: number;
  scale_cm_per_px: number;
  src_crop_base64: string;
  ref_crop_base64: string;
  dn_stats: {
    min: number;
    max: number;
    mean: number;
    std: number;
    bit_depth: string;
  };
  local_slz: {
    slope_deg: number;
    slope_pass_rate: number;
    hazard_rating: string;
  };
  local_rmse_px: number;
  local_keypoints: Array<{
    id: number;
    src_xy: [number, number];
    ref_xy: [number, number];
    confidence: number;
    is_inlier: boolean;
  }>;
}

/** Fetch dynamic memory-mapped lossless deep-zoom sub-pixel crop */
export async function getCropData(
  pairId: string,
  normX: number,
  normY: number,
  cropSize = 512,
  zoom = 2.0
): Promise<CropDataResponse | null> {
  return apiFetch<CropDataResponse>(
    `/api/datasets/${encodeURIComponent(pairId)}/crop?norm_x=${normX}&norm_y=${normY}&crop_size=${cropSize}&zoom=${zoom}`
  );
}

/** Get full lunar crater catalog, optionally with search query */
export async function getCraterCatalog(query?: string): Promise<BackendCraterDetail[] | null> {
  const url = query && query.trim() ? `/api/science/craters/?q=${encodeURIComponent(query.trim())}` : '/api/science/craters/';
  return apiFetch<BackendCraterDetail[]>(url);
}

/** Upload user-provided mission imagery files with role selection.
 * Sensor is used ONLY for verification against detected file metadata.
 * Lat/lon coordinates are NEVER sent from the browser — they are always
 * extracted exclusively from the uploaded PDS-4 XML or HTML metadata files.
 */
export async function uploadMissionFiles(
  files: File[],
  pairName?: string,
  sensor?: string,
  roles?: string[]
): Promise<{ status: string; pair_id: string; name: string; message: string; pair?: any }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);

  try {
    const formData = new FormData();
    files.forEach((file) => {
      // Preserve relative directory path (from folder upload) as filename
      const relativePath = (file as any).webkitRelativePath || file.name;
      formData.append('files', file, relativePath);
    });
    if (pairName && pairName.trim()) formData.append('pair_name', pairName.trim());
    // Sensor appended as verify-hint only (backend will reject if mismatch detected)
    if (sensor && sensor.trim()) formData.append('sensor', sensor.trim());
    if (roles && roles.length > 0) formData.append('roles', roles.join(','));
    // NOTE: lat/lon are intentionally NOT appended — coordinates come from metadata files only

    const res = await fetch(`${API_BASE}/api/datasets/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `Upload failed (status: ${res.status})` }));
      throw new Error(err.detail || `Upload failed (status: ${res.status})`);
    }

    return await res.json();
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error('Upload timed out after 5 minutes. Please check your network or try smaller files.');
    }
    console.error('Failed to upload mission files:', error);
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}



