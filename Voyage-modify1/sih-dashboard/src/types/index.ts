// SIH26166 — TypeScript Interfaces
// All types derived from INTERFACES.md and ARCHITECTURE.md schemas

export type SensorType = 'OHRC' | 'TMC-2' | 'IIRS' | 'PDS4';
export type MatcherType = 'sift' | 'rift2' | 'lightglue' | 'crater';
export type TerrainClass = 'polar_highland' | 'polar' | 'highland' | 'mare' | 'equatorial_mare';

export const PipelineStage = {
  Idle: 'idle',
  Ingesting: 'ingesting',
  GraphMatching: 'graph_matching',
  MAGSAC: 'magsac',
  Warping: 'warping',
  Done: 'done',
} as const;

export type PipelineStage = (typeof PipelineStage)[keyof typeof PipelineStage];

export interface ActiveProcessingState {
  status: 'idle' | 'processing' | 'completed' | 'error';
  pairName?: string;
  fileCount?: number;
  stageMessage?: string;
  newScene?: ScenePreset | null;
  errorMessage?: string;
}

export interface ScenePreset {
  id: string;
  name: string;
  lat: number;
  lon: number;
  height: number; // camera altitude in meters
  terrainClass?: TerrainClass;
  craterDensity?: number;
  solarIncidenceDeg?: number;
  solarAzimuthDeg?: number;
  gsdM?: number;
  overlayOpacity?: number;
  description: string;
}

export interface TelemetryData {
  rmsePx: number;
  ssim: number;
  inlierRatio: number;
  inlierCount: number;
  candidateCount: number;
  spatialCoverage: number;
  gridDensityStd: number;
  refinementGainPx: number;
  solarIncidenceDeg: number;
  solarEmissionDeg: number;
  solarAzimuthDeg: number;
  matcherWinner: MatcherType;
  pairId: string;
  utc: string;
  runtimeS: number;
  ladderLevel: number; // 0=similarity, 1=affine, 2=homography
  rotationDeg?: number;
  scaleFactor?: number;
  translationDxPx?: number;
  translationDyPx?: number;
  translationDxM?: number;
  translationDyM?: number;
  homographyMatrix?: number[][];
  matcherBenchmarks?: Record<
    string,
    {
      rmse_px?: number;
      rmsePx?: number;
      inlier_ratio?: number;
      inlierRatio?: number;
      inliers?: number;
      candidates?: number;
      status?: string;
      runtime_s?: number;
      runtimeS?: number;
    }
  >;
}

export interface SpectralDataPoint {
  wavelength: number; // µm
  reflectance: number; // 0-1
}

export interface SpectralData {
  pairId: string;
  sensor: SensorType;
  band: number;
  probeCoord: [number, number]; // [lon, lat]
  data: SpectralDataPoint[];
  absorptionTroughWavelength: number; // µm (3.0 for water-ice)
  absorptionDepth: number; // relative depth 0-1
}

export interface KeypointMatch {
  id: number;
  srcXy: [number, number]; // [col, row]
  refXy: [number, number];
  confidence: number;
  isInlier: boolean;
  isShadowOutlier: boolean;
  refinedDelta?: [number, number];
  refineSharpness?: number;
}

export interface ProcessingOptions {
  percentileClipping: boolean;
  clahe: boolean;
  morphologicalGradients: boolean;
  pcaBandReduction: boolean;
  selectedMatcher: MatcherType;
}

export interface SLZDiagnostic {
  slopeDeg: number;
  slopeThresholdDeg: number;
  slopePassRate: number;
  boulderClearanceM: number;
  boulderThresholdM: number;
  boulderPassRate: number;
  overallSafetyScore: number; // 0-100
  goNoGo: 'GO' | 'NO-GO' | 'MARGINAL';
  terrainRoughnessCm?: number;
  craterDensityKm2?: number;
  optimalLandingSite?: {
    lat: number;
    lon: number;
    elevationM?: number;
    hazardProbability?: number;
  };
}

export interface PipelineOptions {
  activeMatcher: MatcherType;
  adaptiveMsm: boolean;
  clahe: boolean;
  percentileClipping: boolean;
  morphologicalGradients: boolean;
  pcaBandReduction: boolean;
  ransacThreshold?: number;
  maxFeatures?: number;
  confidenceThreshold?: number;
}

export interface LayerVisibility {
  basemap: boolean;
  dem: boolean;
  waterIce: boolean;
  craters: boolean;
  grid: boolean;
  ohrc: boolean;
  tmc2Slope: boolean;
  iirsHyperspectral: boolean;
  slzOverlay: boolean;
}

export interface UploadedFile {
  name: string;
  size: number;
  sensor: SensorType;
  status: 'pending' | 'processing' | 'done' | 'ready' | 'error';
}

export interface CraterDetail {
  id: string;
  name: string;
  lat: number;
  lon: number;
  height: number;
  diameterKm: number;
  depthKm: number;
  region: string;
  floorInclinationDeg: number;
  wallSlopeDeg: number;
  orbitInclinationDeg: number;
  solarIncidenceDeg: number;
  solarAzimuthDeg: number;
  waterAbsorptionDepthPct: number;
  waterIceConcentrationWtPct: number;
  waterIcePpm: number;
  psrStatus: string;
  subsurfaceHydrationLevel: string;
  surfaceTempKelvin: number;
  frostIndex: number;
  spectrometerBand: number;
  description: string;
  // Direct backend snake_case aliases
  diameter_km?: number;
  depth_km?: number;
  floor_inclination_deg?: number;
  wall_slope_deg?: number;
  orbit_inclination_deg?: number;
  solar_incidence_deg?: number;
  solar_azimuth_deg?: number;
  water_absorption_depth_pct?: number;
  water_ice_concentration_wt_pct?: number;
  water_ice_ppm?: number;
  psr_status?: string;
  subsurface_hydration_level?: string;
  surface_temp_kelvin?: number;
  frost_index?: number;
  spectrometer_band?: number;
}

export const PIPELINE_STAGE_LABELS: Record<string, string> = {
  idle: 'Ready',
  ingesting: 'Ingesting & Calibrating (L0)',
  graph_matching: 'Graph Matching — LightGlue M2 (L2)',
  magsac: 'MAGSAC++ Geometric Verification (L4)',
  warping: 'Warping → GeoTIFF (L6)',
  done: 'Co-registration Complete',
};

export interface MatcherParameters {
  sift: {
    n_features: number;
    ratio_thresh: number;
    contrast_threshold?: number;
  };
  rift2: {
    n_scale: number;
    n_orient: number;
  };
  lnift: {
    patch_size: number;
  };
  lightglue: {
    max_num_keypoints: number;
  };
  crater: {
    model_path: string;
    min_diameter_px: number;
  };
}
