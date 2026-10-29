/**
 * sih-dashboard/src/hooks/useBackendData.ts
 * -------------------------------------------
 * Primary React hook providing live data from the backend API.
 * Connects directly to the FastAPI server at localhost:8000.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  checkHealth,
  listPairs,
  getDatasetStats,
  getMatchersConfig,
  updateMatchersConfig,
  getPipelineHistory,
  getSensorConfig,
  runPipeline,
  getCraterCatalog,
  getSlzData,
  getSpectralData,
  getKeypointMatches,
  type PairSummary,
  type DatasetStats,
  type MatchersConfig,
  type PipelineResult,
  type PipelineRunRequest,
  type BackendSLZDiagnostic,
  type BackendSpectralData,
  type BackendKeypointMatch,
  type BackendCraterDetail,
} from '../services/api';
import type { CraterDetail } from '../types';

export interface BackendData {
  isBackendOnline: boolean;
  isLoading: boolean;
  backendLatencyMs: number | null;
  craters: CraterDetail[];
  backendPairs: PairSummary[];
  datasetStats: DatasetStats | null;
  matcherConfig: MatchersConfig | null;
  pipelineHistory: PipelineResult[];
  lastPipelineResult: PipelineResult | null;
  fetchSensorConfig: (sensorName: string) => Promise<Record<string, unknown> | null>;
  saveMatcherConfig: (config: MatchersConfig) => Promise<boolean>;
  runPipelineOnPair: (request: PipelineRunRequest) => Promise<PipelineResult | null>;
  fetchSlzData: (sceneId: string) => Promise<BackendSLZDiagnostic | null>;
  fetchSpectralData: (sceneId: string) => Promise<BackendSpectralData | null>;
  fetchKeypointMatches: (pairId: string) => Promise<BackendKeypointMatch[] | null>;
  refreshData: () => void;
}

export function useBackendData(): BackendData {
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [backendLatencyMs, setBackendLatencyMs] = useState<number | null>(null);
  const [craters, setCraters] = useState<CraterDetail[]>([]);
  const [backendPairs, setBackendPairs] = useState<PairSummary[]>([]);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [matcherConfig, setMatcherConfig] = useState<MatchersConfig | null>(null);
  const [pipelineHistory, setPipelineHistory] = useState<PipelineResult[]>([]);
  const [lastPipelineResult, setLastPipelineResult] = useState<PipelineResult | null>(null);
  const mountedRef = useRef(true);

  const fetchAll = useCallback(async () => {
    setIsLoading(true);

    const t0 = performance.now();
    const health = await checkHealth();
    const elapsed = Math.round(performance.now() - t0);
    const online = health?.status === 'online';

    if (!mountedRef.current) return;
    setIsBackendOnline(online);
    setBackendLatencyMs(online ? elapsed : null);

    if (online) {
      // Fetch all data in parallel from the backend
      const [craterCatalog, pairs, stats, matchers, history] = await Promise.all([
        getCraterCatalog(),
        listPairs(),
        getDatasetStats(),
        getMatchersConfig(),
        getPipelineHistory(),
      ]);

      if (!mountedRef.current) return;

      if (craterCatalog && craterCatalog.length > 0) {
        // Map backend crater detail to frontend CraterDetail type
        const mappedCraters: CraterDetail[] = craterCatalog.map((c: BackendCraterDetail) => ({
          id: c.id,
          name: c.name,
          lat: c.lat,
          lon: c.lon,
          height: c.height,
          diameterKm: c.diameter_km,
          depthKm: c.depth_km,
          region: c.region,
          floorInclinationDeg: c.floor_inclination_deg,
          wallSlopeDeg: c.wall_slope_deg,
          orbitInclinationDeg: c.orbit_inclination_deg,
          solarIncidenceDeg: c.solar_incidence_deg,
          solarAzimuthDeg: c.solar_azimuth_deg,
          waterAbsorptionDepthPct: c.water_absorption_depth_pct,
          waterIceConcentrationWtPct: c.water_ice_concentration_wt_pct,
          waterIcePpm: c.water_ice_ppm,
          psrStatus: c.psr_status as any,
          subsurfaceHydrationLevel: c.subsurface_hydration_level as any,
          surfaceTempKelvin: c.surface_temp_kelvin,
          frostIndex: c.frost_index,
          spectrometerBand: c.spectrometer_band,
          description: c.description,
        }));
        setCraters(mappedCraters);
      }

      if (pairs) setBackendPairs(pairs);
      if (stats) setDatasetStats(stats);
      if (matchers) setMatcherConfig(matchers);
      if (history) setPipelineHistory(history);
    }

    if (mountedRef.current) setIsLoading(false);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchAll();

    // Re-check backend health every 15 seconds
    const interval = setInterval(async () => {
      const t0 = performance.now();
      const health = await checkHealth();
      const elapsed = Math.round(performance.now() - t0);
      const online = health?.status === 'online';
      if (mountedRef.current) {
        setIsBackendOnline(online);
        setBackendLatencyMs(online ? elapsed : null);
      }
    }, 15000);

    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [fetchAll]);

  const fetchSensorConfig = useCallback(async (sensorName: string) => {
    return getSensorConfig(sensorName);
  }, []);

  const saveMatcherConfig = useCallback(async (newConfig: MatchersConfig) => {
    const res = await updateMatchersConfig(newConfig);
    if (res && (res as any).status === 'success') {
      setMatcherConfig(newConfig);
      return true;
    }
    return false;
  }, []);

  const runPipelineOnPair = useCallback(async (request: PipelineRunRequest) => {
    const result = await runPipeline(request);
    if (result && mountedRef.current) {
      setLastPipelineResult(result);
      setPipelineHistory((prev) => [result, ...prev]);
    }
    return result;
  }, []);

  const fetchSlzData = useCallback(async (sceneId: string) => {
    return getSlzData(sceneId);
  }, []);

  const fetchSpectralData = useCallback(async (sceneId: string) => {
    return getSpectralData(sceneId);
  }, []);

  const fetchKeypointMatches = useCallback(async (pairId: string) => {
    return getKeypointMatches(pairId);
  }, []);

  return {
    isBackendOnline,
    isLoading,
    backendLatencyMs,
    craters,
    backendPairs,
    datasetStats,
    matcherConfig,
    pipelineHistory,
    lastPipelineResult,
    fetchSensorConfig,
    saveMatcherConfig,
    runPipelineOnPair,
    fetchSlzData,
    fetchSpectralData,
    fetchKeypointMatches,
    refreshData: fetchAll,
  };
}
