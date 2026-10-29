import { useState, useMemo, useEffect } from 'react';
import { LunarisLanding } from './components/landing/LunarisLanding';
import { MoonWorkbenchOverlay } from './components/MoonWorkbenchOverlay';
import { LunarTargetPalette } from './components/LunarTargetPalette';
import { EngineInspector } from './components/EngineInspector';
import { KeypointViewer } from './components/KeypointViewer';
import { ResultsView } from './components/ResultsView';
import { AddFilesModal } from './components/AddFilesModal';
import { WebGlMoonViewer } from './components/landing/WebGlMoonViewer';
import { useBackendData } from './hooks/useBackendData';
import { getSlzData, getSpectralData, getTelemetryData, uploadMissionFiles, API_BASE } from './services/api';
import {
  PipelineStage,
  type PipelineOptions,
  type MatcherParameters,
  type ScenePreset,
  type LayerVisibility,
  type TelemetryData,
  type SLZDiagnostic,
  type SpectralData,
  type ActiveProcessingState,
} from './types';

const DEFAULT_SCENE: ScenePreset = {
  id: 'boguslawsky',
  name: 'Boguslawsky Crater (South Pole)',
  lat: -72.8,
  lon: 43.1,
  height: 80000,
  terrainClass: 'polar_highland',
  craterDensity: 4.7,
  solarIncidenceDeg: 68.2,
  solarAzimuthDeg: 178.5,
  gsdM: 0.31,
  overlayOpacity: 0.75,
  description: 'Primary Chandrayaan-4 SLZ target corridor',
};

const DEFAULT_TELEMETRY: TelemetryData = {
  rmsePx: 0.36,
  ssim: 0.89,
  inlierRatio: 0.58,
  inlierCount: 26,
  candidateCount: 45,
  spatialCoverage: 0.82,
  gridDensityStd: 2.3,
  refinementGainPx: 0.23,
  solarIncidenceDeg: 68.2,
  solarEmissionDeg: 2.1,
  solarAzimuthDeg: 178.5,
  matcherWinner: 'lightglue',
  pairId: 'boguslawsky',
  utc: '2020-08-27T00:30:10.884Z',
  runtimeS: 4.8,
  ladderLevel: 2,
};

const DEFAULT_SLZ: SLZDiagnostic = {
  slopeDeg: 6.8,
  slopeThresholdDeg: 10,
  slopePassRate: 0.942,
  boulderClearanceM: 3.2,
  boulderThresholdM: 2.0,
  boulderPassRate: 0.97,
  overallSafetyScore: 94.2,
  goNoGo: 'GO',
};

const DEFAULT_SPECTRAL: SpectralData = {
  pairId: 'boguslawsky',
  sensor: 'IIRS',
  band: 187,
  probeCoord: [43.1, -72.8],
  data: [
    { wavelength: 2.6, reflectance: 0.28 },
    { wavelength: 2.7, reflectance: 0.27 },
    { wavelength: 2.8, reflectance: 0.24 },
    { wavelength: 2.85, reflectance: 0.21 },
    { wavelength: 2.9, reflectance: 0.17 },
    { wavelength: 2.95, reflectance: 0.12 },
    { wavelength: 3.0, reflectance: 0.08 },
    { wavelength: 3.05, reflectance: 0.11 },
    { wavelength: 3.1, reflectance: 0.16 },
    { wavelength: 3.15, reflectance: 0.22 },
    { wavelength: 3.2, reflectance: 0.26 },
    { wavelength: 3.3, reflectance: 0.29 },
  ],
  absorptionTroughWavelength: 3.0,
  absorptionDepth: 0.142,
};

export function App() {
  // Route State: Landing Page vs 3D Mission Control Workbench
  const [appMode, setAppMode] = useState<'landing' | 'workbench'>(() => {
    if (typeof window !== 'undefined' && window.location.hash === '#workbench') {
      return 'workbench';
    }
    return 'landing';
  });

  useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === '#workbench') {
        setAppMode('workbench');
      } else {
        setAppMode('landing');
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Backend Live API Data Integration
  const {
    isBackendOnline,
    craters,
    backendPairs,
    datasetStats,
    matcherConfig,
    lastPipelineResult,
    saveMatcherConfig,
    runPipelineOnPair,
    refreshData,
  } = useBackendData();

  // Workbench State
  const [selectedScene, setSelectedScene] = useState<ScenePreset>(DEFAULT_SCENE);
  const [activeCenterTab, setActiveCenterTab] = useState<'3d' | '2d' | 'results'>('3d');
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>(PipelineStage.Idle);
  const [isTargetPaletteOpen, setIsTargetPaletteOpen] = useState(false);
  const [isEngineInspectorOpen, setIsEngineInspectorOpen] = useState(false);
  const [isAddFilesOpen, setIsAddFilesOpen] = useState(false);
  const [cameraZoom, setCameraZoom] = useState<number>(3.8);
  const [processingState, setProcessingState] = useState<ActiveProcessingState | null>(null);

  const handleLaunchFromLanding = () => {
    setAppMode('workbench');
    window.location.hash = 'workbench';
  };

  const handleBackToLanding = () => {
    setAppMode('landing');
    window.location.hash = 'landing';
  };

  const handleZoomIn = () => {
    setCameraZoom((prev) => Math.max(2.2, prev - 0.4));
  };

  const handleZoomOut = () => {
    setCameraZoom((prev) => Math.min(6.5, prev + 0.4));
  };

  const handleResetView = () => {
    setCameraZoom(3.8);
    setSelectedScene(DEFAULT_SCENE);
  };

  // Live scientific data from backend
  const [liveSlz, setLiveSlz] = useState<SLZDiagnostic | null>(null);
  const [liveSpectral, setLiveSpectral] = useState<SpectralData | null>(null);
  const [liveTelemetry, setLiveTelemetry] = useState<TelemetryData | null>(null);
  const [isLoadingScience, setIsLoadingScience] = useState<boolean>(false);

  // Pair sensor metadata for dynamic labels in KeypointViewer
  const [pairSensorInfo, setPairSensorInfo] = useState<{
    srcSensor: string; srcGsd: string; refSensor: string; refGsd: string;
  } | null>(null);

  useEffect(() => {
    // First try to find in already-fetched backendPairs list
    const found = backendPairs.find(
      (p) => p.pair_id === selectedScene.id || p.pair_id.toLowerCase() === selectedScene.id.toLowerCase()
    );
    if (found) {
      setPairSensorInfo({
        srcSensor: `CH-2 ${found.src.sensor}`,
        srcGsd: `${found.src.gsd_m}m/px`,
        refSensor: `LRO ${found.ref.type || 'NAC'}`,
        refGsd: `${found.ref.gsd_m}m/px`,
      });
      return;
    }
    // Fallback: fetch directly from /api/datasets/{pair_id}
    if (!isBackendOnline) return;
    fetch(`${API_BASE}/api/datasets/${encodeURIComponent(selectedScene.id)}`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (!data) return;
        const srcSensor = data?.src?.sensor || 'OHRC';
        const srcGsd = data?.src?.gsd_m ? `${data.src.gsd_m}m/px` : '0.31m/px';
        const refType = data?.ref?.type || 'NAC';
        const refGsd = data?.ref?.gsd_m ? `${data.ref.gsd_m}m/px` : '0.5m/px';
        setPairSensorInfo({
          srcSensor: `CH-2 ${srcSensor}`,
          srcGsd,
          refSensor: `LRO ${refType}`,
          refGsd,
        });
      })
      .catch(() => setPairSensorInfo(null));
  }, [selectedScene.id, backendPairs, isBackendOnline]);

  useEffect(() => {
    if (!isBackendOnline) {
      setLiveSlz(null);
      setLiveSpectral(null);
      setLiveTelemetry(null);
      setIsLoadingScience(false);
      return;
    }

    let isMounted = true;
    setIsLoadingScience(true);

    Promise.all([
      getSlzData(selectedScene.id),
      getSpectralData(selectedScene.id),
      getTelemetryData(selectedScene.id),
    ])
      .then(([slzData, spectralData, telemetryData]) => {
        if (!isMounted) return;

        if (slzData) {
          setLiveSlz({
            slopeDeg: slzData.slope_deg,
            slopeThresholdDeg: slzData.slope_threshold_deg,
            slopePassRate: slzData.slope_pass_rate,
            boulderClearanceM: slzData.boulder_clearance_m,
            boulderThresholdM: slzData.boulder_threshold_m,
            boulderPassRate: slzData.boulder_pass_rate,
            overallSafetyScore: slzData.overall_safety_score,
            goNoGo: slzData.go_no_go as any,
            terrainRoughnessCm: slzData.terrain_roughness_cm,
            craterDensityKm2: slzData.crater_density_km2,
            optimalLandingSite: slzData.optimal_landing_site,
          });
        }

        if (spectralData) {
          setLiveSpectral({
            pairId: spectralData.pair_id,
            sensor: spectralData.sensor as any,
            band: spectralData.band,
            probeCoord: spectralData.probe_coord,
            data: spectralData.data,
            absorptionTroughWavelength: spectralData.absorption_trough_wavelength,
            absorptionDepth: spectralData.absorption_depth,
          });
        }

        if (telemetryData) {
          setLiveTelemetry({
            rmsePx: telemetryData.rmse_px,
            ssim: telemetryData.ssim,
            inlierRatio: telemetryData.inlier_ratio,
            inlierCount: telemetryData.inlier_count,
            candidateCount: telemetryData.candidate_count,
            spatialCoverage: telemetryData.spatial_coverage,
            gridDensityStd: telemetryData.grid_density_std,
            refinementGainPx: telemetryData.refinement_gain_px,
            solarIncidenceDeg: telemetryData.solar_incidence_deg,
            solarEmissionDeg: telemetryData.solar_emission_deg,
            solarAzimuthDeg: telemetryData.solar_azimuth_deg,
            matcherWinner: telemetryData.matcher_winner as any,
            pairId: telemetryData.pair_id,
            utc: telemetryData.utc || new Date().toISOString(),
            runtimeS: telemetryData.runtime_s,
            ladderLevel: telemetryData.ladder_level,
            homographyMatrix: telemetryData.homography_matrix,
            translationDxPx: telemetryData.translation_dx_px,
            translationDyPx: telemetryData.translation_dy_px,
            translationDxM: telemetryData.translation_dx_m,
            translationDyM: telemetryData.translation_dy_m,
            rotationDeg: telemetryData.rotation_deg,
            scaleFactor: telemetryData.scale_factor,
            matcherBenchmarks: telemetryData.matcher_benchmarks,
          });
        }

        setIsLoadingScience(false);
      })
      .catch((err) => {
        console.warn('Failed to fetch authentic lunar science telemetry:', err);
        if (isMounted) setIsLoadingScience(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedScene.id, isBackendOnline]);

  const activeTelemetry = useMemo(() => {
    const base = liveTelemetry || DEFAULT_TELEMETRY;
    if (lastPipelineResult && (lastPipelineResult.pair_id === selectedScene.id || selectedScene.id.includes(lastPipelineResult.pair_id)) && lastPipelineResult.metrics) {
      const m = lastPipelineResult.metrics as Record<string, any>;
      return {
        ...base,
        rmsePx: typeof m.rmse_px === 'number' ? m.rmse_px : base.rmsePx,
        ssim: typeof m.ssim === 'number' ? m.ssim : base.ssim,
        inlierRatio: typeof m.inlier_ratio === 'number' ? m.inlier_ratio : base.inlierRatio,
        inlierCount: typeof m.inlier_count === 'number' ? m.inlier_count : base.inlierCount,
        candidateCount: typeof m.candidate_count === 'number' ? m.candidate_count : base.candidateCount,
        spatialCoverage: typeof m.spatial_coverage === 'number' ? m.spatial_coverage : base.spatialCoverage,
        runtimeS: typeof lastPipelineResult.runtime_s === 'number' ? lastPipelineResult.runtime_s : base.runtimeS,
        matcherWinner: (lastPipelineResult.matcher as any) || base.matcherWinner,
      };
    }
    return base;
  }, [selectedScene.id, liveTelemetry, lastPipelineResult]);

  const currentSlz = liveSlz || DEFAULT_SLZ;
  const currentSpectral = liveSpectral || DEFAULT_SPECTRAL;

  const [options, setOptions] = useState<PipelineOptions>({
    activeMatcher: 'sift',
    adaptiveMsm: true,
    clahe: true,
    percentileClipping: true,
    morphologicalGradients: true,
    pcaBandReduction: true,
    ransacThreshold: 1.5,
    maxFeatures: 2000,
    confidenceThreshold: 0.8,
  });

  const [layers, setLayers] = useState<LayerVisibility>({
    basemap: true,
    dem: false,
    waterIce: false,
    craters: false,
    grid: false,
    ohrc: true,
    tmc2Slope: false,
    iirsHyperspectral: true,
    slzOverlay: true,
  });

  const [matcherParams, setMatcherParams] = useState<MatcherParameters>({
    sift: { n_features: 2000, ratio_thresh: 0.75 },
    rift2: { n_scale: 4, n_orient: 6 },
    lnift: { patch_size: 32 },
    lightglue: { max_num_keypoints: 2048 },
    crater: { model_path: 'models/crater_real_best.pt', min_diameter_px: 8 },
  });

  useEffect(() => {
    if (matcherConfig) {
      setMatcherParams(matcherConfig as unknown as MatcherParameters);
    }
  }, [matcherConfig]);

  // Pipeline execution handler calling backend runPipelineOnPair
  const handleRunPipeline = async () => {
    if (pipelineStage !== PipelineStage.Idle && pipelineStage !== PipelineStage.Done) return;
    setPipelineStage(PipelineStage.Ingesting);

    if (isBackendOnline) {
      const t1 = setTimeout(() => setPipelineStage(PipelineStage.GraphMatching), 500);
      const t2 = setTimeout(() => setPipelineStage(PipelineStage.MAGSAC), 1000);
      const t3 = setTimeout(() => setPipelineStage(PipelineStage.Warping), 1500);

      try {
        await runPipelineOnPair({
          pair_id: selectedScene.id,
          matcher: options.activeMatcher,
          options: {
            percentile_clipping: options.percentileClipping,
            clahe: options.clahe,
            morphological_gradients: options.morphologicalGradients,
            pca_band_reduction: options.pcaBandReduction,
          },
        });
      } catch (err) {
        console.warn('Backend run failed:', err);
      } finally {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
        setPipelineStage(PipelineStage.Done);
      }
    } else {
      setTimeout(() => setPipelineStage(PipelineStage.GraphMatching), 700);
      setTimeout(() => setPipelineStage(PipelineStage.MAGSAC), 1400);
      setTimeout(() => setPipelineStage(PipelineStage.Warping), 2100);
      setTimeout(() => setPipelineStage(PipelineStage.Done), 2800);
    }
  };

  // Ingest & background processing handler for user-provided folders/files
  const handleStartUpload = async (
    files: File[],
    pairName: string,
    sensor: string,
    fileRoles: ('src' | 'ref')[]
  ) => {
    // 1. Instantly return user to the home page with the Moon
    setAppMode('workbench');
    setActiveCenterTab('3d');
    setIsAddFilesOpen(false);
    window.location.hash = 'workbench';

    const displayName = pairName.trim() || (files[0] ? files[0].name.replace(/\.[^/.]+$/, '') : 'Mission Dataset');

    // 2. Set processing state and pipeline stage
    setProcessingState({
      status: 'processing',
      pairName: displayName,
      fileCount: files.length,
      stageMessage: 'Memmapping PDS-4 files & validating metadata...',
      newScene: null,
    });
    setPipelineStage(PipelineStage.Ingesting);

    try {
      // Background upload
      const res = await uploadMissionFiles(files, pairName, sensor, fileRoles);

      if (res && res.status === 'success' && res.pair_id) {
        const newScene: ScenePreset = {
          id: res.pair_id,
          name: res.name || res.pair_id.replace(/_/g, ' ').toUpperCase(),
          lat: res.pair?.latitude_center_deg ?? -70.0,
          lon: res.pair?.longitude_center_deg ?? 35.0,
          height: 80000,
          terrainClass: (res.pair?.terrain_class as any) ?? 'polar_highland',
          solarIncidenceDeg: 66.0,
          solarAzimuthDeg: 175.0,
          gsdM: sensor === 'OHRC' ? 0.31 : 0.50,
          description: `User-ingested mission pair (${files.length} files, verified ${sensor})`,
        };

        // Advance pipeline stage and status message
        setProcessingState((prev) => ({
          ...prev,
          status: 'processing',
          stageMessage: 'Sub-pixel co-registration & MAGSAC...',
        }));
        setPipelineStage(PipelineStage.MAGSAC);

        if (isBackendOnline) {
          try {
            await runPipelineOnPair({
              pair_id: res.pair_id,
              matcher: options.activeMatcher,
              options: {
                percentile_clipping: options.percentileClipping,
                clahe: options.clahe,
                morphological_gradients: options.morphologicalGradients,
                pca_band_reduction: options.pcaBandReduction,
              },
            });
          } catch (pipelineErr) {
            console.warn('Post-upload pipeline run warning:', pipelineErr);
          }
        }

        setPipelineStage(PipelineStage.Done);
        refreshData();

        // 3. Set completed state with newScene for complete processing transition
        setProcessingState({
          status: 'completed',
          pairName: res.name || displayName,
          fileCount: files.length,
          stageMessage: 'Co-registration complete. Ready for inspection.',
          newScene,
        });
      } else {
        throw new Error(res?.message || 'Failed to ingest mission files');
      }
    } catch (err: any) {
      console.error('Mission file ingestion failed:', err);
      setPipelineStage(PipelineStage.Idle);
      setProcessingState({
        status: 'error',
        pairName: displayName,
        fileCount: files.length,
        errorMessage: err?.message || 'Failed to process files. Ensure backend is running.',
      });
    }
  };

  const handleCompleteProcessing = () => {
    if (processingState?.newScene) {
      setSelectedScene(processingState.newScene);
    }
    setActiveCenterTab('2d');
    setProcessingState(null);
  };

  const handleDismissProcessing = () => {
    setProcessingState(null);
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black text-white font-sans select-none">
      {/* Spatial WebGL 3D Lunar Engine (Always mounted in background for both Landing & Workbench) */}
      <div className="absolute inset-0 w-full h-full z-0">
        <WebGlMoonViewer
          isWorkbenchMode={appMode === 'workbench'}
          selectedCrater={selectedScene}
          cameraZoom={cameraZoom}
          craters={craters}
          isDrawerOpen={isTargetPaletteOpen || isEngineInspectorOpen}
          onSelectCrater={(c) => setSelectedScene(c as any)}
          onInspectIn2D={() => setActiveCenterTab('2d')}
          layers={layers}
        />
      </div>

      {/* ── 1. LANDING PAGE OVERVIEW ── */}
      {appMode === 'landing' ? (
        <div className="relative z-10 w-full h-full overflow-y-auto">
          <LunarisLanding
            onLaunchWorkbench={handleLaunchFromLanding}
          />
        </div>
      ) : (
        /* ── 2. 3D MISSION CONTROL WORKBENCH ── */
        <div className="relative z-10 w-full h-full overflow-hidden pointer-events-none">
          {/* Center 2D Registration & Findings View (Conditional overlay) */}
          {activeCenterTab === '2d' && (
            <div className="absolute inset-0 w-full h-full z-10 pt-20 pb-20 px-4 md:px-12 bg-black/85 backdrop-blur-2xl overflow-hidden pointer-events-auto">
              <KeypointViewer
                key={selectedScene.id}
                pairId={selectedScene.id}
                rmsePx={activeTelemetry.rmsePx}
                srcSensor={pairSensorInfo?.srcSensor}
                refSensor={pairSensorInfo?.refSensor}
                srcGsd={pairSensorInfo?.srcGsd}
                refGsd={pairSensorInfo?.refGsd}
              />
            </div>
          )}

          {activeCenterTab === 'results' && (
            <div className="absolute inset-0 w-full h-full z-10 pt-20 pb-20 px-4 md:px-12 bg-black/85 backdrop-blur-2xl overflow-y-auto sidebar-scroll pointer-events-auto">
              <ResultsView
                telemetry={activeTelemetry}
                slz={currentSlz}
                spectralData={currentSpectral}
                selectedScene={selectedScene}
                isBackendOnline={isBackendOnline}
                isLoading={isLoadingScience}
              />
            </div>
          )}

          {/* Floating Aerospace Island HUD */}
          <MoonWorkbenchOverlay
            selectedScene={selectedScene}
            onSelectScene={(scene) => setSelectedScene(scene)}
            layers={layers}
            onLayerChange={(l) => setLayers(l)}
            options={options}
            onOptionsChange={(opts) => setOptions(opts)}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onResetView={handleResetView}
            activeTab={activeCenterTab}
            onTabChange={(t) => setActiveCenterTab(t)}
            onBackToLanding={handleBackToLanding}
            onOpenTargetPalette={() => setIsTargetPaletteOpen(true)}
            onOpenEngineInspector={() => setIsEngineInspectorOpen(true)}
            onOpenAddFiles={() => setIsAddFilesOpen(true)}
            pipelineStage={pipelineStage}
            onRunPipeline={handleRunPipeline}
            isBackendOnline={isBackendOnline}
            processingState={processingState}
            onCompleteProcessing={handleCompleteProcessing}
            onDismissProcessing={handleDismissProcessing}
          />

          {/* Modern Aerospace Modal 1: Lunar Target Command Palette */}
          {isTargetPaletteOpen && (
            <div className="pointer-events-auto">
              <LunarTargetPalette
                selectedScene={selectedScene}
                onSelectScene={(scene) => {
                  setSelectedScene(scene);
                  setIsTargetPaletteOpen(false);
                }}
                onClose={() => setIsTargetPaletteOpen(false)}
                craters={craters}
              />
            </div>
          )}

          {/* Modern Aerospace Modal 2: Registration Engine & Data Inspector */}
          <div className="pointer-events-auto">
            <EngineInspector
              isOpen={isEngineInspectorOpen}
              onClose={() => setIsEngineInspectorOpen(false)}
              options={options}
              onOptionsChange={setOptions}
              matcherParams={matcherParams}
              onUpdateMatcherParams={setMatcherParams}
              onSaveMatcherConfig={saveMatcherConfig}
              pairs={backendPairs}
              stats={datasetStats}
              selectedPairId={selectedScene.id}
              onSelectPair={(pairId) => {
                // Switch scene matching pair
                const foundCrater = craters.find((c) => c.id === pairId);
                if (foundCrater) {
                  setSelectedScene({
                    id: foundCrater.id,
                    name: foundCrater.name,
                    lat: foundCrater.lat,
                    lon: foundCrater.lon,
                    height: foundCrater.height,
                    description: foundCrater.description,
                  });
                } else {
                  const foundPair = backendPairs.find((p) => p.pair_id === pairId);
                  if (foundPair) {
                    setSelectedScene({
                      id: foundPair.pair_id,
                      name: foundPair.pair_id.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
                      lat: foundPair.latitude_center_deg ?? -72.0,
                      lon: foundPair.longitude_center_deg ?? 40.0,
                      height: 80000,
                      terrainClass: (foundPair.terrain_class as any) ?? 'highland',
                      solarIncidenceDeg: foundPair.src.solar_incidence_deg ?? 65.0,
                      solarAzimuthDeg: foundPair.src.solar_azimuth_deg ?? 180.0,
                      gsdM: foundPair.src.gsd_m,
                      description: `Multi-Sensor Pair: ${foundPair.src.sensor} (${foundPair.src.gsd_m}m) ➔ ${foundPair.ref.type}`,
                    });
                  }
                }
              }}
            />
          </div>

          {/* Modern Aerospace Modal 3: Add Mission Imagery Ingestion */}
          <div className="pointer-events-auto">
            <AddFilesModal
              isOpen={isAddFilesOpen}
              onClose={() => setIsAddFilesOpen(false)}
              onStartUpload={handleStartUpload}
              onPairCreated={(newScene) => {
                setSelectedScene(newScene);
                setActiveCenterTab('2d');
                refreshData();
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
