import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as Cesium from 'cesium';
import {
  Eye, EyeOff, Compass, Layers, ZoomIn, ZoomOut, RotateCcw,
  Mountain, Droplets, Orbit, Check, X, Maximize2, ChevronDown,
  Crosshair, Scan, Radio
} from 'lucide-react';
import type { ScenePreset, LayerVisibility, CraterDetail } from '../types';
import { getCraterCatalog } from '../services/api';
export type StarDimmerMode = 'cinematic' | 'deep' | 'subtle' | 'off';

Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN || '';

interface CesiumViewerProps {
  selectedScene: ScenePreset;
  layers: LayerVisibility;
  onLayerChange: (layers: LayerVisibility) => void;
  onSelectScene?: (scene: ScenePreset) => void;
  hideControls?: boolean;
  hudVisible?: boolean;
  craters?: CraterDetail[];
}

interface LayerConfig {
  key: keyof LayerVisibility;
  label: string;
  color: string;
}

const LAYER_CONFIG: LayerConfig[] = [
  { key: 'ohrc', label: 'OHRC 0.25m Warp', color: 'text-[#D4C59A]' },
  { key: 'tmc2Slope', label: 'TMC-2 3D Slope', color: 'text-[#C2B080]' },
  { key: 'iirsHyperspectral', label: 'IIRS Hyperspectral', color: 'text-[#FBBF24]' },
  { key: 'slzOverlay', label: 'SLZ Safe Zone', color: 'text-[#4ADE80]' },
];

const MOON_ELLIPSOID = Cesium.Ellipsoid.MOON;

// In-memory cache for circular feathered texture data URLs to avoid re-generating
const featheredTextureCache = new Map<string, string>();

/**
 * Creates a seamless circular feathered texture from a source image:
 * - Crops off any rectangular margins or bottom scale/watermark text
 * - Tone-maps shadows & highlights to match the 8K Cesium Moon regolith ambient illumination
 * - Applies a continuous cubic Hermite smoothstep alpha falloff (100% center -> 0% invisible at rim)
 * - Returns a transparent PNG data URL that blends naturally into the Moon globe texture
 */
function getCircularFeatheredTexture(
  imageUrl: string,
  isSpectral: boolean = false
): Promise<string> {
  const cacheKey = `${imageUrl}_${isSpectral ? 'spectral' : 'albedo'}`;
  const cached = featheredTextureCache.get(cacheKey);
  if (cached) return Promise.resolve(cached);

  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const size = 1024;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(imageUrl);
        return;
      }

      // Crop out bottom 7% to strip any source watermark/text bar
      const cropHeight = img.height * 0.93;
      ctx.drawImage(img, 0, 0, img.width, cropHeight, 0, 0, size, size);

      // Pixel-level processing: Tone-mapping + smoothstep alpha blending
      const imgData = ctx.getImageData(0, 0, size, size);
      const data = imgData.data;

      const centerX = size / 2;
      const centerY = size / 2;
      const maxRadius = size * 0.485;
      const innerRadius = isSpectral ? size * 0.24 : size * 0.30;

      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = (y * size + x) * 4;

          const dx = x - centerX;
          const dy = y - centerY;
          const dist = Math.sqrt(dx * dx + dy * dy);

          let alpha = 0.0;
          if (dist < innerRadius) {
            alpha = 1.0;
          } else if (dist < maxRadius) {
            // Cubic Hermite smoothstep fade to invisible
            const t = (dist - innerRadius) / (maxRadius - innerRadius);
            const smoothT = t * t * (3 - 2 * t);
            alpha = Math.max(0, 1.0 - smoothT);
          } else {
            alpha = 0.0;
          }

          if (alpha <= 0.002) {
            data[idx + 3] = 0;
            continue;
          }

          if (!isSpectral) {
            // Tone-mapping to match lunar regolith ambient illumination:
            // Lift deep shadows from 0 to 42 so shadows aren't pitch-black holes
            // Compress bright peaks so highlights blend seamlessly with Moon surface
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            const toneMapped = Math.min(220, Math.max(40, Math.round(38 + lum * 0.72)));

            data[idx] = toneMapped;
            data[idx + 1] = toneMapped;
            data[idx + 2] = toneMapped;
            data[idx + 3] = Math.round(alpha * 245);
          } else {
            // Spectral heatmap: preserve false-color thermal bands with translucent alpha
            data[idx + 3] = Math.round(alpha * 195);
          }
        }
      }

      ctx.putImageData(imgData, 0, 0);

      const dataUrl = canvas.toDataURL('image/png');
      featheredTextureCache.set(cacheKey, dataUrl);
      resolve(dataUrl);
    };
    img.onerror = () => resolve(imageUrl);
    img.src = imageUrl;
  });
}

// Helper to compute physically accurate Rectangles scaled to each crater's true landing / imaging swath
function computeAccurateCraterRectangle(
  lat: number,
  lon: number,
  diameterKm: number,
  scaleMultiplier: number = 1.55
): Cesium.Rectangle {
  const KM_PER_DEG_LAT = 30.3233; // Lunar circumference / 360 = 30.3233 km / deg
  
  // Real metric ground swath in kilometers:
  // The crater rim in the 1024x1024 photo occupies ~64.5% of the frame.
  // Multiplying diameter by 1.55 scales the crater rim to match the 3D Moon 1:1 in size!
  const trueSwathKm = diameterKm * scaleMultiplier;
  const radiusKm = trueSwathKm / 2;

  // Latitude span (north/south) in degrees
  const deltaLat = radiusKm / KM_PER_DEG_LAT;

  // Longitude span (east/west) adjusted for lunar latitude convergence
  // Clamped at 85° to prevent polar meridian singularity
  const latRad = Cesium.Math.toRadians(Math.min(Math.abs(lat), 85.0));
  const cosLat = Math.max(Math.cos(latRad), 0.087);
  const deltaLon = Math.min(deltaLat / cosLat, 15.0);

  const south = Math.max(-89.99, lat - deltaLat);
  const north = Math.min(89.99, lat + deltaLat);
  const west = Math.max(-180, lon - deltaLon);
  const east = Math.min(180, lon + deltaLon);

  return Cesium.Rectangle.fromDegrees(west, south, east, north);
}

export const CesiumViewer: React.FC<CesiumViewerProps> = ({
  selectedScene,
  layers,
  onLayerChange,
  onSelectScene,
  hideControls = false,
  hudVisible = true,
  craters: initialCraters = [],
}) => {
  const [allCraters, setAllCraters] = useState<CraterDetail[]>(initialCraters);

  useEffect(() => {
    if (initialCraters && initialCraters.length > 0) {
      setAllCraters(initialCraters);
      return;
    }
    getCraterCatalog().then((res) => {
      if (res && res.length > 0) {
        setAllCraters(
          res.map((c) => ({
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
          }))
        );
      }
    });
  }, [initialCraters]);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const ohrcLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const iirsLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const tmc2LayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const pinEntityRef = useRef<Cesium.Entity | null>(null);
  const craterEntitiesRef = useRef<Cesium.Entity[]>([]);
  // Ground-view refs — DOM updates only, zero React re-renders in the 60fps loop
  const groundListenerRef = useRef<(() => void) | null>(null);
  const groundHeadingRef = useRef(0);

  const [isFlying, setIsFlying] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  // Start with NO crater pre-selected — clean global Moon model at launch
  const [selectedCrater, setSelectedCrater] = useState<CraterDetail | null>(null);
  const [showInspector, setShowInspector] = useState(false);
  const [isGroundMode, setIsGroundMode] = useState(false);
  const [showLayerMenu, setShowLayerMenu] = useState(false);
  const [showTargetDropdown, setShowTargetDropdown] = useState(false);
  const groundDragRef = useRef<{ active: boolean; x: number; y: number } | null>(null);

  // ── Mission Control Target Acquisition & Survey Sequence ──
  const [targetStatus, setTargetStatus] = useState<{
    crater: CraterDetail;
    stage: 'locking' | 'approaching' | 'surveying' | 'locked';
    mode: 'global' | 'survey' | 'recon';
  } | null>(null);
  const surveyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flightTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Inspection Altitude Modes: 'global' (4200km full Moon) | 'survey' (160km) | 'recon' (24km 3D close-up)
  const [reconMode, setReconMode] = useState<'global' | 'survey' | 'recon'>('global');

  // Update dynamic OHRC / IIRS draping layers when selected crater changes
  const updateCraterDrapeLayers = useCallback((crater: CraterDetail) => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    // Remove previous local layers
    if (ohrcLayerRef.current) {
      viewer.imageryLayers.remove(ohrcLayerRef.current, true);
      ohrcLayerRef.current = null;
    }
    if (iirsLayerRef.current) {
      viewer.imageryLayers.remove(iirsLayerRef.current, true);
      iirsLayerRef.current = null;
    }

    // Compute physically accurate bounds matching the crater's real diameter in km
    // Scale multiplier 1.55 matches the ~64.5% crater rim inside the 1024x1024 image 1:1 with Moon surface
    const ohrcRect = computeAccurateCraterRectangle(crater.lat, crater.lon, crater.diameterKm, 1.55);
    const iirsRect = computeAccurateCraterRectangle(crater.lat, crater.lon, crater.diameterKm, 1.85);

    // 0.3m OHRC Ultra-Res Local Crater Drape — seamless tone-mapped circular feathered texture
    getCircularFeatheredTexture('/assets/ohrc.jpg', false).then((featheredUrl) => {
      if (viewer.isDestroyed()) return;
      Cesium.SingleTileImageryProvider.fromUrl(featheredUrl, {
        rectangle: ohrcRect,
        ellipsoid: MOON_ELLIPSOID,
      }).then((provider) => {
        if (!viewer.isDestroyed()) {
          const layer = viewer.imageryLayers.addImageryProvider(provider);
          layer.alpha = 0.85;
          layer.brightness = 1.0;
          layer.contrast = 1.02;
          layer.show = layers.ohrc;
          ohrcLayerRef.current = layer;
        }
      });
    });

    // IIRS Hyperspectral Thermal Drape — seamless translucent heatmap texture
    getCircularFeatheredTexture('/assets/iirs.jpg', true).then((featheredUrl) => {
      if (viewer.isDestroyed()) return;
      Cesium.SingleTileImageryProvider.fromUrl(featheredUrl, {
        rectangle: iirsRect,
        ellipsoid: MOON_ELLIPSOID,
      }).then((provider) => {
        if (!viewer.isDestroyed()) {
          const layer = viewer.imageryLayers.addImageryProvider(provider);
          layer.alpha = 0.52;
          layer.brightness = 1.05;
          layer.show = layers.iirsHyperspectral;
          iirsLayerRef.current = layer;
        }
      });
    });
  }, [layers.ohrc, layers.iirsHyperspectral]);

  // Stable ref so the click handler inside useEffect always calls the latest version
  const rotateToCraterRef = useRef<(crater: CraterDetail, mode?: 'global' | 'survey' | 'recon') => void>(() => {});

  // 3-phase cinematic animation: zoom out → rotate to crater → dive in
  // Multi-stage mission control targeting & survey sequence with realistic delays
  const rotateToCrater = useCallback((crater: CraterDetail, mode: 'global' | 'survey' | 'recon' = 'recon') => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    // ── Clear any ongoing transition timers ──
    if (surveyTimeoutRef.current) clearTimeout(surveyTimeoutRef.current);
    if (flightTimeoutRef.current) clearTimeout(flightTimeoutRef.current);

    // ── 1. Stop any ground-mode rotation & reset camera transform ──
    if (groundListenerRef.current) {
      groundListenerRef.current();
      groundListenerRef.current = null;
    }
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
    setIsGroundMode(false);
    viewer.camera.cancelFlight();

    // ── 2. Immediate Target Selection Acquisition State (Deliberate lock-on delay) ──
    setSelectedCrater(crater);
    setReconMode(mode);
    setIsFlying(true);
    setTargetStatus({ crater, stage: 'locking', mode });

    const craterRadiusMeters = (crater.diameterKm * 1000) / 2;
    let targetAltitude: number;
    if (mode === 'recon') {
      targetAltitude = Math.max(28000, craterRadiusMeters * 2.2);
    } else if (mode === 'survey') {
      targetAltitude = Math.max(95000, craterRadiusMeters * 4.3);
    } else {
      targetAltitude = 4600000;
    }

    // ── Step 1 Delay: Target Vector Acquisition & Lock-on (1.2s Deliberate Lock) ──
    surveyTimeoutRef.current = setTimeout(() => {
      const v = viewerRef.current;
      if (!v || v.isDestroyed()) return;

      // Update HUD to orbital approach phase
      setTargetStatus((prev) => (prev ? { ...prev, stage: 'approaching' } : null));

      // ── Step 2: Smooth, Slow Cinematic Approach Flight (3.2s Majestic Descent) ──
      v.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(
          crater.lon,
          crater.lat,
          targetAltitude,
          MOON_ELLIPSOID
        ),
        orientation: {
          heading: 0,
          pitch: -Cesium.Math.PI_OVER_TWO,
          roll: 0,
        },
        duration: 3.2,
        easingFunction: Cesium.EasingFunction.QUADRATIC_IN_OUT,
        complete: () => {
          if (v.isDestroyed()) return;
          v.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

          // ── Step 3 Delay: Thorough Orbital Survey Scanning & Multi-spectral Draping (1.6s Scan) ──
          setTargetStatus((prev) => (prev ? { ...prev, stage: 'surveying' } : null));
          updateCraterDrapeLayers(crater);

          flightTimeoutRef.current = setTimeout(() => {
            // ── Step 4: Survey Complete & Telemetry Locked ──
            setTargetStatus((prev) => (prev ? { ...prev, stage: 'locked' } : null));
            setIsFlying(false);
            setShowInspector(true);

            // Keep confirmation badge active for 3.8s then fade out cleanly
            setTimeout(() => {
              setTargetStatus((prev) => (prev?.stage === 'locked' ? null : prev));
            }, 3800);
          }, 1600);
        },
        cancel: () => {
          setIsFlying(false);
          setTargetStatus(null);
          if (!v.isDestroyed()) {
            v.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
          }
        },
      });
    }, 1200);
  }, [updateCraterDrapeLayers]);

  // Keep the ref in sync with the latest callback so the useEffect click handler is never stale
  useEffect(() => {
    rotateToCraterRef.current = rotateToCrater;
  }, [rotateToCrater]);

  // Initialize Cesium on Mount
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    try {
      const viewer = new Cesium.Viewer(containerRef.current, {
        timeline: false,
        animation: false,
        baseLayerPicker: false,
        navigationHelpButton: false,
        homeButton: false,
        geocoder: false,
        sceneModePicker: false,
        projectionPicker: false,
        infoBox: false,
        selectionIndicator: false,
        fullscreenButton: false,
        vrButton: false,
        requestRenderMode: false,
        globe: new Cesium.Globe(MOON_ELLIPSOID),
        terrainProvider: new Cesium.EllipsoidTerrainProvider({ ellipsoid: MOON_ELLIPSOID }),
        contextOptions: {
          webgl: {
            alpha: false,
            depth: true,
            stencil: false,
            antialias: true,
            powerPreference: 'high-performance',
          },
        },
      });

      // ── HIGH-FPS BUTTERY-SMOOTH RENDERING CONFIGURATION ──
      // Cap resolution scale at 1.25 for crisp high-DPI while maintaining locked 60 FPS
      viewer.resolutionScale = Math.min(window.devicePixelRatio || 1.0, 1.25);
      viewer.scene.globe.maximumScreenSpaceError = 2.0; // Standard optimal LOD (eliminates tile re-render micro-stutter)
      viewer.scene.globe.tileCacheSize = 1000;
      viewer.scene.globe.loadingDescendantLimit = 32;
      viewer.scene.globe.depthTestAgainstTerrain = false;
      viewer.scene.msaaSamples = 1; // 1x MSAA (eliminates GPU fill-rate lag)

      // ── RESPONSIVE, SNAPPY 3D CAMERA CONTROLS (Zero Sluggish Drag Lag) ──
      const controller = viewer.scene.screenSpaceCameraController;
      controller.enableZoom = false; // We use our universal extra-sensitive handler below so zoom works anywhere on screen
      controller.enableRotate = true;
      controller.enableTilt = true;
      controller.enableTranslate = false; // Keep Moon sphere centered during drag
      controller.enableLook = false;
      controller.enableCollisionDetection = false; // Never freeze zoom near surface

      controller.rotateEventTypes = Cesium.CameraEventType.LEFT_DRAG;
      controller.tiltEventTypes = [
        Cesium.CameraEventType.MIDDLE_DRAG,
        Cesium.CameraEventType.PINCH,
        { eventType: Cesium.CameraEventType.LEFT_DRAG, modifier: Cesium.KeyboardEventModifier.CTRL },
      ];
      controller.inertiaSpin = 0.28; // Snappy, precise rotational stopping

      // ── EXTRA-SENSITIVE UNIVERSAL MOUSE WHEEL ZOOM ──
      // Works everywhere across the canvas (hovering on Moon craters OR in deep space)
      const canvas = viewer.scene.canvas;
      const onCanvasWheel = (e: WheelEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (viewer.isDestroyed()) return;

        const camera = viewer.camera;
        viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

        const distanceToCenter = Cesium.Cartesian3.magnitude(camera.positionWC);
        const altitudeAboveMoon = Math.max(60, distanceToCenter - 1_737_400);

        // Gentle, slower, unhurried zoom sensitivity (8.5% step with delta scaling)
        const delta = Math.sign(e.deltaY);
        const intensity = Math.min(1.0, Math.max(0.4, Math.abs(e.deltaY) * 0.008));
        const zoomStep = Math.max(160, altitudeAboveMoon * 0.085 * intensity);

        if (delta < 0) {
          // Wheel Up: Zoom IN
          if (altitudeAboveMoon > 80) {
            camera.zoomIn(zoomStep);
          }
        } else {
          // Wheel Down: Zoom OUT
          if (altitudeAboveMoon < 25_000_000) {
            camera.zoomOut(zoomStep);
          }
        }
      };

      canvas.addEventListener('wheel', onCanvasWheel, { passive: false });

      // Clear default imagery
      viewer.imageryLayers.removeAll();

      // Configure Moon Globe — warm neutral grey, zero blue atmosphere haze
      viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#020604');
      viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#8A8580'); // Warm regolith grey — NO blue tint
      viewer.scene.globe.enableLighting = false;
      viewer.scene.globe.showGroundAtmosphere = false;
      viewer.scene.fog.enabled = false; // Kill Cesium's distance fog that adds blue haze at close range
      viewer.scene.globe.atmosphereLightIntensity = 0; // Zero atmospheric scattering
      viewer.scene.globe.atmosphereRayleighScaleHeight = 0;
      viewer.scene.globe.atmosphereMieScaleHeight = 0;

      if (viewer.scene.skyAtmosphere) {
        viewer.scene.skyAtmosphere.show = false;
      }
      if (viewer.scene.skyBox) {
        viewer.scene.skyBox.show = true; // Real 3D celestial stars & Milky Way
      }
      if (viewer.scene.sun) {
        viewer.scene.sun.show = false; // No sun lens flare
      }
      if (viewer.scene.moon) {
        viewer.scene.moon.show = false; // Hide Cesium's default moon (we ARE the moon)
      }

      // High-contrast directional illumination for crisp micro-crater relief
      viewer.scene.light = new Cesium.DirectionalLight({
        direction: new Cesium.Cartesian3(0.5, 0.5, -0.7),
        color: Cesium.Color.fromCssColorString('#FFFFFF'),
        intensity: 2.2,
      });

      // ── 1. PRIMARY: Full-sphere Moon Global Basemap (NASA 8K LROC Master Dataset) ──
      Cesium.SingleTileImageryProvider.fromUrl('/assets/moon_global.jpg?v=nasa_8k_master', {
        rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
        ellipsoid: MOON_ELLIPSOID,
      }).then((provider) => {
        if (!viewer.isDestroyed()) {
          const globalLayer = viewer.imageryLayers.addImageryProvider(provider);
          globalLayer.alpha = 1.0;
        }
      });

      // ── 2. TMC-2 3D Slope Elevation Grid Layer ──
      const tmc2SlopeProvider = new Cesium.GridImageryProvider({
        color: Cesium.Color.fromCssColorString('rgba(52, 211, 153, 0.35)'),
        glowColor: Cesium.Color.fromCssColorString('rgba(16, 185, 129, 0.15)'),
        cells: 16,
      });
      const tmc2Layer = viewer.imageryLayers.addImageryProvider(tmc2SlopeProvider);
      tmc2Layer.show = layers.tmc2Slope;
      tmc2LayerRef.current = tmc2Layer;

      // ── 3. Add Interactive 3D Markers & Landing Targets for All Lunar Craters ──
      const createdEntities: Cesium.Entity[] = [];
      allCraters.forEach((crater) => {
        const isSelected = crater.id === selectedScene.id;
        const craterEntity = viewer.entities.add({
          id: `crater_${crater.id}`,
          position: Cesium.Cartesian3.fromDegrees(crater.lon, crater.lat, 1000, MOON_ELLIPSOID),
          point: {
            pixelSize: isSelected ? 13 : 8,
            color: Cesium.Color.fromCssColorString(crater.waterAbsorptionDepthPct > 10 ? '#38BDF8' : '#FDBA74'),
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          ellipse: {
            semiMajorAxis: (crater.diameterKm * 1000) / 2,
            semiMinorAxis: (crater.diameterKm * 1000) / 2,
            material: Cesium.Color.fromCssColorString(
              crater.waterAbsorptionDepthPct > 10 ? 'rgba(56, 189, 248, 0.18)' : 'rgba(52, 211, 153, 0.15)'
            ),
            outline: true,
            outlineColor: Cesium.Color.fromCssColorString(crater.waterAbsorptionDepthPct > 10 ? '#38BDF8' : '#FDBA74'),
            outlineWidth: 1.5,
          },
          label: {
            text: `${crater.name} (${crater.waterAbsorptionDepthPct}% H2O)`,
            font: 'bold 11px "JetBrains Mono", monospace',
            fillColor: Cesium.Color.fromCssColorString('#FFFFFF'),
            outlineColor: Cesium.Color.fromCssColorString('#022C22'),
            outlineWidth: 4,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            pixelOffset: new Cesium.Cartesian2(0, -14),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        });
        createdEntities.push(craterEntity);
      });
      craterEntitiesRef.current = createdEntities;

      // ── 4. Active Mission Target Pin ──
      const pin = viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(
          selectedScene.lon,
          selectedScene.lat,
          800,
          MOON_ELLIPSOID
        ),
        point: {
          pixelSize: 12,
          color: Cesium.Color.fromCssColorString('#FDBA74'),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: `SLZ TARGET: ${selectedScene.name}`,
          font: 'bold 12px "JetBrains Mono", monospace',
          fillColor: Cesium.Color.fromCssColorString('#FDBA74'),
          outlineColor: Cesium.Color.fromCssColorString('#064E3B'),
          outlineWidth: 4,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -20),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      pinEntityRef.current = pin;

      // ── 5. Interactive Click Handler on Moon Surface & Craters ──
      const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
      handler.setInputAction((click: { position: Cesium.Cartesian2 }) => {
        const pickedObject = viewer.scene.pick(click.position);
        if (Cesium.defined(pickedObject) && pickedObject.id) {
          const entityId = String(pickedObject.id.id || '');
          if (entityId.startsWith('crater_')) {
            const craterId = entityId.replace('crater_', '');
            const foundCrater = allCraters.find((c) => c.id === craterId);
            if (foundCrater) {
              // Use ref so we always call the latest rotateToCrater, never a stale closure
              rotateToCraterRef.current(foundCrater, 'recon');
              return;
            }
          }
        }

        // If clicked on Moon surface / globe ellipsoid
        const ray = viewer.camera.getPickRay(click.position);
        if (ray) {
          const cartesian = viewer.scene.globe.pick(ray, viewer.scene) || viewer.camera.pickEllipsoid(click.position, MOON_ELLIPSOID);
          if (cartesian) {
            const cartographic = Cesium.Cartographic.fromCartesian(cartesian, MOON_ELLIPSOID);
            const latDeg = Cesium.Math.toDegrees(cartographic.latitude);
            const lonDeg = Cesium.Math.toDegrees(cartographic.longitude);
            
            // Only select if user clicked within tight proximity (<= 3.5°) of an actual verified crater marker
            let closestCrater: CraterDetail | null = null;
            let minDist = 3.5;
            for (const c of allCraters) {
              const dist = Math.hypot(c.lat - latDeg, c.lon - lonDeg);
              if (dist < minDist) {
                minDist = dist;
                closestCrater = c;
              }
            }
            if (closestCrater) {
              rotateToCraterRef.current(closestCrater, 'recon');
            }
            // If clicked on generic empty terrain, DO NOT relocate or paste the crater image!
          }
        }
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

      // ── Canonical Near Side Alignment (Seamless 1:1 match with landing page 3D Moon) ──
      viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(0, 0, 4_600_000, MOON_ELLIPSOID),
        orientation: {
          heading: 0,
          pitch: -Cesium.Math.PI_OVER_TWO,
          roll: 0,
        },
      });

      // Force canvas buffer to resize to true DOM dimensions and render the initial frame immediately
      requestAnimationFrame(() => {
        if (viewer && !viewer.isDestroyed()) {
          viewer.resize();
          viewer.scene.render();
        }
      });

      viewerRef.current = viewer;
      setIsInitialized(true);
    } catch (err) {
      console.warn('Cesium initialization notice:', err);
    }

    return () => {
      if (surveyTimeoutRef.current) clearTimeout(surveyTimeoutRef.current);
      if (flightTimeoutRef.current) clearTimeout(flightTimeoutRef.current);
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only ONCE — Cesium must never be re-initialized on scene change

  // Update pin and layers when selectedScene changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed() || !isInitialized) return;

    if (pinEntityRef.current) {
      pinEntityRef.current.position = new Cesium.ConstantPositionProperty(
        Cesium.Cartesian3.fromDegrees(
          selectedScene.lon,
          selectedScene.lat,
          800,
          MOON_ELLIPSOID
        )
      );
      if (pinEntityRef.current.label) {
        pinEntityRef.current.label.text = new Cesium.ConstantProperty(
          `SLZ TARGET: ${selectedScene.name}`
        );
      }
    }
  }, [selectedScene, isInitialized]);

  // Ensure SkyBox real 3D stars are always visible
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed() || !isInitialized) return;
    if (viewer.scene.skyBox) {
      viewer.scene.skyBox.show = true;
    }
  }, [isInitialized]);

  // Update layer visibility
  useEffect(() => {
    if (ohrcLayerRef.current) ohrcLayerRef.current.show = layers.ohrc;
    if (iirsLayerRef.current) iirsLayerRef.current.show = layers.iirsHyperspectral;
    if (tmc2LayerRef.current) tmc2LayerRef.current.show = layers.tmc2Slope;
  }, [layers]);

  // ── Stop ground-level rotation ──
  const exitGroundMode = useCallback(() => {
    const viewer = viewerRef.current;
    if (groundListenerRef.current) {
      groundListenerRef.current();
      groundListenerRef.current = null;
    }
    // Release the camera.lookAt lock
    if (viewer && !viewer.isDestroyed()) {
      viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
    }
    setIsGroundMode(false);
    setIsFlying(false);
    if (selectedCrater) {
      rotateToCrater(selectedCrater, 'recon');
    }
  }, [selectedCrater, rotateToCrater]);

  // GROUND-LEVEL SURFACE VIEW — Free-Look Horizon View
  // Places camera at crater surface altitude and gives user 100% manual free-look control
  const handleGroundLevelView = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed() || !selectedCrater) return;

    // Toggle off if already in ground mode
    if (isGroundMode) {
      exitGroundMode();
      return;
    }

    // Cancel any existing rotation listener & release transform lock
    if (groundListenerRef.current) {
      groundListenerRef.current();
      groundListenerRef.current = null;
    }
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

    setIsFlying(true);
    setIsGroundMode(false);
    setShowInspector(false);
    updateCraterDrapeLayers(selectedCrater);

    const crater = selectedCrater;
    groundHeadingRef.current = 0;

    // The crater center on the surface
    const craterPos = Cesium.Cartesian3.fromDegrees(crater.lon, crater.lat, 0, MOON_ELLIPSOID);
    const craterRadiusMeters = (crater.diameterKm * 1000) / 2;
    const sphere = new Cesium.BoundingSphere(craterPos, Math.max(5000, craterRadiusMeters));

    const isPolar = Math.abs(crater.lat) > 75;
    const orbitRange = Math.max(12000, Math.min(craterRadiusMeters * 1.1, 90000));
    const orbitPitch = isPolar ? -38 : -28; // Oblique horizon angle looking across the landscape

    // ── Fly down to crater surface level ──
    viewer.camera.flyToBoundingSphere(sphere, {
      offset: new Cesium.HeadingPitchRange(
        Cesium.Math.toRadians(0),
        Cesium.Math.toRadians(orbitPitch),
        orbitRange
      ),
      duration: 2.2,
      easingFunction: Cesium.EasingFunction.SINUSOIDAL_IN_OUT,
      complete: () => {
        setIsFlying(false);
        setIsGroundMode(true);
        // Release lookAt transform so mouse drag / touch allows user to look around freely!
        viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
      },
      cancel: () => {
        setIsFlying(false);
      },
    });
  }, [selectedCrater, updateCraterDrapeLayers, isGroundMode, exitGroundMode]);

  // Interactive Pan / Pitch adjustments in Surface View
  const handleSurfacePan = (deltaDeg: number) => {
    const viewer = viewerRef.current;
    if (!viewer || !selectedCrater) return;
    const craterPos = Cesium.Cartesian3.fromDegrees(selectedCrater.lon, selectedCrater.lat, 0, MOON_ELLIPSOID);
    const hpr = viewer.camera.heading;
    const pitch = viewer.camera.pitch;
    const range = Cesium.Cartesian3.distance(viewer.camera.position, craterPos);
    viewer.camera.lookAt(
      craterPos,
      new Cesium.HeadingPitchRange(
        hpr + Cesium.Math.toRadians(deltaDeg),
        pitch,
        range
      )
    );
  };

  const handleSurfacePitch = (pitchDeg: number) => {
    const viewer = viewerRef.current;
    if (!viewer || !selectedCrater) return;
    const craterPos = Cesium.Cartesian3.fromDegrees(selectedCrater.lon, selectedCrater.lat, 0, MOON_ELLIPSOID);
    const hpr = viewer.camera.heading;
    const range = Cesium.Cartesian3.distance(viewer.camera.position, craterPos);
    viewer.camera.lookAt(
      craterPos,
      new Cesium.HeadingPitchRange(
        hpr,
        Cesium.Math.toRadians(pitchDeg),
        range
      )
    );
  };

  const handleGroundPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isGroundMode) return;
    groundDragRef.current = { active: true, x: event.clientX, y: event.clientY };
  };

  const handleGroundPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const viewer = viewerRef.current;
    if (!viewer || !isGroundMode || !groundDragRef.current || !selectedCrater) return;

    const deltaX = event.clientX - groundDragRef.current.x;
    const deltaY = event.clientY - groundDragRef.current.y;

    groundDragRef.current = { active: true, x: event.clientX, y: event.clientY };

    const rotationX = Cesium.Math.toRadians(deltaX * 0.22);
    const rotationY = Cesium.Math.toRadians(deltaY * 0.18);

    viewer.camera.rotateRight(rotationX);
    viewer.camera.rotateUp(rotationY);
  };

  const handleGroundPointerUp = () => {
    groundDragRef.current = null;
  };

  const handleZoom = (inOut: 'in' | 'out') => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;
    const camera = viewer.camera;
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

    const distanceToCenter = Cesium.Cartesian3.magnitude(camera.positionWC);
    const altitudeAboveMoon = Math.max(80, distanceToCenter - 1_737_400);

    // Gentle, slower 14% step per button click (smooth & steady)
    const zoomStep = Math.max(350, altitudeAboveMoon * 0.14);

    if (inOut === 'in') {
      if (altitudeAboveMoon > 100) {
        camera.zoomIn(zoomStep);
      }
    } else {
      if (altitudeAboveMoon < 25_000_000) {
        camera.zoomOut(zoomStep);
      }
    }
  };

  const handleResetView = () => {
    setSelectedCrater(null);
    setShowInspector(false);
    setReconMode('global');
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;
    if (ohrcLayerRef.current) {
      viewer.imageryLayers.remove(ohrcLayerRef.current, true);
      ohrcLayerRef.current = null;
    }
    if (iirsLayerRef.current) {
      viewer.imageryLayers.remove(iirsLayerRef.current, true);
      iirsLayerRef.current = null;
    }
    viewer.camera.cancelFlight();
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        0,
        0,
        4_600_000,
        MOON_ELLIPSOID
      ),
      orientation: {
        heading: 0,
        pitch: -Cesium.Math.PI_OVER_TWO,
        roll: 0,
      },
      duration: 2.2,
      easingFunction: Cesium.EasingFunction.QUADRATIC_OUT,
      complete: () => {
        if (viewer && !viewer.isDestroyed()) {
          viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
        }
      },
    });
  };

  const toggleLayer = (key: keyof LayerVisibility) => {
    onLayerChange({ ...layers, [key]: !layers[key] });
  };

  const handleApplyCraterAsScene = () => {
    if (!selectedCrater || !onSelectScene) return;
    const matchedPreset: ScenePreset = {
      id: selectedCrater.id,
      name: selectedCrater.name,
      lat: selectedCrater.lat,
      lon: selectedCrater.lon,
      height: selectedCrater.height || 80000,
      terrainClass: selectedCrater.lat < -60 ? 'polar_highland' : 'highland',
      craterDensity: 4.5,
      solarIncidenceDeg: selectedCrater.solarIncidenceDeg,
      solarAzimuthDeg: selectedCrater.solarAzimuthDeg,
      gsdM: 0.31,
      overlayOpacity: 0.75,
      description: selectedCrater.description,
    };
    onSelectScene(matchedPreset);
  };

  return (
    <div
      className="relative w-full h-full flex flex-col bg-transparent overflow-hidden"
      onPointerDown={handleGroundPointerDown}
      onPointerMove={handleGroundPointerMove}
      onPointerUp={handleGroundPointerUp}
      onPointerLeave={handleGroundPointerUp}
    >
      {/* ── UNIFIED TOP FLOATING CONTROL BAR (Zero Overlap & Multi-Recon Modes) ── */}
      {!hideControls && (
        <div className={`absolute top-2.5 left-2.5 right-2.5 z-20 flex items-center justify-between gap-2 pointer-events-auto transition-all duration-700 ease-out transform ${
          hudVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'
        }`}>
        {/* Left: Terrain Draping Layer Toggle & Star Dimmer */}
        <div className="flex items-center gap-1.5">
          <div className="relative">
            <button
              onClick={() => setShowLayerMenu(!showLayerMenu)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-black/40 hover:bg-black/60 backdrop-blur-xl border border-[#D4C59A]/30 text-[10px] font-mono font-bold text-white shadow-xl transition-all"
            >
              <Layers size={13} className="text-[#D4C59A]" />
              <span className="hidden sm:inline">Terrain Layers</span>
            </button>

            {showLayerMenu && (
              <div className="absolute top-full mt-1.5 left-0 z-40 bg-[#0D0E12]/95 backdrop-blur-2xl rounded-2xl border border-[#D4C59A]/40 p-2 flex flex-col gap-1 shadow-2xl min-w-[195px]">
                <div className="flex items-center gap-1.5 px-1 mb-0.5 text-[9px] font-mono text-[#D4C59A] font-extrabold uppercase tracking-wider">
                  <Layers size={11} className="text-[#D4C59A]" />
                  <span>Active Overlays</span>
                </div>
                {LAYER_CONFIG.map((cfg) => (
                  <button
                    key={cfg.key}
                    onClick={() => toggleLayer(cfg.key)}
                    className={`flex items-center gap-2 px-2.5 py-1.5 rounded-xl text-left transition-all text-[9.5px] font-mono font-bold ${
                      layers[cfg.key]
                        ? 'bg-[#222018] border border-[#D4C59A] text-white shadow-md'
                        : 'bg-[#090A0E] text-slate-400 hover:text-white hover:bg-[#141620]'
                    }`}
                  >
                    {layers[cfg.key] ? (
                      <Eye size={12} className={cfg.color} />
                    ) : (
                      <EyeOff size={12} className="text-slate-600" />
                    )}
                    <span className={layers[cfg.key] ? 'text-white' : 'text-slate-400'}>
                      {cfg.label}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center: Target Lunar Sites Transparent Dropdown Button */}
        <div className="relative">
          <button
            onClick={() => {
              setShowTargetDropdown(!showTargetDropdown);
              setShowLayerMenu(false);
            }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/40 hover:bg-black/60 backdrop-blur-2xl border border-[#D4C59A]/30 hover:border-[#D4C59A]/60 text-[10px] font-mono text-white shadow-xl transition-all group"
          >
            <span className="text-[8.5px] font-bold text-[#D4C59A] uppercase tracking-widest">TARGETS:</span>
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold text-white group-hover:text-[#D4C59A] transition-colors">
                {selectedCrater ? selectedCrater.name : 'Select Target Crater'}
              </span>
            </div>
            {selectedCrater && (
              <span className="text-[7.5px] font-mono px-1.5 py-0.5 rounded bg-[#1C1A14]/90 text-[#D4C59A] border border-[#D4C59A]/30">
                {selectedCrater.waterAbsorptionDepthPct}% H₂O
              </span>
            )}
            <ChevronDown
              size={12}
              className={`text-[#D4C59A] transition-transform duration-200 ${showTargetDropdown ? 'rotate-180' : ''}`}
            />
          </button>

          {showTargetDropdown && (
            <div className="absolute top-full mt-1.5 left-0 z-40 bg-[#07080A]/95 backdrop-blur-2xl rounded-2xl border border-[#D4C59A]/40 p-2 flex flex-col gap-1 shadow-2xl min-w-[260px] max-h-80 overflow-y-auto sidebar-scroll">
              <div className="flex items-center justify-between px-1.5 pb-1 mb-1 border-b border-[#D4C59A]/20">
                <span className="text-[9px] font-mono text-[#D4C59A] font-extrabold uppercase tracking-wider">
                  Lunar Landing Targets ({allCraters.length})
                </span>
                <span className="text-[8px] font-mono text-slate-400">3D Recon Sites</span>
              </div>

              {allCraters.map((crater) => {
                const isTarget = selectedCrater?.id === crater.id;
                return (
                  <button
                    key={crater.id}
                    onClick={() => {
                      rotateToCrater(crater, 'recon');
                      setShowTargetDropdown(false);
                    }}
                    className={`flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-xl text-left transition-all font-mono text-[9.5px] ${
                      isTarget
                        ? 'bg-[#D4C59A] text-black font-extrabold shadow-md'
                        : 'bg-black/40 hover:bg-[#141620] text-slate-200 hover:text-white border border-transparent hover:border-[#D4C59A]/20'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${isTarget ? 'bg-black' : 'bg-[#D4C59A]'}`} />
                      <div className="truncate">
                        <span className="block truncate font-bold">{crater.name}</span>
                        <span className={`text-[7.5px] block ${isTarget ? 'text-black/70' : 'text-slate-400'}`}>
                          [{Math.abs(crater.lat).toFixed(1)}°S, {Math.abs(crater.lon).toFixed(1)}°E] · ⌀ {crater.diameterKm} km
                        </span>
                      </div>
                    </div>
                    <span className={`text-[7.5px] font-mono px-1.5 py-0.5 rounded shrink-0 font-bold ${
                      isTarget ? 'bg-black/20 text-black' : 'bg-[#1C1A14] text-[#D4C59A] border border-[#D4C59A]/30'
                    }`}>
                      {crater.waterAbsorptionDepthPct}% H₂O
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Recon Altitude Switcher & Zoom Controls */}
        <div className="flex items-center gap-1.5 min-w-max">
          {/* Recon Modes: 160km Survey / Global */}
          <div className="bg-[#0D0E12]/95 backdrop-blur-xl rounded-xl border border-[#D4C59A]/25 p-0.5 flex items-center gap-0.5 shadow-lg">
            <button
              onClick={() => rotateToCrater(selectedCrater || allCraters[0], 'survey')}
              title="160km Orbital Survey"
              className={`px-2 py-1 rounded-lg text-[9px] font-mono font-extrabold transition-all ${
                reconMode === 'survey'
                  ? 'bg-[#D4C59A]/25 text-[#D4C59A] border border-[#D4C59A]/50'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Survey
            </button>
            <button
              onClick={handleResetView}
              title="4,500km Full Moon Global Sphere — Fit Whole Moon on Screen"
              className={`px-2 py-1 rounded-lg text-[9px] font-mono font-extrabold transition-all ${
                reconMode === 'global'
                  ? 'bg-[#D4C59A]/25 text-[#D4C59A] border border-[#D4C59A]/50'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Fit Whole Moon
            </button>
          </div>
        </div>
      </div>
      )}

      {/* ── MISSION CONTROL TARGET ACQUISITION & SURVEY HUD BANNER ── */}
      {targetStatus && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 z-30 pointer-events-none animate-in fade-in zoom-in-95 duration-300">
          <div className="flex items-center gap-3 px-3.5 py-2 rounded-2xl bg-[#080A0F]/95 backdrop-blur-2xl border border-[#D4C59A]/40 shadow-[0_0_30px_rgba(212,197,154,0.25)] min-w-[310px] max-w-md">
            {/* Animated Reticle Icon */}
            <div className="relative flex items-center justify-center w-8 h-8 rounded-xl bg-[#141824] border border-[#D4C59A]/30 text-[#D4C59A] shrink-0">
              {targetStatus.stage === 'locking' && (
                <Crosshair size={18} className="animate-spin text-amber-300" style={{ animationDuration: '3s' }} />
              )}
              {targetStatus.stage === 'approaching' && (
                <Radio size={18} className="animate-pulse text-sky-300" />
              )}
              {targetStatus.stage === 'surveying' && (
                <Scan size={18} className="animate-bounce text-emerald-300" />
              )}
              {targetStatus.stage === 'locked' && (
                <Check size={18} className="text-[#D4C59A]" />
              )}
            </div>

            {/* Target Telemetry Details */}
            <div className="flex-1 flex flex-col min-w-0">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] font-mono font-extrabold text-white tracking-wider truncate">
                  TARGET: {targetStatus.crater.name.toUpperCase()}
                </span>
                <span className={`text-[8px] font-mono font-extrabold px-1.5 py-0.5 rounded border uppercase shrink-0 ${
                  targetStatus.stage === 'locking'
                    ? 'bg-amber-950/80 text-amber-300 border-amber-500/40 animate-pulse'
                    : targetStatus.stage === 'approaching'
                    ? 'bg-sky-950/80 text-sky-300 border-sky-500/40'
                    : targetStatus.stage === 'surveying'
                    ? 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40 animate-pulse'
                    : 'bg-[#1C1A14] text-[#D4C59A] border-[#D4C59A]/40'
                }`}>
                  {targetStatus.stage === 'locking' && 'ACQUIRING...'}
                  {targetStatus.stage === 'approaching' && (targetStatus.mode === 'survey' ? 'SURVEY PASS' : 'DESCENT')}
                  {targetStatus.stage === 'surveying' && 'SURVEY SCAN'}
                  {targetStatus.stage === 'locked' && 'SURVEY COMPLETE'}
                </span>
              </div>

              <span className="text-[8.5px] font-mono text-slate-300 mt-0.5 truncate">
                {targetStatus.stage === 'locking' && 'Locking coordinates & computing trajectory...'}
                {targetStatus.stage === 'approaching' && `Gliding to ${targetStatus.mode === 'survey' ? '160km orbital survey' : 'reconnaissance altitude'}...`}
                {targetStatus.stage === 'surveying' && 'Capturing multi-spectral & 5m DEM telemetry...'}
                {targetStatus.stage === 'locked' && `${targetStatus.crater.lat.toFixed(2)}°S, ${targetStatus.crater.lon.toFixed(2)}°E · ${targetStatus.crater.waterAbsorptionDepthPct}% H₂O signature`}
              </span>

              {/* Progress Bar */}
              <div className="w-full h-1 bg-white/10 rounded-full mt-1.5 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-amber-400 via-sky-400 to-[#D4C59A] transition-all duration-500"
                  style={{
                    width:
                      targetStatus.stage === 'locking'
                        ? '25%'
                        : targetStatus.stage === 'approaching'
                        ? '65%'
                        : targetStatus.stage === 'surveying'
                        ? '90%'
                        : '100%',
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}



      {/* ── GROUND-MODE HORIZON HUD CONTROLS ── */}
      {isGroundMode && (
        <div
          className="absolute inset-0 z-25 pointer-events-none"
          style={{ animation: 'fadeInOverlay 0.5s ease forwards' }}
        >
          {/* Subtle cinematic vignette */}
          <div className="absolute inset-0" style={{
            background: 'radial-gradient(ellipse 85% 75% at 50% 50%, transparent 50%, rgba(0,0,0,0.75) 100%)'
          }} />

          {/* Top-left: Mode badge + crater info */}
          <div className="absolute top-14 left-3 flex flex-col gap-1.5 pointer-events-auto">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-xl border border-[#D4C59A]/50 shadow-[0_0_20px_rgba(212,197,154,0.3)]">
              <span className="w-2 h-2 rounded-full bg-[#4ADE80] animate-pulse" />
              <span className="text-[10px] font-mono font-extrabold text-[#D4C59A] uppercase tracking-widest">3D SURFACE HORIZON VIEW</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-black/75 backdrop-blur-xl border border-[#D4C59A]/25">
              <span className="text-[9px] font-mono text-[#D4C59A] font-bold block">{selectedCrater?.name}</span>
              <span className="text-[8px] font-mono text-slate-300">Diameter: {selectedCrater?.diameterKm} km · Depth: {selectedCrater?.depthKm} km</span>
            </div>
          </div>

          {/* Top-right: STOP button */}
          <div className="absolute top-14 right-3 pointer-events-auto">
            <button
              onClick={exitGroundMode}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-600/90 hover:bg-rose-500 active:bg-rose-700 border border-rose-300/60 text-white font-mono font-extrabold text-[11px] shadow-2xl backdrop-blur-xl transition-all shadow-[0_0_20px_rgba(239,68,68,0.5)]"
            >
              <X size={13} />
              <span>EXIT SURFACE VIEW</span>
            </button>
          </div>

          {/* Bottom-center: Interactive Horizon Pan & Pitch Controls Bar */}
          <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 pointer-events-auto">
            <div className="flex items-center gap-2 px-4 py-2 rounded-2xl bg-black/90 backdrop-blur-2xl border border-[#D4C59A]/40 shadow-[0_0_28px_rgba(212,197,154,0.3)]">
              {/* 360° Pan Buttons */}
              <div className="flex items-center gap-1 border-r border-white/20 pr-2.5">
                <span className="text-[9px] font-mono text-[#D4C59A] font-bold uppercase">PAN:</span>
                <button
                  onClick={() => handleSurfacePan(-30)}
                  title="Pan 30° Left"
                  className="px-2 py-1 rounded-lg bg-[#141620] hover:bg-[#1C202C] border border-[#D4C59A]/30 text-[#EBE2CD] text-[10px] font-mono font-extrabold transition-all"
                >
                  ⟲ Left
                </button>
                <button
                  onClick={() => handleSurfacePan(30)}
                  title="Pan 30° Right"
                  className="px-2 py-1 rounded-lg bg-[#141620] hover:bg-[#1C202C] border border-[#D4C59A]/30 text-[#EBE2CD] text-[10px] font-mono font-extrabold transition-all"
                >
                  ⟳ Right
                </button>
              </div>

              {/* Pitch Angle Preset Toggles */}
              <div className="flex items-center gap-1 border-r border-white/20 pr-2.5">
                <span className="text-[9px] font-mono text-[#D4C59A] font-bold uppercase">ANGLE:</span>
                <button
                  onClick={() => handleSurfacePitch(-20)}
                  title="Look towards the distant horizon (-20°)"
                  className="px-2 py-1 rounded-lg bg-[#141620] hover:bg-[#1C202C] border border-[#D4C59A]/30 text-[#EBE2CD] text-[10px] font-mono font-bold transition-all"
                >
                  Horizon
                </button>
                <button
                  onClick={() => handleSurfacePitch(-45)}
                  title="Oblique crater slope angle (-45°)"
                  className="px-2 py-1 rounded-lg bg-[#141620] hover:bg-[#1C202C] border border-[#D4C59A]/30 text-[#EBE2CD] text-[10px] font-mono font-bold transition-all"
                >
                  Oblique
                </button>
                <button
                  onClick={() => handleSurfacePitch(-85)}
                  title="Look straight down at crater floor (-85°)"
                  className="px-2 py-1 rounded-lg bg-[#141620] hover:bg-[#1C202C] border border-[#D4C59A]/30 text-[#EBE2CD] text-[10px] font-mono font-bold transition-all"
                >
                  Overhead
                </button>
              </div>

              {/* Mouse Drag Hint */}
              <div className="text-[9px] font-mono text-white/80 hidden sm:block">
                <span className="text-[#D4C59A] font-bold">Left Drag</span> to Orbit · <span className="text-[#D4C59A] font-bold">Scroll</span> to Zoom
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Interactive Crater Inclination & Water-Ice Inspector (Glass Popup) ── */}
      {selectedCrater && showInspector && !hideControls && (
        <div className="absolute bottom-10 right-4 z-30 w-84 max-w-[92vw] bg-gradient-to-b from-[#0D0E12]/98 via-[#0A0B0F]/98 to-[#07080A]/99 backdrop-blur-2xl rounded-2xl border border-[#D4C59A]/40 p-3 shadow-[0_16px_50px_rgba(0,0,0,0.95)] inspector-enter">
          {/* Inspector Header */}
          <div className="flex items-start justify-between pb-2 border-b border-[#D4C59A]/25">
            <div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#4ADE80] animate-pulse" />
                <h3 className="text-xs font-bold text-white font-mono leading-tight">
                  {selectedCrater.name}
                </h3>
              </div>
              <p className="text-[9px] font-mono text-[#D4C59A] mt-0.5">
                {selectedCrater.region} · [{selectedCrater.lat.toFixed(2)}°S, {selectedCrater.lon.toFixed(2)}°E]
              </p>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="text-[8px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#1C1A14] text-[#D4C59A] border border-[#D4C59A]/30">
                  ⌀ {selectedCrater.diameterKm} km
                </span>
                <span className="text-[8px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#141620] text-[#EBE2CD] border border-[#D4C59A]/30">
                  Depth: {selectedCrater.depthKm} km
                </span>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={() => rotateToCrater(selectedCrater)}
                title="Rotate & Center Camera"
                className="p-1 rounded-lg bg-[#141620] hover:bg-[#1C202C] text-[#D4C59A] border border-[#D4C59A]/30"
              >
                <Maximize2 size={11} />
              </button>
              <button
                onClick={() => setShowInspector(false)}
                className="p-1 rounded-lg bg-[#141620] hover:bg-rose-950 text-slate-400 hover:text-rose-300 border border-[#D4C59A]/30 transition-colors"
              >
                <X size={11} />
              </button>
            </div>
          </div>

          {/* ── SECTION 1: INCLINATION & SLOPES ── */}
          <div className="mt-2 space-y-1.5">
            <div className="flex items-center gap-1 text-[9px] font-mono font-extrabold text-[#D4C59A] uppercase tracking-wider">
              <Orbit size={11} className="text-[#D4C59A]" />
              <span>Inclination & Topographic Slope</span>
            </div>

            <div className="grid grid-cols-3 gap-1 text-center font-mono">
              <div className="bg-[#090A0E]/90 p-1.5 rounded-xl border border-[#D4C59A]/15">
                <span className="text-[7.5px] text-slate-400 block">FLOOR INCL.</span>
                <span className="text-[11px] font-extrabold text-white">{selectedCrater.floorInclinationDeg}°</span>
              </div>
              <div className="bg-[#090A0E]/90 p-1.5 rounded-xl border border-[#D4C59A]/15">
                <span className="text-[7.5px] text-slate-400 block">WALL SLOPE</span>
                <span className="text-[11px] font-extrabold text-[#FBBF24]">{selectedCrater.wallSlopeDeg}°</span>
              </div>
              <div className="bg-[#090A0E]/90 p-1.5 rounded-xl border border-[#D4C59A]/15">
                <span className="text-[7.5px] text-slate-400 block">ORBIT INCL.</span>
                <span className="text-[11px] font-extrabold text-[#EBE2CD]">{selectedCrater.orbitInclinationDeg}°</span>
              </div>
            </div>

            <div className="flex items-center justify-between text-[8px] font-mono px-1 py-0.5 bg-[#090A0E]/80 rounded-lg border border-[#D4C59A]/15 text-slate-200">
              <span>Solar Sun Incidence: <strong className="text-[#D4C59A]">{selectedCrater.solarIncidenceDeg}°</strong></span>
              <span>Azimuth: <strong className="text-[#EBE2CD]">{selectedCrater.solarAzimuthDeg}°</strong></span>
            </div>
          </div>

          {/* ── SECTION 2: 3.0 µm WATER-ICE & HYDRATION TELEMETRY ── */}
          <div className="mt-2.5 pt-2 border-t border-[#D4C59A]/25 space-y-1.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1 text-[9px] font-mono font-extrabold text-[#D4C59A] uppercase tracking-wider">
                <Droplets size={11} className="text-[#D4C59A]" />
                <span>3.0 µm Water-Ice Hydration</span>
              </div>
              <span className={`text-[8px] font-mono font-extrabold px-1.5 py-0.2 rounded border ${
                selectedCrater.psrStatus.includes('PSR')
                  ? 'bg-[#1C1A14] text-[#D4C59A] border-[#D4C59A]/60 shadow-[0_0_8px_rgba(212,197,154,0.3)]'
                  : 'bg-[#141620] text-[#EBE2CD] border-[#D4C59A]/30'
              }`}>
                {selectedCrater.psrStatus}
              </span>
            </div>

            {/* Absorption Depth Gauge */}
            <div className="bg-[#090A0E]/90 p-2 rounded-xl border border-[#D4C59A]/20">
              <div className="flex items-center justify-between text-[9px] font-mono mb-1">
                <span className="text-slate-200">3.0µm OH/H₂O Absorption Depth</span>
                <span className="text-[#D4C59A] font-extrabold">{selectedCrater.waterAbsorptionDepthPct}%</span>
              </div>
              <div className="w-full h-1.5 bg-black/80 rounded-full overflow-hidden border border-[#D4C59A]/20">
                <div
                  className="h-full rounded-full transition-all duration-700 bg-gradient-to-r from-[#D4C59A] via-[#EBE2CD] to-[#4ADE80]"
                  style={{ width: `${Math.min(100, selectedCrater.waterAbsorptionDepthPct * 3.5)}%` }}
                />
              </div>

              <div className="grid grid-cols-2 gap-1.5 mt-2 font-mono text-[9px]">
                <div className="bg-[#07080A] p-1.5 rounded-lg border border-[#D4C59A]/15">
                  <span className="text-[7.5px] text-slate-400 block">EST. WATER CONCENTRATION</span>
                  <span className="text-xs font-extrabold text-[#D4C59A]">{selectedCrater.waterIceConcentrationWtPct} wt%</span>
                  <span className="text-[8px] text-slate-400 block">({selectedCrater.waterIcePpm.toLocaleString()} ppm)</span>
                </div>
                <div className="bg-[#07080A] p-1.5 rounded-lg border border-[#D4C59A]/15">
                  <span className="text-[7.5px] text-slate-400 block">SURFACE TEMP / FROST</span>
                  <span className="text-xs font-extrabold text-amber-200">{selectedCrater.surfaceTempKelvin} K</span>
                  <span className="text-[8px] text-[#4ADE80] block font-bold">Frost Index: {selectedCrater.frostIndex}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Inspector Footer Actions */}
          <div className="mt-2.5 pt-2 border-t border-[#D4C59A]/25 flex items-center gap-1.5">
            <button
              onClick={handleApplyCraterAsScene}
              className="flex-1 flex items-center justify-center gap-1 px-2.5 py-1.5 rounded-xl bg-[#D4C59A] hover:bg-[#EBE2CD] text-black font-mono font-extrabold text-[10px] shadow-lg transition-all"
            >
              <Check size={12} />
              <span>Set as Active Scene</span>
            </button>
            <button
              onClick={handleGroundLevelView}
              disabled={isFlying || isGroundMode}
              className="flex items-center justify-center gap-1 px-2.5 py-1.5 rounded-xl bg-[#141620] hover:bg-[#1C202C] border border-[#D4C59A]/40 text-[#D4C59A] font-mono font-extrabold text-[10px] transition-all shadow-lg"
            >
              <Maximize2 size={12} />
              <span>Surface View</span>
            </button>
          </div>
        </div>
      )}

      {/* Floating Bottom-Right Controls: Crater Findings Pill + Zoom & Reset Navigation */}
      {!hideControls && (
        <div className={`absolute bottom-10 right-3 z-30 flex items-center gap-2 pointer-events-auto transition-all duration-700 ease-out transform ${
          hudVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'
        }`}>
        {!showInspector && (
          <button
            onClick={() => setShowInspector(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-black/40 hover:bg-black/60 backdrop-blur-xl border border-[#D4C59A]/30 text-[10px] font-mono text-white shadow-xl transition-all group"
          >
            <Droplets size={12} className="text-[#D4C59A] group-hover:scale-110 transition-transform" />
            <span>{selectedCrater ? `${selectedCrater.name} Findings` : 'Crater Findings'}</span>
            <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-[#1C1A14] text-[#D4C59A] border border-[#D4C59A]/30">
              {selectedCrater ? `${selectedCrater.waterAbsorptionDepthPct}% H₂O` : 'Inspect'}
            </span>
          </button>
        )}

        {/* Zoom & Reset Controls on Down Right */}
        <div className="flex items-center gap-0.5 bg-[#0D0E12]/95 backdrop-blur-2xl p-0.5 rounded-xl border border-[#D4C59A]/25 shadow-2xl">
          <button
            onClick={() => handleZoom('in')}
            title="Zoom In"
            className="p-1.5 rounded-lg hover:bg-[#181B24] text-slate-300 hover:text-[#D4C59A] transition-colors"
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={() => handleZoom('out')}
            title="Zoom Out"
            className="p-1.5 rounded-lg hover:bg-[#181B24] text-slate-300 hover:text-[#D4C59A] transition-colors"
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={handleResetView}
            title="Reset Lunar View"
            className="p-1.5 rounded-lg hover:bg-[#181B24] text-slate-300 hover:text-[#D4C59A] transition-colors"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>
      )}

      {/* Cesium Container (100% Full Canvas Height) */}
      <div
        ref={containerRef}
        className="w-full h-full rounded-xl overflow-hidden bg-transparent pointer-events-auto cursor-grab active:cursor-grabbing"
      />


      {/* Floating Viewport Footer Telemetry Bar */}
      {!hideControls && (
        <div className={`absolute bottom-2 left-2 right-2 z-20 flex items-center justify-between px-3 py-1.5 bg-[#0D0E12]/90 backdrop-blur-xl rounded-xl border border-[#D4C59A]/20 text-[9.5px] font-mono text-white shadow-xl pointer-events-auto transition-all duration-700 ease-out transform ${
          hudVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'
        }`}>
        <span className="flex items-center gap-1.5">
          <Compass size={12} className="text-[#D4C59A]" />
          <span className="font-bold">Target: {selectedScene.lat.toFixed(2)}°S, {selectedScene.lon.toFixed(2)}°E</span>
        </span>
        <span className="hidden sm:flex items-center gap-1.5">
          <Mountain size={12} className="text-[#EBE2CD]" />
          <span className="font-bold">TMC-2 DEM: 5m/px Vertical · GSD: {selectedScene.gsdM}m</span>
        </span>
        <span className="text-[#D4C59A] font-extrabold flex items-center gap-1">
          <Droplets size={11} className="text-[#D4C59A]" />
          <span>Click on Moon or select Target Crater to rotate & inspect</span>
        </span>
      </div>
      )}
    </div>
  );
};
