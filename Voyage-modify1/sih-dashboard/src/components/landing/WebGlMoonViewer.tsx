import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { ScenePreset, CraterDetail, LayerVisibility } from '../../types';
import { getCraterCatalog } from '../../services/api';
import ohrcThumb from '../../assets/images/ohrc_orbital_fallback.jpg';
import ohrcCrater from '../../assets/images/ohrc_lunar_crater_1788336805774.jpg';
import iirsThumb from '../../assets/images/iirs_hyperspectral_overlay_1788336834453.jpg';
import tmc2Thumb from '../../assets/images/tmc2_terrain_context_1788336820221.jpg';
import lroThumb from '../../assets/images/lro_reference_baseline_1788336850293.jpg';
import surfaceHero from '../../assets/images/lunar_surface_hero_1788336791925.jpg';

const FALLBACK_THUMBNAILS: Record<string, string> = {
  boguslawsky: ohrcThumb,
  copernicus: ohrcCrater,
  shackleton: tmc2Thumb,
  manzinus: lroThumb,
  cabeus: iirsThumb,
  clavius: surfaceHero,
  tycho: lroThumb,
};

function getCraterPreviewImage(craterId?: string, lat?: number): string {
  if (!craterId) return ohrcCrater;
  const id = craterId.toLowerCase();
  for (const key of Object.keys(FALLBACK_THUMBNAILS)) {
    if (id.includes(key)) return FALLBACK_THUMBNAILS[key];
  }
  if (lat !== undefined && Math.abs(lat) > 60) return ohrcThumb;
  return ohrcCrater;
}

function getCraterSensorBadge(craterId?: string): { sensor: string; res: string } {
  const id = (craterId || '').toLowerCase();
  if (id.includes('boguslawsky') || id.includes('copernicus')) {
    return { sensor: 'OHRC HIGH-RES ORBITAL', res: '0.25 m/px' };
  }
  if (id.includes('shackleton')) {
    return { sensor: 'TMC-2 STEREO 3D', res: '5.0 m/px' };
  }
  if (id.includes('cabeus')) {
    return { sensor: 'IIRS HYPERSPECTRAL', res: '80 m/px' };
  }
  if (id.includes('manzinus') || id.includes('tycho')) {
    return { sensor: 'LRO-NAC POLAR REF', res: '0.50 m/px' };
  }
  return { sensor: 'CH-2 PAYLOAD SUITE', res: 'SUB-METER' };
}


// Procedural high-resolution crinkled gold thermal foil texture
function createGoldFoilTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#b8860b';
    ctx.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 3000; i++) {
      const x = Math.random() * 256;
      const y = Math.random() * 256;
      const l = Math.random() * 14 + 3;
      const shade = Math.random() > 0.5 ? 'rgba(255, 235, 140, 0.4)' : 'rgba(70, 45, 0, 0.45)';
      ctx.strokeStyle = shade;
      ctx.lineWidth = Math.random() * 1.5 + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (Math.random() - 0.5) * l, y + (Math.random() - 0.5) * l);
      ctx.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

// Procedural photovoltaic solar array texture with silver busbars
function createSolarPanelTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#06122c';
    ctx.fillRect(0, 0, 512, 256);

    const cols = 8;
    const rows = 4;
    const cellW = (512 - 20) / cols;
    const cellH = (256 - 16) / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = 10 + c * cellW + 1.5;
        const y = 8 + r * cellH + 1.5;
        const w = cellW - 3;
        const h = cellH - 3;

        ctx.fillStyle = '#0a1d4a';
        ctx.fillRect(x, y, w, h);

        ctx.strokeStyle = 'rgba(100, 180, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w, h);

        ctx.strokeStyle = 'rgba(200, 230, 255, 0.7)';
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        ctx.moveTo(x + w * 0.33, y);
        ctx.lineTo(x + w * 0.33, y + h);
        ctx.moveTo(x + w * 0.66, y);
        ctx.lineTo(x + w * 0.66, y + h);
        ctx.stroke();
      }
    }

    ctx.strokeStyle = '#99aabf';
    ctx.lineWidth = 3;
    ctx.strokeRect(2, 2, 508, 252);
  }
  return new THREE.CanvasTexture(canvas);
}

export interface WebGlMoonViewerProps {
  onLaunchWorkbench?: () => void;
  isLaunching?: boolean;
  isWorkbenchMode?: boolean;
  selectedCrater?: ScenePreset | null;
  cameraZoom?: number;
  craters?: CraterDetail[];
  isDrawerOpen?: boolean;
  sunAzimuthDeg?: number;
  onSelectCrater?: (crater: CraterDetail | ScenePreset) => void;
  onInspectIn2D?: () => void;
  layers?: LayerVisibility;
}

export const WebGlMoonViewer: React.FC<WebGlMoonViewerProps> = ({
  isLaunching = false,
  isWorkbenchMode = false,
  selectedCrater = null,
  cameraZoom,
  craters = [],
  isDrawerOpen = false,
  sunAzimuthDeg = 68,
  layers,
  onSelectCrater,
  onInspectIn2D,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [hoveredCrater, setHoveredCrater] = useState<CraterDetail | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const hoveredCraterRef = useRef<CraterDetail | null>(null);
  const onInspectIn2DRef = useRef(onInspectIn2D);
  useEffect(() => {
    onInspectIn2DRef.current = onInspectIn2D;
  }, [onInspectIn2D]);

  // References for Three.js state
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const moonGroupRef = useRef<THREE.Group | null>(null);
  const satGroupRef = useRef<THREE.Group | null>(null);
  const starFieldRef = useRef<THREE.Points | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const craterPinsGroupRef = useRef<THREE.Group | null>(null);
  const keyLightRef = useRef<THREE.DirectionalLight | null>(null);
  const moonMeshRef = useRef<THREE.Mesh | null>(null);
  const targetBeaconGroupRef = useRef<THREE.Group | null>(null);
  const selectedCraterRef = useRef(selectedCrater);
  const hudRef = useRef<HTMLDivElement | null>(null);
  const hudNameRef = useRef<HTMLSpanElement | null>(null);
  const hudCoordRef = useRef<HTMLSpanElement | null>(null);

  const targetRotationY = useRef(0);
  const targetRotationX = useRef(0);
  const isDragging = useRef(false);
  const previousPointerPosition = useRef({ x: 0, y: 0 });
  const pointerDownPos = useRef({ x: 0, y: 0 });
  const isLaunchingRef = useRef(isLaunching);
  const isWorkbenchModeRef = useRef(isWorkbenchMode);
  const isDrawerOpenRef = useRef(isDrawerOpen);
  const sunAzimuthRef = useRef(sunAzimuthDeg);
  const onSelectCraterRef = useRef(onSelectCrater);
  const layersRef = useRef(layers);
  const gridGroupRef = useRef<THREE.Group | null>(null);
  const waterIceGroupRef = useRef<THREE.Group | null>(null);
  const moonMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const moonOffsetRef = useRef({ x: 1.48, y: 0 });
  const targetCameraZ = useRef(3.8);

  const getCraterPositionOnMesh = (lat: number, lon: number, r: number = 0.95 * 1.004) => {
    const theta = ((90 - lat) * Math.PI) / 180;
    const phi = (0.5 + lon / 360) * 2 * Math.PI;
    const px = -r * Math.cos(phi) * Math.sin(theta);
    const py = r * Math.cos(theta);
    const pz = r * Math.sin(phi) * Math.sin(theta);
    return new THREE.Vector3(px, py, pz);
  };

  const updateTargetCraterBeacon = (crater: ScenePreset | null) => {
    selectedCraterRef.current = crater;
    if (hudNameRef.current && crater) {
      hudNameRef.current.innerText = crater.name;
    }
    if (hudCoordRef.current && crater) {
      hudCoordRef.current.innerText = `${Math.abs(crater.lat).toFixed(1)}°${crater.lat < 0 ? 'S' : 'N'}, ${Math.abs(crater.lon).toFixed(1)}°${crater.lon < 0 ? 'W' : 'E'}`;
    }

    const beacon = targetBeaconGroupRef.current;
    if (!beacon) return;

    if (!crater || !isWorkbenchModeRef.current) {
      beacon.visible = false;
      if (hudRef.current) hudRef.current.style.display = 'none';
      return;
    }

    const pMesh = getCraterPositionOnMesh(crater.lat, crater.lon, 0.95 * 1.006);
    beacon.position.copy(pMesh);
    const normal = pMesh.clone().normalize();
    beacon.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
    beacon.visible = true;

    // Transform position from moonMesh to moonGroup space (moonMesh has rotation.y = -Math.PI / 2):
    const pGroup = pMesh.clone().applyEuler(new THREE.Euler(0, -Math.PI / 2, 0, 'XYZ'));

    // Compute Euler angles (order 'YXZ') to align crater directly with camera along +Z:
    const rotX = Math.atan2(pGroup.y, pGroup.z);
    const z_intermediate = pGroup.y * Math.sin(rotX) + pGroup.z * Math.cos(rotX);
    const rotY = Math.atan2(-pGroup.x, z_intermediate);

    targetRotationX.current = rotX;
    targetRotationY.current = rotY;
    targetCameraZ.current = 2.6;
  };

  useEffect(() => {
    onSelectCraterRef.current = onSelectCrater;
  }, [onSelectCrater]);

  useEffect(() => {
    isDrawerOpenRef.current = isDrawerOpen;
  }, [isDrawerOpen]);

  useEffect(() => {
    isLaunchingRef.current = isLaunching;
  }, [isLaunching]);

  useEffect(() => {
    isWorkbenchModeRef.current = isWorkbenchMode;
    if (isWorkbenchMode && selectedCraterRef.current) {
      updateTargetCraterBeacon(selectedCraterRef.current);
    } else if (!isWorkbenchMode) {
      if (targetBeaconGroupRef.current) targetBeaconGroupRef.current.visible = false;
      if (hudRef.current) hudRef.current.style.display = 'none';
    }
  }, [isWorkbenchMode]);

  useEffect(() => {
    layersRef.current = layers;
    if (gridGroupRef.current) {
      gridGroupRef.current.visible = Boolean(layers?.grid);
    }
    if (waterIceGroupRef.current) {
      waterIceGroupRef.current.visible = Boolean(layers?.waterIce);
    }
    if (craterPinsGroupRef.current) {
      craterPinsGroupRef.current.visible = Boolean(layers?.craters);
    }
    if (moonMaterialRef.current) {
      if (layers?.dem) {
        moonMaterialRef.current.color.setHex(0xa8caff);
        moonMaterialRef.current.roughness = 0.65;
      } else {
        moonMaterialRef.current.color.setHex(0xdcd8d2);
        moonMaterialRef.current.roughness = 0.95;
      }
    }
  }, [layers]);

  useEffect(() => {
    sunAzimuthRef.current = sunAzimuthDeg;
    if (keyLightRef.current) {
      const rad = (sunAzimuthDeg * Math.PI) / 180;
      keyLightRef.current.position.set(6 * Math.cos(rad), 2.8, 6 * Math.sin(rad));
    }
  }, [sunAzimuthDeg]);

  useEffect(() => {
    if (cameraZoom !== undefined) {
      targetCameraZ.current = cameraZoom;
    }
  }, [cameraZoom]);

  useEffect(() => {
    selectedCraterRef.current = selectedCrater;
    if (selectedCrater && isWorkbenchModeRef.current) {
      updateTargetCraterBeacon(selectedCrater);
    }
  }, [selectedCrater]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = mountRef.current;
    if (!canvas || !container) return;

    let animId: number;
    let width = container.clientWidth;
    let height = container.clientHeight;

    // 1. Full-Screen Scene & Perspective Camera
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    let aspect = width / height;
    const camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 100);
    camera.position.set(0, 0, 5.0);
    cameraRef.current = camera;

    // Calculate Moon X position based on aspect ratio
    const updateMoonPosition = () => {
      const asp = container.clientWidth / container.clientHeight;
      if (asp > 1.3) {
        moonOffsetRef.current.x = 1.50;
        camera.position.z = 4.8;
      } else if (asp > 0.85) {
        moonOffsetRef.current.x = 1.48;
        camera.position.z = 5.2;
      } else {
        moonOffsetRef.current.x = 0.65;
        camera.position.z = 5.8;
      }
      if (moonGroupRef.current) {
        moonGroupRef.current.position.set(moonOffsetRef.current.x, 0, 0);
      }
    };

    // 2. High-Performance Hardware-Accelerated Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
    });
    rendererRef.current = renderer;
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;

    // 3. Space Lighting calibrated to match Cesium's authentic NASA 8K lunar regolith illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.35);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfffef8, 1.1);
    const initialRad = (sunAzimuthRef.current * Math.PI) / 180;
    keyLight.position.set(6 * Math.cos(initialRad), 2.8, 6 * Math.sin(initialRad));
    scene.add(keyLight);
    keyLightRef.current = keyLight;

    const fillLight = new THREE.DirectionalLight(0xbfbcb5, 0.45);
    fillLight.position.set(-4, -2, 3);
    scene.add(fillLight);

    // 4. Hardware-Accelerated 3D WebGL Stars (optimized count)
    const starCount = 1000;
    const starPositions = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const i3 = i * 3;
      starPositions[i3] = (Math.random() - 0.5) * 70;
      starPositions[i3 + 1] = (Math.random() - 0.5) * 70;
      starPositions[i3 + 2] = (Math.random() - 0.5) * 70;

      const choice = Math.random();
      if (choice > 0.7) {
        starColors[i3] = 0.45; starColors[i3 + 1] = 0.75; starColors[i3 + 2] = 1.0; // Electric blue star
      } else if (choice > 0.45) {
        starColors[i3] = 1.0; starColors[i3 + 1] = 0.95; starColors[i3 + 2] = 0.85; // Warm gold star
      } else {
        starColors[i3] = 0.95; starColors[i3 + 1] = 0.95; starColors[i3 + 2] = 0.95; // Crisp white star
      }
    }

    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

    const starMat = new THREE.PointsMaterial({
      size: 0.08,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });

    const starField = new THREE.Points(starGeo, starMat);
    starFieldRef.current = starField;
    scene.add(starField);

    // 5. Main Moon Group - Optimized smooth geometry
    const moonGroup = new THREE.Group();
    moonGroup.rotation.order = 'YXZ';
    moonGroupRef.current = moonGroup;
    updateMoonPosition();
    scene.add(moonGroup);

    const radius = 0.95;
    const moonGeometry = new THREE.SphereGeometry(radius, 64, 64);

    const textureLoader = new THREE.TextureLoader();
    const moonTexture = textureLoader.load('/assets/moon_global_4k.jpg');
    moonTexture.colorSpace = THREE.SRGBColorSpace;
    moonTexture.generateMipmaps = true;
    moonTexture.minFilter = THREE.LinearMipmapLinearFilter;
    moonTexture.magFilter = THREE.LinearFilter;

    const moonMaterial = new THREE.MeshStandardMaterial({
      map: moonTexture,
      roughness: 0.95,
      metalness: 0.0,
      color: 0xdcd8d2,
    });

    const moonMesh = new THREE.Mesh(moonGeometry, moonMaterial);
    // Align with Cesium's canonical Near Side orientation (Mare Imbrium, Oceanus Procellarum, Tycho)
    moonMesh.rotation.y = -Math.PI / 2;
    moonGroup.add(moonMesh);
    moonMaterialRef.current = moonMaterial;
    moonMeshRef.current = moonMesh;

    // Selenographic Coordinate Grid (Lat/Lon parallels and meridians)
    const gridGroup = new THREE.Group();
    gridGroup.visible = Boolean(layersRef.current?.grid);
    gridGroupRef.current = gridGroup;
    moonMesh.add(gridGroup);

    const gridMat = new THREE.LineBasicMaterial({
      color: 0x4aa3ff,
      transparent: true,
      opacity: 0.35,
    });

    [-60, -30, 0, 30, 60].forEach((latDeg) => {
      const latR = (latDeg * Math.PI) / 180;
      const circleR = radius * 1.003 * Math.cos(latR);
      const circleY = radius * 1.003 * Math.sin(latR);
      const circleGeo = new THREE.BufferGeometry();
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * circleR, circleY, Math.sin(a) * circleR));
      }
      circleGeo.setFromPoints(pts);
      gridGroup.add(new THREE.Line(circleGeo, gridMat));
    });

    for (let lonDeg = 0; lonDeg < 180; lonDeg += 30) {
      const lonR = (lonDeg * Math.PI) / 180;
      const meridianGeo = new THREE.BufferGeometry();
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        pts.push(new THREE.Vector3(
          Math.sin(a) * radius * 1.003 * Math.cos(lonR),
          Math.cos(a) * radius * 1.003,
          Math.sin(a) * radius * 1.003 * Math.sin(lonR)
        ));
      }
      meridianGeo.setFromPoints(pts);
      gridGroup.add(new THREE.Line(meridianGeo, gridMat));
    }

    // Water Ice Cryogenic Traps Overlay (Polar Caps)
    const waterIceGroup = new THREE.Group();
    waterIceGroup.visible = Boolean(layersRef.current?.waterIce);
    waterIceGroupRef.current = waterIceGroup;
    moonMesh.add(waterIceGroup);

    const southCryoCap = new THREE.Mesh(
      new THREE.RingGeometry(radius * 0.05, radius * 0.38, 32),
      new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      })
    );
    southCryoCap.rotation.x = Math.PI / 2;
    southCryoCap.position.y = -radius * 0.985;
    waterIceGroup.add(southCryoCap);

    const northCryoCap = new THREE.Mesh(
      new THREE.RingGeometry(radius * 0.05, radius * 0.38, 32),
      new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      })
    );
    northCryoCap.rotation.x = Math.PI / 2;
    northCryoCap.position.y = radius * 0.985;
    waterIceGroup.add(northCryoCap);

    // 3D Glowing Crater Pins on Lunar Surface
    const craterPinsGroup = new THREE.Group();
    craterPinsGroupRef.current = craterPinsGroup;
    craterPinsGroup.visible = Boolean(layersRef.current?.craters);
    moonMesh.add(craterPinsGroup);

    // 3D Targeted Crater Beacon & Reticle System (for currently selected crater)
    const targetBeaconGroup = new THREE.Group();
    targetBeaconGroup.visible = false;
    moonMesh.add(targetBeaconGroup);
    targetBeaconGroupRef.current = targetBeaconGroup;

    // 1. Outer Pulsing Target Ring (Electric Cyan)
    const targetOuterRing = new THREE.Mesh(
      new THREE.RingGeometry(0.040, 0.056, 36),
      new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.95,
      })
    );
    targetOuterRing.rotation.x = Math.PI / 2;
    targetBeaconGroup.add(targetOuterRing);

    // 2. Inner Tactical Reticle Ring (Emerald Green)
    const targetInnerRing = new THREE.Mesh(
      new THREE.RingGeometry(0.018, 0.028, 36),
      new THREE.MeshBasicMaterial({
        color: 0x00ff88,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9,
      })
    );
    targetInnerRing.rotation.x = Math.PI / 2;
    targetBeaconGroup.add(targetInnerRing);

    // 3. Center Target Core Bead (Crisp Pure White)
    const targetCenterDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.015, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    targetBeaconGroup.add(targetCenterDot);

    // 4. Tactical Reticle Cardinal Crosshair Ticks
    const crosshairMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.9 });
    const tickGeoH = new THREE.BoxGeometry(0.14, 0.003, 0.003);
    const tickGeoV = new THREE.BoxGeometry(0.003, 0.003, 0.14);
    const tickH = new THREE.Mesh(tickGeoH, crosshairMat);
    const tickV = new THREE.Mesh(tickGeoV, crosshairMat);
    targetBeaconGroup.add(tickH);
    targetBeaconGroup.add(tickV);

    // 5. Vertical Holographic Laser Light Beam / Pillar
    const pillarGeo = new THREE.CylinderGeometry(0.003, 0.016, 0.32, 16);
    pillarGeo.translate(0, 0.16, 0); // extend outward from ground
    const pillarMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.7,
    });
    const laserPillar = new THREE.Mesh(pillarGeo, pillarMat);
    targetBeaconGroup.add(laserPillar);

    // 6. Glowing Beacon Tip at the top of the laser beam
    const tipGeo = new THREE.SphereGeometry(0.012, 12, 12);
    tipGeo.translate(0, 0.32, 0);
    const tipMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const laserTip = new THREE.Mesh(tipGeo, tipMat);
    targetBeaconGroup.add(laserTip);

    // Initialize selected crater beacon on mount
    if (selectedCraterRef.current && isWorkbenchModeRef.current) {
      updateTargetCraterBeacon(selectedCraterRef.current);
    }



    const renderCraterDots = (items: CraterDetail[]) => {
      while (craterPinsGroup.children.length > 0) {
        craterPinsGroup.remove(craterPinsGroup.children[0]);
      }
      items.forEach((crater) => {
        const p = getCraterPositionOnMesh(crater.lat, crater.lon, radius * 1.004);

        const dotGeo = new THREE.SphereGeometry(0.018, 12, 12);
        const dotMat = new THREE.MeshBasicMaterial({
          color: (crater.waterAbsorptionDepthPct ?? 0) > 10 ? 0x38bdf8 : 0xfdba74,
        });
        const dot = new THREE.Mesh(dotGeo, dotMat);
        dot.userData = { crater };
        dot.position.copy(p);
        craterPinsGroup.add(dot);
      });
    };

    if (craters && craters.length > 0) {
      renderCraterDots(craters);
    } else {
      getCraterCatalog().then((list) => {
        if (list && list.length > 0) {
          renderCraterDots(
            list.map((c) => ({
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
    }

    // 6. Genuine Physical 3D Chandrayaan-2 Model (75% scale)
    const satGroup = new THREE.Group();
    satGroup.scale.set(0.75, 0.75, 0.75);
    satGroupRef.current = satGroup;

    // Materials
    const goldFoilTex = createGoldFoilTexture();
    const solarTex = createSolarPanelTexture();

    const goldFoilMat = new THREE.MeshStandardMaterial({
      map: goldFoilTex,
      color: 0xffd700,
      metalness: 0.90,
      roughness: 0.28,
    });

    const busChassisMat = new THREE.MeshStandardMaterial({
      color: 0x222226,
      metalness: 0.85,
      roughness: 0.2,
    });

    const solarPanelMat = new THREE.MeshStandardMaterial({
      map: solarTex,
      metalness: 0.7,
      roughness: 0.22,
    });

    const dishMat = new THREE.MeshStandardMaterial({
      color: 0xf5f5f7,
      metalness: 0.4,
      roughness: 0.3,
    });

    // Central Avionics Bus Core
    const busMesh = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 0.24), goldFoilMat);
    satGroup.add(busMesh);

    // Structural equipment deck collars
    const topDeck = new THREE.Mesh(new THREE.BoxGeometry(0.185, 0.04, 0.185), busChassisMat);
    topDeck.position.y = 0.10;
    satGroup.add(topDeck);

    const bottomDeck = new THREE.Mesh(new THREE.BoxGeometry(0.185, 0.03, 0.185), busChassisMat);
    bottomDeck.position.y = -0.10;
    satGroup.add(bottomDeck);

    // Dual High-Resolution Solar Array Wings (textured silicon cells)
    const wingGeo = new THREE.BoxGeometry(0.48, 0.012, 0.20);

    const leftWing = new THREE.Mesh(wingGeo, solarPanelMat);
    leftWing.position.x = -0.34;
    satGroup.add(leftWing);

    const rightWing = new THREE.Mesh(wingGeo, solarPanelMat);
    rightWing.position.x = 0.34;
    satGroup.add(rightWing);

    // Gold solar panel hinge struts
    const hingeGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.10);
    const leftHinge = new THREE.Mesh(hingeGeo, goldFoilMat);
    leftHinge.rotation.z = Math.PI / 2;
    leftHinge.position.x = -0.12;
    satGroup.add(leftHinge);

    const rightHinge = new THREE.Mesh(hingeGeo, goldFoilMat);
    rightHinge.rotation.z = Math.PI / 2;
    rightHinge.position.x = 0.12;
    satGroup.add(rightHinge);

    // Optical High-Resolution Camera (OHRC) Payload Lens pointing down (nadir)
    const ohrcBarrel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.055, 0.12, 16),
      new THREE.MeshStandardMaterial({ color: 0x111115, metalness: 0.95, roughness: 0.1 })
    );
    ohrcBarrel.position.set(0, -0.14, 0);
    satGroup.add(ohrcBarrel);

    // Steerable High-Gain Parabolic Dish Antenna
    const dishGroup = new THREE.Group();
    const dishMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.12, 0.02, 0.035, 20),
      dishMat
    );
    dishGroup.add(dishMesh);

    // Feed horn sub-reflector tripod
    const feedHorn = new THREE.Mesh(
      new THREE.ConeGeometry(0.02, 0.04, 8),
      new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.9, roughness: 0.2 })
    );
    feedHorn.position.y = 0.03;
    dishGroup.add(feedHorn);

    dishGroup.position.set(0, 0.14, 0.08);
    dishGroup.rotation.x = Math.PI / 3.5;
    satGroup.add(dishGroup);

    // 4 Corner Reaction Control Thruster (RCS) pods
    const thrusterMat = new THREE.MeshStandardMaterial({ color: 0x888890, metalness: 0.8, roughness: 0.2 });
    const thrusterCorners = [
      [0.08, -0.08, 0.10],
      [-0.08, -0.08, 0.10],
      [0.08, -0.08, -0.10],
      [-0.08, -0.08, -0.10],
    ];
    thrusterCorners.forEach(([tx, ty, tz]) => {
      const cone = new THREE.Mesh(new THREE.ConeGeometry(0.015, 0.03, 8), thrusterMat);
      cone.position.set(tx, ty, tz);
      cone.rotation.x = Math.PI;
      satGroup.add(cone);
    });

    // Active Cyan Telemetry Beacon LED
    const beaconLight = new THREE.PointLight(0x00f0ff, 3.0, 2.0);
    beaconLight.position.set(0, 0.16, -0.12);
    satGroup.add(beaconLight);

    const beaconDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.03, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x00ffff })
    );
    beaconDot.position.copy(beaconLight.position);
    satGroup.add(beaconDot);

    scene.add(satGroup);

    // 7. Raycaster: Only rotate when dragging directly on or near the Moon!
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const handlePointerDown = (e: PointerEvent) => {
      if (isDrawerOpenRef.current) return;
      if ((e.target as HTMLElement)?.closest('[data-sidebar], .sidebar-scroll, [role="dialog"], button, a, input, select, textarea')) return;

      const rect = container.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects([moonMesh], true);

      const moonScreenPos = moonGroup.position.clone().project(camera);
      const distToMoon = Math.hypot(mouse.x - moonScreenPos.x, mouse.y - moonScreenPos.y);

      if (isWorkbenchModeRef.current) {
        isDragging.current = true;
        previousPointerPosition.current = { x: e.clientX, y: e.clientY };
        pointerDownPos.current = { x: e.clientX, y: e.clientY };
        if (hoveredCraterRef.current) {
          hoveredCraterRef.current = null;
          setHoveredCrater(null);
        }
      } else {
        if (intersects.length > 0 || distToMoon < 0.28) {
          isDragging.current = true;
          previousPointerPosition.current = { x: e.clientX, y: e.clientY };
          pointerDownPos.current = { x: e.clientX, y: e.clientY };
        }
      }
    };

    const handlePointerMove = (e: PointerEvent) => {
      if (isDragging.current && moonGroupRef.current) {
        const deltaX = e.clientX - previousPointerPosition.current.x;
        const deltaY = e.clientY - previousPointerPosition.current.y;

        targetRotationY.current += deltaX * 0.005;
        targetRotationX.current += deltaY * 0.005;
        // Allow full selenographic polar rotation from -90° to +90°
        targetRotationX.current = Math.max(-1.52, Math.min(1.52, targetRotationX.current));

        previousPointerPosition.current = { x: e.clientX, y: e.clientY };
        return;
      }

      // Hover reconnaissance raycast across 3D lunar surface & crater pins
      if (!isDragging.current && isWorkbenchModeRef.current && cameraRef.current && moonMeshRef.current) {
        if ((e.target as HTMLElement)?.closest('[data-sidebar], .sidebar-scroll, [role="dialog"], button, a, input, select, textarea')) {
          if (hoveredCraterRef.current) {
            hoveredCraterRef.current = null;
            setHoveredCrater(null);
          }
          return;
        }

        const rect = container.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, cameraRef.current);

        let hitCrater: CraterDetail | null = null;

        // 1. Direct hit on crater pins
        if (craterPinsGroupRef.current && craterPinsGroupRef.current.visible) {
          const pinIntersects = raycaster.intersectObjects(craterPinsGroupRef.current.children);
          if (pinIntersects.length > 0) {
            hitCrater = pinIntersects[0].object.userData?.crater || null;
          }
        }

        // 2. Proximity raycast on moon mesh
        if (!hitCrater && craters && craters.length > 0) {
          const intersects = raycaster.intersectObject(moonMeshRef.current);
          if (intersects.length > 0) {
            const pt = intersects[0].point.clone();
            moonMeshRef.current.worldToLocal(pt);
            const r = pt.length();
            const lat = Math.asin(Math.max(-1, Math.min(1, pt.y / r))) * (180 / Math.PI);
            const lon = Math.atan2(-pt.z, pt.x) * (180 / Math.PI);

            let nearest = null;
            let minDist = 7.5;
            for (const c of craters) {
              const d = Math.hypot(c.lat - lat, c.lon - lon);
              if (d < minDist) {
                minDist = d;
                nearest = c;
              }
            }
            hitCrater = nearest;
          }
        }

        if (hitCrater) {
          if (hoveredCraterRef.current?.id !== hitCrater.id) {
            hoveredCraterRef.current = hitCrater;
            setHoveredCrater(hitCrater);
          }
          setHoverPos({ x: e.clientX, y: e.clientY });
        } else if (hoveredCraterRef.current) {
          hoveredCraterRef.current = null;
          setHoveredCrater(null);
        }
      }
    };

    const handlePointerUp = (e: PointerEvent) => {
      isDragging.current = false;
      const dist = Math.hypot(e.clientX - pointerDownPos.current.x, e.clientY - pointerDownPos.current.y);
      if (dist < 6 && isWorkbenchModeRef.current && cameraRef.current) {
        const rect = container.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, cameraRef.current);

        // 1. Check if user clicked on a 3D crater pin
        if (craterPinsGroupRef.current && craterPinsGroupRef.current.visible) {
          const pinIntersects = raycaster.intersectObjects(craterPinsGroupRef.current.children);
          if (pinIntersects.length > 0) {
            const crater = pinIntersects[0].object.userData?.crater;
            if (crater) {
              if (hoveredCraterRef.current) {
                hoveredCraterRef.current = null;
                setHoveredCrater(null);
              }
              onSelectCraterRef.current?.(crater);
              return;
            }
          }
        }

        // 2. Click on general lunar surface: select nearest catalog crater if clicked within range
        const intersects = raycaster.intersectObject(moonMesh);
        if (intersects.length > 0 && craters && craters.length > 0) {
          const pt = intersects[0].point.clone();
          moonMesh.worldToLocal(pt);
          const r = pt.length();
          const lat = Math.asin(Math.max(-1, Math.min(1, pt.y / r))) * (180 / Math.PI);
          const lon = Math.atan2(-pt.z, pt.x) * (180 / Math.PI);

          let nearest = null;
          let minDist = 12.0;
          for (const c of craters) {
            const d = Math.hypot(c.lat - lat, c.lon - lon);
            if (d < minDist) {
              minDist = d;
              nearest = c;
            }
          }
          if (nearest) {
            if (hoveredCraterRef.current) {
              hoveredCraterRef.current = null;
              setHoveredCrater(null);
            }
            onSelectCraterRef.current?.(nearest);
          }
        }
      }
    };

    const handleWheel = (e: WheelEvent) => {
      if (!isWorkbenchModeRef.current || isDrawerOpenRef.current) return;
      if ((e.target as HTMLElement)?.closest('[data-sidebar], .sidebar-scroll, [role="dialog"], button, a, input, select, textarea')) return;
      e.preventDefault();
      const zoomStep = e.deltaY * 0.003;
      targetCameraZ.current = Math.max(2.2, Math.min(6.5, targetCameraZ.current + zoomStep));
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('wheel', handleWheel, { passive: false });

    // Responsive Canvas Resize
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      width = container.clientWidth;
      height = container.clientHeight;
      aspect = width / height;
      camera.aspect = aspect;
      updateMoonPosition();
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // 8. Animation Loop
    let orbitAngle = 0;
    const orbitRadius = radius * 1.55; // 1.47 units
    const orbitInclination = Math.PI / 3.8;
    const startTime = performance.now();
    let lastTime = performance.now();

    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Do not waste GPU/CPU cycles when tab is inactive
      if (document.hidden) return;

      const now = performance.now();
      const delta = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      const elapsedTime = (now - startTime) * 0.001;
      const currentMoonX = moonOffsetRef.current.x;

      if (craterPinsGroupRef.current) {
        craterPinsGroupRef.current.visible = Boolean(layersRef.current?.craters);
      }

      // Moon Rotation
      if (moonGroupRef.current) {
        if (!isDragging.current) {
          if (isLaunchingRef.current) {
            targetRotationY.current = THREE.MathUtils.lerp(targetRotationY.current, 0, 0.1);
            targetRotationX.current = THREE.MathUtils.lerp(targetRotationX.current, 0, 0.1);
          } else if (!isWorkbenchModeRef.current) {
            targetRotationY.current += 0.075 * delta;
          }
        }

        moonGroupRef.current.rotation.y +=
          (targetRotationY.current - moonGroupRef.current.rotation.y) * 0.08;
        moonGroupRef.current.rotation.x +=
          (targetRotationX.current - moonGroupRef.current.rotation.x) * 0.08;
      }

      // Pulse 3D Targeting Reticle
      if (targetBeaconGroupRef.current && targetBeaconGroupRef.current.visible) {
        const pulse = 1.0 + 0.22 * Math.sin(elapsedTime * 4.5);
        targetOuterRing.scale.set(pulse, pulse, pulse);
        (targetOuterRing.material as THREE.MeshBasicMaterial).opacity = 0.7 + 0.3 * Math.cos(elapsedTime * 4.5);
        targetInnerRing.scale.set(1.0 / pulse, 1.0 / pulse, 1.0 / pulse);
        (laserPillar.material as THREE.MeshBasicMaterial).opacity = 0.5 + 0.3 * Math.sin(elapsedTime * 3.0);
      }

      (window as any).__moonCheck = {
        hud: Boolean(hudRef.current),
        beacon: Boolean(targetBeaconGroupRef.current),
        beaconVis: Boolean(targetBeaconGroupRef.current?.visible),
        isWorkbench: Boolean(isWorkbenchModeRef.current),
        selectedCrater: Boolean(selectedCraterRef.current),
        craterName: selectedCraterRef.current?.name,
        moonMesh: Boolean(moonMeshRef.current),
      };

      // Project Target Crater to 2D Screen for 60fps Aerospace HUD Callout Pin
      if (
        hudRef.current &&
        targetBeaconGroupRef.current &&
        targetBeaconGroupRef.current.visible &&
        isWorkbenchModeRef.current &&
        selectedCraterRef.current &&
        moonMeshRef.current
      ) {
        const worldPos = new THREE.Vector3();
        targetBeaconGroupRef.current.getWorldPosition(worldPos);

        const moonCenter = new THREE.Vector3();
        moonMeshRef.current.getWorldPosition(moonCenter);
        const surfaceNormal = worldPos.clone().sub(moonCenter).normalize();
        const camDir = camera.position.clone().sub(worldPos).normalize();
        const dot = surfaceNormal.dot(camDir);
        const isFacing = dot > -0.1; // More permissive threshold

        (window as any).__moonDebug = {
          worldPos: [worldPos.x, worldPos.y, worldPos.z],
          moonCenter: [moonCenter.x, moonCenter.y, moonCenter.z],
          camPos: [camera.position.x, camera.position.y, camera.position.z],
          dot,
          isFacing,
          visible: targetBeaconGroupRef.current.visible,
          crater: selectedCraterRef.current?.name,
        };

        if (isFacing) {
          const proj = worldPos.clone().project(camera);
          const w = container.clientWidth;
          const h = container.clientHeight;
          const sx = (proj.x * 0.5 + 0.5) * w;
          const sy = (-proj.y * 0.5 + 0.5) * h;

          hudRef.current.style.display = 'block';
          hudRef.current.style.transform = `translate3d(${sx}px, ${sy}px, 0)`;
        } else {
          hudRef.current.style.display = 'none';
        }
      } else if (hudRef.current) {
        hudRef.current.style.display = 'none';
      }

      // Real 3D Satellite Attitude & Orbit: Only on landing page (smoothly hidden in backend workbench)
      if (satGroupRef.current) {
        const targetScale = isWorkbenchModeRef.current ? 0.0 : (isLaunchingRef.current ? 0.25 : 0.75);
        satGroupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
        satGroupRef.current.visible = satGroupRef.current.scale.x > 0.01;

        if (satGroupRef.current.visible) {
          const orbitSpeed = isLaunchingRef.current ? 1.2 : (Math.PI * 2) / 20.0;
          orbitAngle += orbitSpeed * delta;
          const rawX = Math.cos(orbitAngle) * orbitRadius;
          const rawZ = Math.sin(orbitAngle) * orbitRadius;
          const satPos = new THREE.Vector3(rawX, 0, rawZ);
          satPos.applyAxisAngle(new THREE.Vector3(1, 0.2, 0).normalize(), orbitInclination);

          satGroupRef.current.position.set(
            currentMoonX + satPos.x,
            satPos.y,
            satPos.z
          );

          satGroupRef.current.lookAt(currentMoonX, 0, 0);
          satGroupRef.current.rotateZ(orbitAngle * 0.15);
        }
      }

      // Smooth glide to center on Launch or in workbench mode!
      if (isWorkbenchModeRef.current || isLaunchingRef.current) {
        moonOffsetRef.current.x = THREE.MathUtils.lerp(moonOffsetRef.current.x, 0.0, 0.08);
      } else {
        const asp = container.clientWidth / container.clientHeight;
        const targetX = asp > 1.3 ? 1.50 : asp > 0.85 ? 1.48 : 0.65;
        moonOffsetRef.current.x = THREE.MathUtils.lerp(moonOffsetRef.current.x, targetX, 0.08);
      }
      if (moonGroupRef.current) {
        moonGroupRef.current.position.x = moonOffsetRef.current.x;
      }

      // WebGL Stars subtle drift
      if (starFieldRef.current) {
        starFieldRef.current.rotation.y += 0.0002;
        starFieldRef.current.rotation.x += 0.0001;
      }

      // Camera: smoothly transitions to center and matches the Cesium Workbench Moon scale (Z = 3.8)
      if (cameraRef.current) {
        if (isWorkbenchModeRef.current) {
          cameraRef.current.position.lerp(new THREE.Vector3(0, 0, targetCameraZ.current), 0.08);
        } else if (isLaunchingRef.current) {
          cameraRef.current.position.lerp(new THREE.Vector3(0, 0, 3.8), 0.08);
        } else {
          const asp = container.clientWidth / container.clientHeight;
          const defaultZ = asp > 1.3 ? 4.8 : asp > 0.85 ? 5.2 : 5.8;
          cameraRef.current.position.lerp(new THREE.Vector3(0, 0, defaultZ), 0.05);
        }
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('wheel', handleWheel);
      renderer.dispose();
    };
  }, []);

  const activeCraterDetail = craters.find(
    (c) => c.id === selectedCrater?.id || c.name?.toLowerCase() === selectedCrater?.name?.toLowerCase()
  );
  const activeCraterName = selectedCrater?.name || 'Target Crater';
  const activeCraterId = selectedCrater?.id || activeCraterDetail?.id;
  const activeCraterLat = selectedCrater?.lat ?? activeCraterDetail?.lat ?? 0;
  const activeCraterLon = selectedCrater?.lon ?? activeCraterDetail?.lon ?? 0;
  const activeSensor = getCraterSensorBadge(activeCraterId);
  const activePreviewImg = getCraterPreviewImage(activeCraterId, activeCraterLat);

  return (
    <div
      ref={mountRef}
      className="absolute inset-0 w-full h-full pointer-events-auto select-none overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full"
      />

      {/* 60fps Aerospace HUD Target Reticle & Callout Pin */}
      <div
        ref={hudRef}
        className="pointer-events-none absolute top-0 left-0 z-30 -translate-x-1/2 -translate-y-1/2 will-change-transform"
        style={{ display: 'none' }}
      >
        <div className="relative flex items-center justify-center">
          {/* Animated Sonar Radar Ping */}
          <span className="absolute w-14 h-14 rounded-full border border-[#00f0ff]/50 animate-ping pointer-events-none" />

          {/* Precision Reticle Rings */}
          <span className="absolute w-10 h-10 rounded-full border-2 border-[#00f0ff] shadow-[0_0_15px_#00f0ff] pointer-events-none" />
          <span className="absolute w-6 h-6 rounded-full border border-emerald-400/80 shadow-[0_0_8px_#10b981] pointer-events-none" />
          <span className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_10px_#ffffff] pointer-events-none" />

          {/* Cardinal Crosshairs */}
          <div className="absolute w-16 h-[1.5px] bg-gradient-to-r from-transparent via-[#00f0ff] to-transparent shadow-[0_0_8px_#00f0ff] pointer-events-none" />
          <div className="absolute h-16 w-[1.5px] bg-gradient-to-b from-transparent via-[#00f0ff] to-transparent shadow-[0_0_8px_#00f0ff] pointer-events-none" />

          {/* Tactical Corner Brackets */}
          <div className="absolute -top-6 -left-6 w-2.5 h-2.5 border-t-2 border-l-2 border-[#00f0ff] pointer-events-none" />
          <div className="absolute -top-6 -right-6 w-2.5 h-2.5 border-t-2 border-r-2 border-[#00f0ff] pointer-events-none" />
          <div className="absolute -bottom-6 -left-6 w-2.5 h-2.5 border-b-2 border-l-2 border-[#00f0ff] pointer-events-none" />
          <div className="absolute -bottom-6 -right-6 w-2.5 h-2.5 border-b-2 border-r-2 border-[#00f0ff] pointer-events-none" />

          {/* Floating High-Tech Aerospace Callout Card with Hover Preview */}
          <div className="absolute left-8 -top-3.5 pointer-events-auto flex flex-col group cursor-pointer">
            {/* Primary Pill Header */}
            <div className="flex items-center gap-2.5 px-3.5 py-2 rounded-2xl bg-[#080B11]/92 backdrop-blur-2xl border border-[#00f0ff]/60 shadow-[0_12px_40px_rgba(0,240,255,0.35)] transition-all duration-300 group-hover:border-[#00f0ff] group-hover:shadow-[0_0_30px_rgba(0,240,255,0.6)]">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#00f0ff] shadow-[0_0_8px_#00f0ff] animate-pulse" />
                <span
                  ref={hudNameRef}
                  className="text-xs font-bold font-mono text-white tracking-wider uppercase"
                >
                  {activeCraterName}
                </span>
              </div>
              <div className="flex items-center gap-2 text-[10px] font-mono text-white/70">
                <span ref={hudCoordRef} className="text-[#00f0ff] font-semibold">
                  {selectedCrater ? `${Math.abs(activeCraterLat).toFixed(1)}°${activeCraterLat < 0 ? 'S' : 'N'}, ${Math.abs(activeCraterLon).toFixed(1)}°${activeCraterLon < 0 ? 'W' : 'E'}` : ''}
                </span>
                <span className="text-white/30">·</span>
                <span className="text-emerald-400 font-semibold">TARGET CORRIDOR</span>
              </div>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded-full text-[9px] font-mono text-[#00f0ff] bg-[#00f0ff]/10 border border-[#00f0ff]/30 group-hover:bg-[#00f0ff]/20">
                PREVIEW
              </span>
            </div>

            {/* ── EXPANDED AEROSPACE RECONNAISSANCE DOSSIER (Visible on Hover) ── */}
            <div className="hidden group-hover:flex flex-col mt-2 w-80 rounded-2xl overflow-hidden bg-[#070b14]/95 backdrop-blur-3xl border border-[#00f0ff]/50 shadow-[0_25px_60px_rgba(0,0,0,0.95),0_0_35px_rgba(0,240,255,0.3)] animate-in fade-in zoom-in-95 duration-200">
              {/* High-Resolution Orbital Imagery Preview Frame */}
              <div className="relative h-36 w-full overflow-hidden bg-black">
                <img
                  src={activePreviewImg}
                  alt={activeCraterName}
                  className="w-full h-full object-cover object-center filter contrast-125 brightness-95 group-hover:scale-105 transition-transform duration-700"
                />
                {/* HUD Scanline & Gradient Vignette */}
                <div className="absolute inset-0 bg-gradient-to-t from-[#070b14] via-transparent to-black/50 pointer-events-none" />
                <div className="absolute inset-0 bg-[linear-gradient(rgba(0,240,255,0.08)_1px,transparent_1px)] bg-[size:100%_4px] pointer-events-none" />

                {/* Sensor & Resolution Badges */}
                <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-black/80 backdrop-blur-md border border-[#00f0ff]/40 text-[9px] font-mono text-[#00f0ff] font-bold">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] animate-ping" />
                  <span>{activeSensor.sensor}</span>
                </div>
                <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded-md bg-black/80 backdrop-blur-md border border-white/20 text-[9px] font-mono text-emerald-400 font-bold">
                  {activeSensor.res}
                </div>

                {/* Crosshair Overlay Marks */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 pointer-events-none flex items-center justify-center">
                  <span className="w-full h-[1px] bg-[#00f0ff]/60 absolute" />
                  <span className="h-full w-[1px] bg-[#00f0ff]/60 absolute" />
                  <span className="w-3 h-3 rounded-full border border-[#00f0ff]/80" />
                </div>

                {/* Bottom Coordinates & Region Tag */}
                <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between text-[10px] font-mono text-white/90">
                  <span className="font-semibold text-white truncate max-w-[190px]">
                    {activeCraterDetail?.region || (Math.abs(activeCraterLat) > 60 ? 'South Polar Highlands' : 'Equatorial Plains')}
                  </span>
                  <span className="text-[#00f0ff] font-mono font-bold">
                    {Math.abs(activeCraterLat).toFixed(2)}°{activeCraterLat < 0 ? 'S' : 'N'}
                  </span>
                </div>
              </div>

              {/* Telemetry Reconnaissance Data Grid */}
              <div className="p-3 space-y-2.5 font-mono text-xs">
                <div className="grid grid-cols-2 gap-2 text-[10px]">
                  <div className="p-2 rounded-xl bg-white/[0.04] border border-white/10">
                    <span className="text-white/40 block text-[9px] uppercase tracking-wider">Diameter / Depth</span>
                    <span className="text-white font-semibold text-xs">
                      {activeCraterDetail?.diameterKm ?? (activeCraterId?.includes('copernicus') ? 93 : 97)} km ⌀ · {activeCraterDetail?.depthKm ?? (activeCraterId?.includes('copernicus') ? 3.8 : 4.0)} km
                    </span>
                  </div>
                  <div className="p-2 rounded-xl bg-white/[0.04] border border-white/10">
                    <span className="text-white/40 block text-[9px] uppercase tracking-wider">Water-Ice Signature</span>
                    <span className="text-cyan-300 font-semibold text-xs">
                      {activeCraterDetail?.waterIceConcentrationWtPct ?? (Math.abs(activeCraterLat) > 60 ? 3.8 : 0.4)} wt% H₂O
                    </span>
                  </div>
                </div>

                {/* PSR & Lighting Condition Bar */}
                <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-[#00f0ff]/10 border border-[#00f0ff]/20 text-[10px]">
                  <div className="flex items-center gap-1.5 text-white/90">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    <span>{activeCraterDetail?.psrStatus || (Math.abs(activeCraterLat) > 60 ? 'Permanently Shadowed (PSR)' : 'Sunlit Crater Rim')}</span>
                  </div>
                  <span className="text-[#00f0ff] font-bold">
                    {activeCraterDetail?.surfaceTempKelvin ? `${activeCraterDetail.surfaceTempKelvin} K` : (Math.abs(activeCraterLat) > 60 ? '90 K (-183°C)' : '360 K (+87°C)')}
                  </span>
                </div>

                {/* Action Footer Callout */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onInspectIn2DRef.current?.();
                  }}
                  className="w-full flex items-center justify-between pt-1.5 border-t border-white/10 text-[10px] text-white/60 hover:text-[#00f0ff] transition-colors cursor-pointer"
                >
                  <span className="text-emerald-400 font-semibold flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    TARGET LOCKED
                  </span>
                  <span className="font-semibold tracking-wider flex items-center gap-1 text-[#00f0ff]">
                    SWITCH TO 2D ALIGNMENT →
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Hover Preview Tooltip for Any Other Crater Dot on 3D Moon ── */}
      {hoveredCrater && isWorkbenchMode && hoveredCrater.id !== selectedCrater?.id && (
        <div
          className="fixed z-50 pointer-events-none transition-all duration-100 ease-out"
          style={{
            left: `${Math.min(window.innerWidth - 300, hoverPos.x + 18)}px`,
            top: `${Math.min(window.innerHeight - 220, Math.max(20, hoverPos.y - 40))}px`,
          }}
        >
          <div className="w-72 rounded-2xl overflow-hidden bg-[#070b14]/95 backdrop-blur-3xl border border-[#00f0ff]/60 shadow-[0_20px_50px_rgba(0,0,0,0.95),0_0_25px_rgba(0,240,255,0.35)] animate-in fade-in duration-150">
            {/* Thumbnail banner */}
            <div className="relative h-24 w-full overflow-hidden bg-black">
              <img
                src={getCraterPreviewImage(hoveredCrater.id, hoveredCrater.lat)}
                alt={hoveredCrater.name}
                className="w-full h-full object-cover filter contrast-125 brightness-95"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#070b14] via-transparent to-black/40" />
              <div className="absolute top-1.5 left-2 px-1.5 py-0.5 rounded bg-black/80 border border-[#00f0ff]/40 text-[9px] font-mono text-[#00f0ff] font-bold">
                {getCraterSensorBadge(hoveredCrater.id).sensor}
              </div>
              <div className="absolute bottom-1.5 left-2 right-2 flex items-center justify-between text-[10px] font-mono text-white font-bold truncate">
                <span className="truncate">{hoveredCrater.name}</span>
                <span className="text-[#00f0ff] text-[9px] shrink-0 ml-1">
                  {Math.abs(hoveredCrater.lat).toFixed(1)}°{hoveredCrater.lat < 0 ? 'S' : 'N'}
                </span>
              </div>
            </div>
            {/* Telemetry info */}
            <div className="p-2.5 space-y-1.5 font-mono text-[10px]">
              <div className="flex items-center justify-between text-white/80">
                <span className="text-[#00f0ff]">
                  {Math.abs(hoveredCrater.lat).toFixed(1)}°{hoveredCrater.lat < 0 ? 'S' : 'N'}, {Math.abs(hoveredCrater.lon).toFixed(1)}°{hoveredCrater.lon < 0 ? 'W' : 'E'}
                </span>
                <span className="text-white/50">{hoveredCrater.region || 'Lunar Surface'}</span>
              </div>
              <div className="flex items-center justify-between text-white/70 pt-1 border-t border-white/10">
                <span>⌀ {hoveredCrater.diameterKm ?? (hoveredCrater as any).diameter_km ?? 50} km</span>
                <span className="text-cyan-300">
                  {hoveredCrater.waterIceConcentrationWtPct ?? (Math.abs(hoveredCrater.lat) > 60 ? '3.8 wt% H₂O' : '0.4 wt%')}
                </span>
              </div>
              <div className="text-[9px] text-emerald-400 font-semibold text-center pt-0.5 flex items-center justify-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                CLICK CRATER TO TARGET & ROTATE
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
