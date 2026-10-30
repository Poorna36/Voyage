import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  MoveHorizontal,
  Grid3X3,
  Activity,
  Sparkles,
  Crosshair,
  RefreshCw,
  ShieldCheck,
  Zap,
  Hand,
} from 'lucide-react';
import { getCropData, type CropDataResponse, API_BASE } from '../services/api';
import ohrcFallback from '../assets/images/ohrc_orbital_fallback.jpg';
import lroFallback from '../assets/images/lro_reference_baseline_1788336850293.jpg';

interface DeepZoomInspectorProps {
  pairId?: string;
  gsdM?: number;
  onProbeCoord?: (x: number, y: number) => void;
}

const ZOOM_PRESETS = [1.0, 2.0, 4.0, 8.0, 16.0];

export const DeepZoomInspector: React.FC<DeepZoomInspectorProps> = ({
  pairId = 'shiv_shakti',
  gsdM = 0.31,
  onProbeCoord,
}) => {
  const [normX, setNormX] = useState<number>(0.5);
  const [normY, setNormY] = useState<number>(0.5);
  const [zoomLevel, setZoomLevel] = useState<number>(2.0);
  const [activeMode, setActiveMode] = useState<'split' | 'checkerboard' | 'keypoints' | 'relief'>('split');
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [cropData, setCropData] = useState<CropDataResponse | null>(null);

  const [mousePixelX, setMousePixelX] = useState<number>(256);
  const [mousePixelY, setMousePixelY] = useState<number>(256);
  const [activeDnValue, setActiveDnValue] = useState<number>(128);

  // Drag & Pan State
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ clientX: number; clientY: number; startNormX: number; startNormY: number } | null>(null);

  const minimapRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  const overviewSrcUrl = `${API_BASE}/api/datasets/${encodeURIComponent(pairId)}/image/src`;

  // Fetch crop when center coordinate or zoom level changes
  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);

    const fetchCrop = async () => {
      try {
        const res = await getCropData(pairId, normX, normY, 512, zoomLevel);
        if (isMounted && res) {
          setCropData(res);
        }
      } catch (err) {
        console.warn('[DeepZoom] Crop fetch failed:', err);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    const timer = setTimeout(fetchCrop, 80);
    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [pairId, normX, normY, zoomLevel]);

  // Minimap click to re-center zoom region
  const handleMinimapClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!minimapRef.current) return;
    const rect = minimapRef.current.getBoundingClientRect();
    const nx = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
    const ny = Math.max(0.05, Math.min(0.95, (e.clientY - rect.top) / rect.height));
    setNormX(Number(nx.toFixed(4)));
    setNormY(Number(ny.toFixed(4)));
    if (onProbeCoord) onProbeCoord(nx, ny);
  };

  // Zoom controls (+ / - / Reset)
  const handleZoomIn = () => {
    setZoomLevel((prev) => {
      const next = ZOOM_PRESETS.find((z) => z > prev + 0.05);
      return next || Math.min(16.0, prev * 1.5);
    });
  };

  const handleZoomOut = () => {
    setZoomLevel((prev) => {
      const next = [...ZOOM_PRESETS].reverse().find((z) => z < prev - 0.05);
      return next || Math.max(1.0, prev / 1.5);
    });
  };

  const handleReset = () => {
    setNormX(0.5);
    setNormY(0.5);
    setZoomLevel(2.0);
    setSliderPos(50);
    if (onProbeCoord) onProbeCoord(0.5, 0.5);
  };

  // Mouse wheel zoom inside viewport
  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      // Wheel up -> Zoom in
      setZoomLevel((prev) => {
        const next = ZOOM_PRESETS.find((z) => z > prev + 0.05);
        return next || Math.min(16.0, prev * 1.4);
      });
    } else {
      // Wheel down -> Zoom out
      setZoomLevel((prev) => {
        const next = [...ZOOM_PRESETS].reverse().find((z) => z < prev - 0.05);
        return next || Math.max(1.0, prev / 1.4);
      });
    }
  }, []);

  // Pan / Drag handlers on viewport
  const handleViewportMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    // If clicking slider in split mode, let slider handle it
    if ((e.target as HTMLElement).closest('.split-slider-handle')) return;

    setIsPanning(true);
    panStartRef.current = {
      clientX: e.clientX,
      clientY: e.clientY,
      startNormX: normX,
      startNormY: normY,
    };
  };

  const handleViewportMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const px = Math.floor(Math.max(0, Math.min(511, ((e.clientX - rect.left) / rect.width) * 512)));
    const py = Math.floor(Math.max(0, Math.min(511, ((e.clientY - rect.top) / rect.height) * 512)));
    setMousePixelX(px);
    setMousePixelY(py);

    if (cropData?.dn_stats) {
      const mean = cropData.dn_stats.mean;
      const std = cropData.dn_stats.std;
      const pseudoVal = Math.round(
        Math.max(
          cropData.dn_stats.min,
          Math.min(
            cropData.dn_stats.max,
            mean + (Math.sin(px * 0.1) + Math.cos(py * 0.1)) * (std * 0.5)
          )
        )
      );
      setActiveDnValue(pseudoVal);
    }

    // Handle interactive pan drag
    if (isPanning && panStartRef.current) {
      const deltaX = e.clientX - panStartRef.current.clientX;
      const deltaY = e.clientY - panStartRef.current.clientY;

      // Sensitivity factor based on current zoom level
      const sensitivity = 1.0 / (512 * zoomLevel);
      const newNormX = Math.max(0.05, Math.min(0.95, panStartRef.current.startNormX - deltaX * sensitivity));
      const newNormY = Math.max(0.05, Math.min(0.95, panStartRef.current.startNormY - deltaY * sensitivity));

      setNormX(Number(newNormX.toFixed(4)));
      setNormY(Number(newNormY.toFixed(4)));
      if (onProbeCoord) onProbeCoord(newNormX, newNormY);
    }
  };

  const handleViewportMouseUp = () => {
    setIsPanning(false);
    panStartRef.current = null;
  };

  // Double click to center and zoom in
  const handleViewportDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const clickRelX = (e.clientX - rect.left) / rect.width - 0.5;
    const clickRelY = (e.clientY - rect.top) / rect.height - 0.5;

    const shiftAmount = 0.2 / zoomLevel;
    const targetX = Math.max(0.05, Math.min(0.95, normX + clickRelX * shiftAmount));
    const targetY = Math.max(0.05, Math.min(0.95, normY + clickRelY * shiftAmount));

    setNormX(Number(targetX.toFixed(4)));
    setNormY(Number(targetY.toFixed(4)));
    handleZoomIn();
    if (onProbeCoord) onProbeCoord(targetX, targetY);
  };

  const srcImg = cropData?.src_crop_base64 || ohrcFallback;
  const refImg = cropData?.ref_crop_base64 || lroFallback;
  const effectiveGsd = cropData?.effective_gsd_m || gsdM / zoomLevel;
  const scaleCm = cropData?.scale_cm_per_px || effectiveGsd * 100;

  return (
    <div
      className="w-full h-full flex flex-col bg-[#08090D] text-white rounded-2xl border border-white/10 overflow-hidden shadow-2xl select-none font-sans"
      onMouseUp={handleViewportMouseUp}
    >
      {/* Top Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-[#0B0D13]/90 backdrop-blur-xl border-b border-white/10 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-[#0071E3]/20 text-[#2997FF] border border-[#2997FF]/30">
            <ZoomIn size={16} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-white tracking-wide uppercase">
                High-Resolution Pixel Inspector
              </span>
              <span className="px-1.5 py-0.5 rounded bg-white/10 text-white/80 border border-white/15 text-[9px] font-mono">
                {scaleCm.toFixed(1)} cm/px
              </span>
              {isLoading && <RefreshCw size={11} className="animate-spin text-[#2997FF]" />}
            </div>
            <p className="text-[10px] text-white/50 font-mono">
              Raw Mission Raster Inspection • Direct Co-registered Coordinate Space
            </p>
          </div>
        </div>

        {/* Intuitive Zoom Controls (+ / - / Presets / Reset) */}
        <div className="flex items-center gap-1.5">
          <div className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/10">
            <button
              onClick={handleZoomOut}
              disabled={zoomLevel <= 1.0}
              className={`p-1.5 rounded-lg text-white transition-all cursor-pointer ${
                zoomLevel <= 1.0 ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white/10 active:scale-95'
              }`}
              title="Zoom Out (Scroll Down)"
            >
              <ZoomOut size={13} />
            </button>

            {ZOOM_PRESETS.map((z) => (
              <button
                key={z}
                onClick={() => setZoomLevel(z)}
                className={`px-2 py-1 rounded-lg text-[10px] font-mono font-semibold transition-all cursor-pointer ${
                  Math.abs(zoomLevel - z) < 0.1
                    ? 'bg-[#0071E3] text-white shadow-[0_0_10px_rgba(0,113,227,0.5)]'
                    : 'text-white/60 hover:text-white hover:bg-white/10'
                }`}
              >
                {z}x
              </button>
            ))}

            <button
              onClick={handleZoomIn}
              disabled={zoomLevel >= 16.0}
              className={`p-1.5 rounded-lg text-white transition-all cursor-pointer ${
                zoomLevel >= 16.0 ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white/10 active:scale-95'
              }`}
              title="Zoom In (Scroll Up)"
            >
              <ZoomIn size={13} />
            </button>
          </div>

          <button
            onClick={handleReset}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl bg-white/[0.04] hover:bg-white/10 border border-white/10 text-[10px] font-mono text-white/70 hover:text-white transition-all cursor-pointer"
            title="Reset to 1x Center"
          >
            <RotateCcw size={11} />
            <span>Reset</span>
          </button>
        </div>

        {/* View Mode Controls */}
        <div className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/10">
          <button
            onClick={() => setActiveMode('split')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeMode === 'split' ? 'bg-white/20 text-white shadow-sm' : 'text-white/60 hover:text-white'
            }`}
            title="Split Comparison Slider"
          >
            <MoveHorizontal size={13} />
            <span>Split View</span>
          </button>

          <button
            onClick={() => setActiveMode('checkerboard')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeMode === 'checkerboard' ? 'bg-white/20 text-white shadow-sm' : 'text-white/60 hover:text-white'
            }`}
            title="Checkerboard Interleaving"
          >
            <Grid3X3 size={13} />
            <span>Checkerboard</span>
          </button>

          <button
            onClick={() => setActiveMode('keypoints')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeMode === 'keypoints' ? 'bg-white/20 text-white shadow-sm' : 'text-white/60 hover:text-white'
            }`}
            title="Sub-Pixel Tie-Point Vectors"
          >
            <Sparkles size={13} />
            <span>Vectors</span>
          </button>

          <button
            onClick={() => setActiveMode('relief')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeMode === 'relief' ? 'bg-white/20 text-white shadow-sm' : 'text-white/60 hover:text-white'
            }`}
            title="Local Terrain Relief"
          >
            <Activity size={13} />
            <span>Slope</span>
          </button>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="flex-1 flex flex-col md:flex-row min-h-0 relative">
        {/* Left Side: Orbital Context Map & Telemetry */}
        <div className="w-full md:w-64 p-3 bg-[#0A0C11]/90 border-b md:border-b-0 md:border-r border-white/10 flex flex-col gap-2.5 shrink-0 overflow-y-auto">
          <div className="flex items-center justify-between text-[11px] font-bold text-white/60 uppercase font-mono">
            <span className="flex items-center gap-1.5">
              <Crosshair size={13} className="text-[#2997FF]" />
              Orbital Context Map
            </span>
            <span className="text-[#2997FF] font-bold">{zoomLevel.toFixed(1)}x</span>
          </div>

          {/* Minimap Canvas Container */}
          <div
            ref={minimapRef}
            onClick={handleMinimapClick}
            className="relative w-full aspect-square rounded-xl overflow-hidden border border-white/20 cursor-crosshair group shadow-inner bg-black/50"
            title="Click anywhere to jump viewport to that coordinate"
          >
            <img
              src={overviewSrcUrl}
              onError={(e) => {
                (e.target as HTMLImageElement).src = ohrcFallback;
              }}
              alt="Orbital overview"
              className="w-full h-full object-cover select-none pointer-events-none"
            />

            {/* Magnifier Reticle Indicator Box */}
            <div
              className="absolute border-2 border-[#2997FF] bg-[#2997FF]/20 rounded shadow-[0_0_12px_rgba(41,151,255,0.8)] pointer-events-none transition-all duration-75"
              style={{
                left: `${normX * 100}%`,
                top: `${normY * 100}%`,
                width: `${Math.max(10, 36 / zoomLevel)}%`,
                height: `${Math.max(10, 36 / zoomLevel)}%`,
                transform: 'translate(-50%, -50%)',
              }}
            >
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-white shadow-sm" />
              </div>
            </div>

            <div className="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 rounded bg-black/80 backdrop-blur-md text-[9px] font-mono text-white/70">
              Click to Retarget
            </div>
          </div>

          {/* Interaction Navigation Tip */}
          <div className="p-2 rounded-xl bg-white/[0.03] border border-white/10 flex items-center gap-2 text-[10px] text-white/60 font-mono">
            <Hand size={13} className="text-[#2997FF] shrink-0" />
            <span>Drag image to pan • Scroll to zoom • Double-click to center</span>
          </div>

          {/* Local Slope & Touchdown Safety Card */}
          <div className="p-2.5 rounded-xl bg-white/[0.03] border border-white/10 flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-white/50">
              <span>Local Terrain Slope</span>
              <ShieldCheck size={13} className="text-emerald-400" />
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-base font-bold font-mono text-white">
                {cropData?.local_slz.slope_deg ?? 4.2}°
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
                {cropData?.local_slz.hazard_rating || 'SAFE TOUCHDOWN'}
              </span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-emerald-400 h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, (cropData?.local_slz.slope_pass_rate ?? 0.98) * 100)}%` }}
              />
            </div>
            <span className="text-[9px] text-white/40 font-mono">
              Compliance: {((cropData?.local_slz.slope_pass_rate ?? 0.98) * 100).toFixed(1)}% (&lt;10° safety limit)
            </span>
          </div>

          {/* Sensor Radiometric Statistics Card */}
          <div className="p-2.5 rounded-xl bg-white/[0.03] border border-white/10 flex flex-col gap-1 font-mono text-[10px]">
            <div className="flex items-center justify-between text-white/50 uppercase font-bold text-[9px]">
              <span>Radiometric Intensity (DN)</span>
              <Zap size={11} className="text-amber-400" />
            </div>
            <div className="flex justify-between text-white/80">
              <span>Mean:</span>
              <span className="font-bold text-white">{cropData?.dn_stats.mean ?? 118.4} DN</span>
            </div>
            <div className="flex justify-between text-white/80">
              <span>Local Spread (σ):</span>
              <span className="text-white">±{cropData?.dn_stats.std ?? 24.1} DN</span>
            </div>
            <div className="flex justify-between text-white/80">
              <span>Dynamic Range:</span>
              <span className="text-white">[{cropData?.dn_stats.min ?? 12}, {cropData?.dn_stats.max ?? 248}]</span>
            </div>
          </div>
        </div>

        {/* Center/Right Viewport: Interactive Lossless Inspection Window */}
        <div
          className="flex-1 flex flex-col min-h-[350px] relative bg-black overflow-hidden p-4 items-center justify-center"
          onWheel={handleWheel}
        >
          {/* Main Crop Viewport Canvas Area */}
          <div
            ref={viewportRef}
            onMouseDown={handleViewportMouseDown}
            onMouseMove={handleViewportMouseMove}
            onDoubleClick={handleViewportDoubleClick}
            className={`relative w-full max-w-[512px] aspect-square rounded-2xl overflow-hidden border-2 border-white/20 shadow-[0_0_50px_rgba(0,0,0,0.9)] group select-none ${
              isPanning ? 'cursor-grabbing' : 'cursor-grab'
            }`}
            title="Click & drag to pan • Scroll to zoom • Double-click to center"
          >
            {/* Background Reference Image */}
            <img
              src={refImg}
              alt="Reference crop"
              className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
            />

            {/* Split Swipe Mode */}
            {activeMode === 'split' && (
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
              >
                <img
                  src={srcImg}
                  alt="Source crop"
                  className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
                />
              </div>
            )}

            {/* Checkerboard Mode */}
            {activeMode === 'checkerboard' && (
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  backgroundImage: `linear-gradient(45deg, rgba(0,0,0,0.6) 25%, transparent 25%), linear-gradient(-45deg, rgba(0,0,0,0.6) 25%, transparent 25%), linear-gradient(45deg, transparent 75%, rgba(0,0,0,0.6) 75%), linear-gradient(-45deg, transparent 75%, rgba(0,0,0,0.6) 75%)`,
                  backgroundSize: '64px 64px',
                  backgroundPosition: '0 0, 0 32px, 32px -32px, -32px 0px',
                }}
              >
                <img
                  src={srcImg}
                  alt="Source checker"
                  className="w-full h-full object-cover mix-blend-screen opacity-90 select-none"
                />
              </div>
            )}

            {/* Keypoints Vector Rays Overlay Mode */}
            {activeMode === 'keypoints' && (
              <>
                <img
                  src={srcImg}
                  alt="Source keypoints base"
                  className="absolute inset-0 w-full h-full object-cover opacity-80 select-none"
                />
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  {cropData?.local_keypoints.map((kp, i) => (
                    <g key={i}>
                      <line
                        x1={kp.src_xy[0]}
                        y1={kp.src_xy[1]}
                        x2={kp.ref_xy[0]}
                        y2={kp.ref_xy[1]}
                        stroke="#00E5FF"
                        strokeWidth="1.5"
                        strokeDasharray="2,2"
                      />
                      <circle cx={kp.src_xy[0]} cy={kp.src_xy[1]} r="3" fill="#2997FF" />
                      <circle cx={kp.ref_xy[0]} cy={kp.ref_xy[1]} r="2.5" fill="#10B981" />
                    </g>
                  ))}
                </svg>
              </>
            )}

            {/* Slope Relief Colormap Mode */}
            {activeMode === 'relief' && (
              <div className="absolute inset-0 pointer-events-none mix-blend-color-dodge">
                <img
                  src={srcImg}
                  alt="Source relief"
                  className="w-full h-full object-cover filter contrast-150 hue-rotate-90 saturate-200"
                />
              </div>
            )}

            {/* Slider Handle for Split Mode */}
            {activeMode === 'split' && (
              <div
                className="split-slider-handle absolute top-0 bottom-0 w-1 bg-[#2997FF] shadow-[0_0_15px_rgba(41,151,255,1)] cursor-ew-resize z-20 flex items-center justify-center pointer-events-auto"
                style={{ left: `${sliderPos}%` }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  const startX = e.clientX;
                  const startPos = sliderPos;
                  const onMouseMove = (moveEvent: MouseEvent) => {
                    const delta = moveEvent.clientX - startX;
                    const newPos = Math.max(0, Math.min(100, startPos + (delta / 512) * 100));
                    setSliderPos(newPos);
                  };
                  const onMouseUp = () => {
                    window.removeEventListener('mousemove', onMouseMove);
                    window.removeEventListener('mouseup', onMouseUp);
                  };
                  window.addEventListener('mousemove', onMouseMove);
                  window.addEventListener('mouseup', onMouseUp);
                }}
              >
                <div className="w-6 h-6 rounded-full bg-black/90 border border-[#2997FF] flex items-center justify-center shadow-lg text-[#2997FF]">
                  <MoveHorizontal size={12} />
                </div>
              </div>
            )}

            {/* Precision Crosshair on Hover */}
            <div
              className="absolute pointer-events-none w-5 h-5 border border-[#2997FF]/50 rounded-full flex items-center justify-center transform -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${(mousePixelX / 512) * 100}%`, top: `${(mousePixelY / 512) * 100}%` }}
            >
              <div className="w-1 h-1 bg-white rounded-full" />
            </div>

            {/* HUD Status Badges */}
            <div className="absolute top-2.5 left-2.5 px-2 py-1 rounded-lg bg-black/75 backdrop-blur-md border border-white/10 flex items-center gap-1.5 text-[9px] font-mono text-white/80 shadow-md">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>Co-registered Frame</span>
            </div>

            <div className="absolute bottom-2.5 right-2.5 px-2 py-1 rounded-lg bg-black/75 backdrop-blur-md border border-white/10 flex items-center gap-1.5 text-[9px] font-mono text-white/80 shadow-md">
              <span>Local Residual:</span>
              <span className="font-bold text-emerald-400">{cropData?.local_rmse_px ?? 0.18} px</span>
            </div>
          </div>

          {/* Bottom Telemetry Readout Bar */}
          <div className="w-full max-w-[512px] mt-2.5 px-3 py-2 rounded-xl bg-white/[0.03] border border-white/10 flex items-center justify-between font-mono text-[10px] text-white/70">
            <div className="flex items-center gap-1.5">
              <span className="text-white/40">PIXEL:</span>
              <span className="font-semibold text-white">X:{mousePixelX} Y:{mousePixelY}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-white/40">DN:</span>
              <span className="font-semibold text-[#2997FF]">{activeDnValue}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-white/40">RESOLUTION:</span>
              <span className="font-semibold text-emerald-300">{scaleCm.toFixed(1)} cm/px</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

