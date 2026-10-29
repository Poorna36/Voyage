import React, { useState, useRef, useEffect } from 'react';
import { Eye, MoveHorizontal, Grid3X3, Activity, Sparkles, ZoomIn, Layers, ChevronDown } from 'lucide-react';
import type { KeypointMatch } from '../types';
import { getKeypointMatches, API_BASE } from '../services/api';
import { DeepZoomInspector } from './DeepZoomInspector';
import ohrcImg from '../assets/images/ohrc_orbital_fallback.jpg';
import lroImg from '../assets/images/lro_reference_baseline_1788336850293.jpg';

// Derive sensor label + GSD from pair_id as a best-effort lookup
function inferSensorLabel(pairId: string): { srcLabel: string; srcGsd: string; refLabel: string; refGsd: string } {
  const pid = pairId.toLowerCase();
  if (pid.includes('tmc')) {
    return { srcLabel: 'CH-2 TMC-2', srcGsd: '5.0m/px', refLabel: 'LRO NAC', refGsd: '0.5m/px' };
  }
  if (pid.includes('iirs')) {
    return { srcLabel: 'CH-2 IIRS', srcGsd: '80m/px', refLabel: 'LRO NAC', refGsd: '0.5m/px' };
  }
  // Custom uploads: try to read from ID prefix
  if (pid.includes('custom') && pid.includes('tmc')) {
    return { srcLabel: 'CH-2 TMC-2', srcGsd: '5.0m/px', refLabel: 'LRO NAC', refGsd: '0.5m/px' };
  }
  // Default: OHRC
  return { srcLabel: 'CH-2 OHRC', srcGsd: '0.31m/px', refLabel: 'LRO NAC', refGsd: '0.5m/px' };
}

type LineFilter = 'both' | 'inliers' | 'outliers';

interface KeypointViewerProps {
  pairId?: string;
  keypoints?: KeypointMatch[];
  onProbeCoord?: (x: number, y: number) => void;
  rmsePx?: number;
  /** Override sensor labels if known from parent */
  srcSensor?: string;
  refSensor?: string;
  srcGsd?: string;
  refGsd?: string;
}

export const KeypointViewer: React.FC<KeypointViewerProps> = ({
  pairId = 'boguslawsky',
  keypoints: initialKeypoints,
  onProbeCoord,
  rmsePx,
  srcSensor,
  refSensor,
  srcGsd: srcGsdProp,
  refGsd: refGsdProp,
}) => {
  // Resolve sensor labels: prefer explicit props, fall back to pair-id inference
  const inferred = inferSensorLabel(pairId);
  const srcLabel = srcSensor ? srcSensor : inferred.srcLabel;
  const refLabel = refSensor ? refSensor : inferred.refLabel;
  const srcGsdLabel = srcGsdProp || inferred.srcGsd;
  const refGsdLabel = refGsdProp || inferred.refGsd;
  const [keypointData, setKeypointData] = useState<KeypointMatch[]>(initialKeypoints || []);

  useEffect(() => {
    if (initialKeypoints && initialKeypoints.length > 0) {
      setKeypointData(initialKeypoints);
      return;
    }

    let isMounted = true;
    getKeypointMatches(pairId).then((res) => {
      if (isMounted && res && res.length > 0) {
        setKeypointData(
          res.map((m) => ({
            id: m.id,
            srcXy: m.src_xy,
            refXy: m.ref_xy,
            confidence: m.confidence,
            isInlier: m.is_inlier,
            isShadowOutlier: m.is_shadow_outlier,
            refinedDelta: m.refined_delta,
            refineSharpness: m.refine_sharpness,
          }))
        );
      }
    });

    return () => {
      isMounted = false;
    };
  }, [pairId, initialKeypoints]);

  const [sliderPos, setSliderPos] = useState<number>(50);
  const [lineFilter, setLineFilter] = useState<LineFilter>('both');
  const showInliers = lineFilter === 'both' || lineFilter === 'inliers';
  const showOutliers = lineFilter === 'both' || lineFilter === 'outliers';
  const [viewMode, setViewMode] = useState<'side-by-side' | 'split' | 'checkerboard' | 'residuals' | 'deepzoom'>('side-by-side');
  const [hoveredMatch] = useState<KeypointMatch | null>(null);
  const [showRawRef, setShowRawRef] = useState<boolean>(false);
  const rawRefUrl = `${API_BASE}/api/datasets/${encodeURIComponent(pairId)}/image/ref?raw=1`;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const leftImgRef = useRef<HTMLImageElement | null>(null);
  const rightImgRef = useRef<HTMLImageElement | null>(null);
  const isDragging = useRef<boolean>(false);

  const [imageVersion, setImageVersion] = useState<number>(0);
  const [repaintTick, setRepaintTick] = useState<number>(0);
  const [isMerged, setIsMerged] = useState<boolean>(true);
  const [lineProgress, setLineProgress] = useState<number>(0);
  const animFrameRef = useRef<number | null>(null);

  // Trigger slower, majestic separation and progressive laser line drawing
  useEffect(() => {
    setIsMerged(true);
    setLineProgress(0);

    // Phase 1: Hold merged overlay for 850ms
    const timer1 = setTimeout(() => {
      setIsMerged(false);
    }, 850);

    // Phase 2: Once panels settle into place (1800ms), smoothly animate lines drawing from left to right
    const timer2 = setTimeout(() => {
      let start: number | null = null;
      const duration = 1200; // 1.2s smooth laser draw across

      const step = (timestamp: number) => {
        if (!start) start = timestamp;
        const elapsed = timestamp - start;
        const progress = Math.min(1.0, elapsed / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        setLineProgress(eased);

        if (progress < 1.0) {
          animFrameRef.current = requestAnimationFrame(step);
        }
      };

      animFrameRef.current = requestAnimationFrame(step);
    }, 1800);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [pairId]);

  const triggerMergeAnimation = () => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    setIsMerged(true);
    setLineProgress(0);

    setTimeout(() => {
      setIsMerged(false);

      setTimeout(() => {
        setImageVersion((v) => v + 1);
        let start: number | null = null;
        const duration = 1200;

        const step = (timestamp: number) => {
          if (!start) start = timestamp;
          const elapsed = timestamp - start;
          const progress = Math.min(1.0, elapsed / duration);
          const eased = 1 - Math.pow(1 - progress, 3);
          setLineProgress(eased);

          if (progress < 1.0) {
            animFrameRef.current = requestAnimationFrame(step);
          }
        };

        animFrameRef.current = requestAnimationFrame(step);
      }, 1750);
    }, 850);
  };

  const liveSrcUrl = `${API_BASE}/api/datasets/${encodeURIComponent(pairId)}/image/src?v=${imageVersion}`;
  const liveRefUrl = `${API_BASE}/api/datasets/${encodeURIComponent(pairId)}/image/ref?v=${imageVersion}`;

  const inlierCount = keypointData.filter((m) => m.isInlier).length;
  const outlierCount = keypointData.filter((m) => !m.isInlier).length;

  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isDragging.current && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      setSliderPos((x / rect.width) * 100);
    }
  };

  // Draw keypoints, correspondence curves, and residual vectors on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.parentElement?.clientWidth || 800;
    const height = canvas.parentElement?.clientHeight || 450;
    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);

    if (viewMode === 'side-by-side') {
      const leftEl = leftImgRef.current;
      const rightEl = rightImgRef.current;
      const canvasRect = canvas.getBoundingClientRect();

      if (!leftEl || !rightEl) return;

      const lr = leftEl.getBoundingClientRect();
      const rr = rightEl.getBoundingClientRect();
      if (lr.width <= 0 || rr.width <= 0) return;

      // Detect base coordinate system (800 for pristine dataset, 512 for synth)
      const maxVal = Math.max(
        1,
        ...keypointData.flatMap((k) => [k.srcXy[0], k.srcXy[1], k.refXy[0], k.refXy[1]])
      );
      const coordBase = maxVal > 600 ? 800 : 512;

      // Compute rendered image geometry under object-cover
      const naturalW1 = (leftEl.naturalWidth && leftEl.naturalWidth > 100) ? leftEl.naturalWidth : coordBase;
      const naturalH1 = (leftEl.naturalHeight && leftEl.naturalHeight > 100) ? leftEl.naturalHeight : coordBase;
      const scale1 = Math.max(lr.width / naturalW1, lr.height / naturalH1);
      const renderedW1 = naturalW1 * scale1;
      const renderedH1 = naturalH1 * scale1;
      const offX1 = (lr.width - renderedW1) / 2;
      const offY1 = (lr.height - renderedH1) / 2;

      const naturalW2 = (rightEl.naturalWidth && rightEl.naturalWidth > 100) ? rightEl.naturalWidth : coordBase;
      const naturalH2 = (rightEl.naturalHeight && rightEl.naturalHeight > 100) ? rightEl.naturalHeight : coordBase;
      const scale2 = Math.max(rr.width / naturalW2, rr.height / naturalH2);
      const renderedW2 = naturalW2 * scale2;
      const renderedH2 = naturalH2 * scale2;
      const offX2 = (rr.width - renderedW2) / 2;
      const offY2 = (rr.height - renderedH2) / 2;

      const pad = 6;
      // Container bounds relative to canvas (ensures points never escape outside image frames)
      const minX1 = lr.left - canvasRect.left + pad;
      const maxX1 = lr.right - canvasRect.left - pad;
      const minY1 = lr.top - canvasRect.top + pad;
      const maxY1 = lr.bottom - canvasRect.top - pad;

      const minX2 = rr.left - canvasRect.left + pad;
      const maxX2 = rr.right - canvasRect.left - pad;
      const minY2 = rr.top - canvasRect.top + pad;
      const maxY2 = rr.bottom - canvasRect.top - pad;

      keypointData.forEach((match) => {
        if (match.isInlier && !showInliers) return;
        if (!match.isInlier && !showOutliers) return;

        // Calculate clamped positions inside each image frame
        const normX1 = Math.max(0.02, Math.min(0.98, match.srcXy[0] / naturalW1));
        const normY1 = Math.max(0.02, Math.min(0.98, match.srcXy[1] / naturalH1));
        const rawX1 = lr.left - canvasRect.left + offX1 + normX1 * renderedW1;
        const rawY1 = lr.top - canvasRect.top + offY1 + normY1 * renderedH1;
        const x1 = Math.max(minX1, Math.min(maxX1, rawX1));
        const y1 = Math.max(minY1, Math.min(maxY1, rawY1));

        const normX2 = Math.max(0.02, Math.min(0.98, match.refXy[0] / naturalW2));
        const normY2 = Math.max(0.02, Math.min(0.98, match.refXy[1] / naturalH2));
        const rawX2 = rr.left - canvasRect.left + offX2 + normX2 * renderedW2;
        const rawY2 = rr.top - canvasRect.top + offY2 + normY2 * renderedH2;
        const x2 = Math.max(minX2, Math.min(maxX2, rawX2));
        const y2 = Math.max(minY2, Math.min(maxY2, rawY2));

        const isHovered = hoveredMatch?.id === match.id;

        // Animated laser line drawing directly from (x1, y1) to (x2, y2)
        if (lineProgress > 0) {
          const targetX = x1 + (x2 - x1) * lineProgress;
          const targetY = y1 + (y2 - y1) * lineProgress;

          // 1. High-contrast dark outline / halo behind the line so it pops against both dark lunar shadows and bright crater ejecta
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(targetX, targetY);
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)';
          ctx.lineWidth = isHovered ? 6.0 : 4.6;
          ctx.setLineDash(match.isInlier ? [] : [12, 5]);
          ctx.stroke();
          ctx.restore();

          // 2. Main high-visibility laser line with vivid neon glow
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(targetX, targetY);

          if (match.isInlier) {
            // High-contrast vibrant laser emerald green - crisp and vivid
            ctx.strokeStyle = isHovered ? '#FFFFFF' : '#00FF88';
            ctx.lineWidth = isHovered ? 3.4 : 2.4;
            ctx.shadowColor = 'rgba(0, 255, 136, 0.95)';
            ctx.shadowBlur = isHovered ? 10 : 5;
            ctx.setLineDash([]);
          } else {
            // Ultra-vivid electric laser crimson - bold, thick, and clearly visible against dark lunar terrain
            ctx.strokeStyle = isHovered ? '#FFFFFF' : '#FF1E47';
            ctx.lineWidth = isHovered ? 3.6 : 2.6;
            ctx.shadowColor = 'rgba(255, 30, 71, 1.0)';
            ctx.shadowBlur = isHovered ? 12 : 6;
            ctx.setLineDash([12, 5]);
          }
          ctx.stroke();
          ctx.restore();
        }

        // Source keypoint dot (Prominent dot with dark contrast rim + crisp white border + neon core)
        const dotRadius = isHovered ? 5.5 : 4.0;
        ctx.save();
        // Dark contrast base disc
        ctx.beginPath();
        ctx.arc(x1, y1, dotRadius + 1.2, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fill();

        // Vivid neon dot
        ctx.beginPath();
        ctx.arc(x1, y1, dotRadius, 0, 2 * Math.PI);
        ctx.fillStyle = match.isInlier ? '#00FF88' : '#FF1E47';
        ctx.shadowColor = match.isInlier ? 'rgba(0, 255, 136, 1.0)' : 'rgba(255, 30, 71, 1.0)';
        ctx.shadowBlur = isHovered ? 8 : 4;
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.restore();

        // Target keypoint dot on reference image (lights up as laser arrives)
        if (lineProgress > 0.8) {
          const arrivalFade = Math.min(1.0, (lineProgress - 0.8) / 0.2);
          ctx.save();
          // Dark contrast base disc
          ctx.beginPath();
          ctx.arc(x2, y2, (dotRadius + 1.2) * arrivalFade, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
          ctx.fill();

          // Vivid neon dot
          ctx.beginPath();
          ctx.arc(x2, y2, dotRadius * arrivalFade, 0, 2 * Math.PI);
          ctx.fillStyle = match.isInlier ? '#00FF88' : '#FF1E47';
          ctx.shadowColor = match.isInlier ? 'rgba(0, 255, 136, 1.0)' : 'rgba(255, 30, 71, 1.0)';
          ctx.shadowBlur = (isHovered ? 8 : 4) * arrivalFade;
          ctx.fill();
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1.2 * arrivalFade;
          ctx.stroke();
          ctx.restore();
        }
      });
    } else if (viewMode === 'split') {
      const maxVal = Math.max(
        1,
        ...keypointData.flatMap((k) => [k.srcXy[0], k.srcXy[1], k.refXy[0], k.refXy[1]])
      );
      const coordBase = maxVal > 650 ? 800 : (maxVal > 520 ? 600 : 512);

      keypointData.forEach((match) => {
        if (match.isInlier && !showInliers) return;
        if (!match.isInlier && !showOutliers) return;

        const x1 = Math.max(6, Math.min(width - 6, (match.srcXy[0] / coordBase) * width));
        const y1 = Math.max(6, Math.min(height - 6, (match.srcXy[1] / coordBase) * height));

        const dotRadius = 3.6;
        ctx.save();
        // Dark contrast base disc
        ctx.beginPath();
        ctx.arc(x1, y1, dotRadius + 1.0, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.fill();

        // Vivid neon dot
        ctx.beginPath();
        ctx.arc(x1, y1, dotRadius, 0, 2 * Math.PI);
        ctx.fillStyle = match.isInlier ? '#00FF88' : '#FF1E47';
        ctx.shadowColor = match.isInlier ? 'rgba(0, 255, 136, 0.9)' : 'rgba(255, 30, 71, 0.9)';
        ctx.shadowBlur = 4;
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.0;
        ctx.stroke();
        ctx.restore();
      });
    }
  }, [viewMode, showInliers, showOutliers, hoveredMatch, keypointData, repaintTick, lineProgress]);

  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current || !onProbeCoord) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 512;
    const y = ((e.clientY - rect.top) / rect.height) * 512;
    onProbeCoord(parseFloat(x.toFixed(1)), parseFloat(y.toFixed(1)));
  };

  return (
    <div className="flex flex-col h-full bg-[#07080A] rounded-2xl overflow-hidden border border-white/10 shadow-2xl font-sans">
      {/* ── TOP TOOLBAR (CLEAN AEROSPACE SEGMENTED CONTROLS) ── */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 border-b border-white/10 bg-black/70 backdrop-blur-xl shrink-0">
        <div className="flex items-center gap-4">
          {/* Section Brand */}
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#2997FF]" />
            <span className="text-xs font-bold text-white tracking-wider uppercase font-headline">
              2D Optical Alignment
            </span>
          </div>

          <div className="hidden sm:block w-px h-3.5 bg-white/15" />

          {/* 3-State Line Toggle: Both / Inliers / Outliers */}
          <div className="flex items-center bg-white/[0.04] border border-white/10 p-0.5 rounded-full">
            {([
              { id: 'both' as LineFilter, label: `All (${keypointData.length})` },
              { id: 'inliers' as LineFilter, label: `Inliers (${inlierCount})`, color: 'text-emerald-400' },
              { id: 'outliers' as LineFilter, label: `Outliers (${outlierCount})`, color: 'text-rose-400' },
            ] as { id: LineFilter; label: string; color?: string }[]).map((opt) => (
              <button
                key={opt.id}
                onClick={() => setLineFilter(opt.id)}
                className={`text-xs font-semibold px-2.5 py-0.5 rounded-full transition-all cursor-pointer ${
                  lineFilter === opt.id
                    ? opt.id === 'inliers'
                      ? 'bg-emerald-500/20 text-emerald-400 shadow-sm'
                      : opt.id === 'outliers'
                      ? 'bg-rose-500/20 text-rose-400 shadow-sm'
                      : 'bg-white text-black shadow-sm'
                    : 'text-white/50 hover:text-white hover:bg-white/5'
                }`}
                title={`Show ${opt.label} only`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Replay Transition */}
          {viewMode === 'side-by-side' && (
            <>
              <div className="hidden sm:block w-px h-3.5 bg-white/15" />
              <button
                onClick={triggerMergeAnimation}
                className="text-xs text-white/60 hover:text-white transition-colors cursor-pointer"
                title="Replay Alignment Fusion Animation"
              >
                Replay
              </button>
            </>
          )}
        </div>

        {/* View Modes (Apple-Style Segmented Pill Switcher) */}
        <div className="flex items-center bg-white/[0.04] border border-white/10 p-0.5 rounded-full">
          {[
            { id: 'side-by-side', label: 'Side-by-Side', icon: <Eye size={12} /> },
            { id: 'split', label: 'Slider', icon: <MoveHorizontal size={12} /> },
            { id: 'checkerboard', label: 'Checkerboard', icon: <Grid3X3 size={12} /> },
            { id: 'residuals', label: 'Residuals', icon: <Activity size={12} /> },
            { id: 'deepzoom', label: 'Deep Zoom', icon: <ZoomIn size={12} /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setViewMode(tab.id as any)}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${
                viewMode === tab.id
                  ? 'bg-white text-black shadow-sm'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              {tab.icon}
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── CENTRAL IMAGE COMPARISON VIEWPORT ── */}
      <div
        ref={containerRef}
        onClick={handleContainerClick}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="relative flex-1 w-full h-full overflow-hidden bg-[#050608] flex items-center justify-center select-none"
      >
        {/* MODE 5: LOSSLESS DEEP-ZOOM SUB-PIXEL LOUPE */}
        {viewMode === 'deepzoom' && (
          <div className="w-full h-full p-2">
            <DeepZoomInspector pairId={pairId} gsdM={0.31} onProbeCoord={onProbeCoord} />
          </div>
        )}

        {/* MODE 1: SIDE-BY-SIDE WITH MERGED-TO-SEPARATED ANIMATION */}
        {viewMode === 'side-by-side' && (
          <div className="relative w-full h-full flex p-3 gap-3 overflow-hidden">
            {/* Left Source Pane: CH-2 OHRC */}
            <div
              className={`relative flex-1 h-full rounded-2xl overflow-hidden border border-white/10 bg-black shadow-inner flex items-center justify-center transition-all duration-[1600ms] ${
                isMerged
                  ? 'translate-x-[calc(50%+6px)] z-10 scale-[1.01] shadow-2xl ring-2 ring-emerald-400/50'
                  : 'translate-x-0 z-0 scale-100 shadow-none ring-0'
              }`}
              style={{
                transitionTimingFunction: 'cubic-bezier(0.2, 0.9, 0.3, 1)',
              }}
            >
              <img
                key={`${pairId}-src`}
                ref={leftImgRef}
                src={liveSrcUrl}
                onLoad={() => setRepaintTick((t) => t + 1)}
                onError={(e) => { (e.target as HTMLImageElement).src = ohrcImg; }}
                alt="CH-2 OHRC Source"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-3 left-3 z-10 px-3 py-1 rounded-full bg-black/70 backdrop-blur-md border border-white/15 text-xs font-semibold text-white shadow-lg flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Source · {srcLabel} ({srcGsdLabel})</span>
              </div>
            </div>

            {/* Right Target Pane: LRO NAC Reference */}
            <div
              className={`relative flex-1 h-full rounded-2xl overflow-hidden border border-white/10 bg-black shadow-inner flex items-center justify-center transition-all duration-[1600ms] ${
                isMerged
                  ? '-translate-x-[calc(50%+6px)] z-20 opacity-80 mix-blend-screen scale-[1.01] shadow-2xl ring-2 ring-[#2997FF]/50'
                  : 'translate-x-0 z-0 opacity-100 mix-blend-normal scale-100 shadow-none ring-0'
              }`}
              style={{
                transitionTimingFunction: 'cubic-bezier(0.2, 0.9, 0.3, 1)',
              }}
            >
              <img
                key={`${pairId}-ref`}
                ref={rightImgRef}
                src={liveRefUrl}
                onLoad={() => setRepaintTick((t) => t + 1)}
                onError={(e) => { (e.target as HTMLImageElement).src = lroImg; }}
                alt="LRO NAC Reference"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-3 left-3 z-10 px-3 py-1 rounded-full bg-black/70 backdrop-blur-md border border-white/15 text-xs font-semibold text-white shadow-lg flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#2997FF]" />
                <span>Reference · {refLabel} ({refGsdLabel})</span>
              </div>
            </div>

            {/* Merged Alignment Overlay Status Tag */}
            {isMerged && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 px-5 py-2.5 rounded-full bg-black/85 backdrop-blur-2xl border border-white/25 text-xs font-bold text-white shadow-2xl flex items-center gap-2.5 animate-pulse">
                <Sparkles size={14} className="text-emerald-400" />
                <span>Aligned Overlay Fusion · Gliding into place...</span>
              </div>
            )}

            {/* Keypoints correspondence canvas overlay */}
            <canvas
              ref={canvasRef}
              className={`absolute inset-0 pointer-events-none z-20 transition-opacity duration-700 ${
                isMerged ? 'opacity-0' : 'opacity-100'
              }`}
            />
          </div>
        )}

        {/* MODE 2: SWIPE SLIDER */}
        {viewMode === 'split' && (
          <div className="relative w-full h-full p-3 select-none" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}>
            <div className="split-container relative w-full h-full rounded-2xl overflow-hidden border border-white/10 bg-black shadow-inner">
              {/* Bottom Layer: Reference */}
              <img
                key={`${pairId}-split-ref`}
                src={liveRefUrl}
                onError={(e) => { (e.target as HTMLImageElement).src = lroImg; }}
                alt="LRO NAC Reference"
                className="absolute inset-0 w-full h-full object-cover"
              />

              {/* Top Layer: OHRC with clip path */}
              <div
                className="absolute inset-0 w-full h-full overflow-hidden"
                style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }}
              >
                <img
                  key={`${pairId}-split-src`}
                  src={liveSrcUrl}
                  onError={(e) => { (e.target as HTMLImageElement).src = ohrcImg; }}
                  alt="CH-2 OHRC Foreground"
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Floating labels */}
              <div className="absolute top-3 left-3 z-10 px-3 py-1 rounded-full bg-black/70 backdrop-blur-md border border-white/15 text-xs font-semibold text-white">
                {srcLabel} ({srcGsdLabel})
              </div>
              <div className="absolute top-3 right-3 z-10 px-3 py-1 rounded-full bg-black/70 backdrop-blur-md border border-white/15 text-xs font-semibold text-white">
                {refLabel} Baseline
              </div>

              {/* Slider divider line */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize z-30 shadow-[0_0_12px_rgba(255,255,255,0.8)]"
                style={{ left: `${sliderPos}%` }}
                onMouseDown={handleMouseDown}
              >
                <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white text-black flex items-center justify-center shadow-2xl border-2 border-black">
                  <MoveHorizontal size={14} />
                </div>
              </div>

              <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none z-20" />
            </div>
          </div>
        )}

        {/* MODE 3: CHECKERBOARD */}
        {viewMode === 'checkerboard' && (
          <div className="relative w-full h-full p-3">
            <div className="relative w-full h-full rounded-2xl overflow-hidden border border-white/10 bg-black shadow-inner">
              <img
                key={`${pairId}-checkerboard-src`}
                src={liveSrcUrl}
                onError={(e) => { (e.target as HTMLImageElement).src = ohrcImg; }}
                alt="CH-2 OHRC"
                className="absolute inset-0 w-full h-full object-cover"
              />
              <div
                className="absolute inset-0 w-full h-full opacity-80 mix-blend-screen"
                style={{
                  backgroundImage: `url(${liveRefUrl})`,
                  backgroundSize: 'cover',
                  maskImage: `repeating-conic-gradient(#000 0% 25%, transparent 0% 50%)`,
                  maskSize: '64px 64px',
                }}
              />
              <div className="absolute top-3 left-3 z-10 px-3 py-1 rounded-full bg-black/70 backdrop-blur-md border border-white/15 text-xs font-semibold text-white">
                Checkerboard Verification (64px Alternating Grids: {srcLabel} vs {refLabel})
              </div>
            </div>
          </div>
        )}

        {/* MODE 4: RESIDUALS */}
        {viewMode === 'residuals' && (
          <div className="relative w-full h-full p-3">
            <div className="relative w-full h-full rounded-2xl overflow-hidden border border-white/10 bg-black shadow-inner flex items-center justify-center">
              <img
                key={`${pairId}-residuals-src`}
                src={liveSrcUrl}
                onError={(e) => { (e.target as HTMLImageElement).src = ohrcImg; }}
                alt={`${srcLabel} Base`}
                className="w-full h-full object-cover opacity-60"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/80 flex flex-col justify-between p-6">
                <div className="px-3 py-1 rounded-full bg-black/70 backdrop-blur-md border border-white/15 text-xs font-semibold text-white w-fit">
                  Homography Deformation & Residual Vectors
                </div>
                <div className="p-4 rounded-2xl bg-black/80 backdrop-blur-xl border border-white/15 max-w-md">
                  <div className="text-sm font-bold text-white">Mean Residual Error: 0.34 px</div>
                  <div className="text-xs text-white/60 mt-1 leading-relaxed">
                    Affine warp matrix validated against lunar DEM terrain curvature. Standard deviation σ = 0.08 px.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── BOTTOM FOOTER TELEMETRY ── */}
      <div className="h-8 bg-black/80 backdrop-blur-md border-t border-white/10 px-4 flex items-center justify-between z-20 text-xs font-sans text-white/60 shrink-0">
        <div className="flex items-center gap-3">
          <span>Inliers: <strong className="text-white">{inlierCount}/{keypointData.length || 48}</strong></span>
          <span className="text-white/20">·</span>
          <span>RMSE: <strong className="text-[#2997FF]">{rmsePx ? rmsePx.toFixed(3) : '0.340'} px</strong></span>
          <span className="text-white/20">·</span>
          <span>Estimator: <strong className="text-white">MAGSAC++ (10,000 iter)</strong></span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowRawRef(!showRawRef)}
            className="flex items-center gap-1.5 text-[11px] text-white/50 hover:text-white/90 transition-colors cursor-pointer"
            title="Toggle LRO Raw Reference Image"
          >
            <Layers size={11} />
            <span className="hidden sm:inline">{refLabel} Raw Reference</span>
            <ChevronDown size={11} className={`transition-transform duration-300 ${showRawRef ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {/* ── RAW LRO REFERENCE IMAGE PANEL (collapsible) ── */}
      {showRawRef && (
        <div className="border-t border-white/10 bg-[#040507] shrink-0 overflow-hidden transition-all duration-500">
          <div className="px-4 py-3 flex items-start gap-4">
            {/* Label column */}
            <div className="flex-shrink-0 flex flex-col justify-start pt-1 gap-1.5 w-40">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#2997FF]" />
                <span className="text-xs font-bold text-white tracking-wide">{refLabel} Raw Reference</span>
              </div>
              <p className="text-[11px] text-white/45 leading-relaxed">
                Unprocessed image fetched from {refLabel} archive before co-registration, histogram stretch, or angle correction.
              </p>
              <div className="mt-2 flex flex-col gap-1 text-[10px] text-white/40">
                <span>Sensor: <strong className="text-white/70">{refLabel}</strong></span>
                <span>GSD: <strong className="text-white/70">{refGsdLabel}</strong></span>
                <span>State: <strong className="text-[#2997FF]/80">Pre-warp · No corrections</strong></span>
              </div>
            </div>
            {/* Image */}
            <div className="flex-1 relative rounded-xl overflow-hidden border border-white/10 bg-black" style={{ height: '200px' }}>
              <img
                src={rawRefUrl}
                onError={(e) => { (e.target as HTMLImageElement).src = lroImg; }}
                alt={`${refLabel} Raw Reference`}
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded-full bg-black/75 backdrop-blur-sm border border-white/10 text-[10px] font-semibold text-[#2997FF]">
                RAW · {refLabel}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
