import React, { useState, useEffect } from 'react';
import {
  Plus, Minus, RotateCcw, Globe, GitCommit, FlaskConical,
  ChevronDown, Play, Check, Sliders, Upload,
  RefreshCw, CheckCircle2, ArrowRight, AlertTriangle, X
} from 'lucide-react';
import {
  PipelineStage,
  type ScenePreset,
  type LayerVisibility,
  type PipelineOptions,
  type ActiveProcessingState
} from '../types';
import { MapLayerControl } from './MapLayerControl';

export interface ProbedLocation {
  lat: number;
  lon: number;
  name: string;
  region: string;
  elevationM: number;
  solarIncidence: number;
  temperatureK: number;
  waterIceWtPct: number;
  psrStatus: string;
  craterId?: string;
}

const VisualProgressBar: React.FC<{
  status: 'idle' | 'processing' | 'completed' | 'error';
}> = ({ status }) => {
  const [progress, setProgress] = useState(14);

  useEffect(() => {
    if (status === 'processing') {
      setProgress(16);
      const timer = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 93) return prev;
          const remaining = 93 - prev;
          const step = Math.max(1, Math.floor(remaining * 0.1));
          return Math.min(93, prev + step);
        });
      }, 350);
      return () => clearInterval(timer);
    } else if (status === 'completed') {
      setProgress(100);
    } else {
      setProgress(0);
    }
  }, [status]);

  if (status !== 'processing' && status !== 'completed') return null;

  const isDone = status === 'completed';

  return (
    <div className="w-full flex flex-col gap-1.5 mt-1">
      <div className="flex items-center justify-between text-[10px] font-mono tracking-wide">
        <span className="text-white/60">
          {isDone ? 'Pipeline Complete' : 'Processing Progress'}
        </span>
        <span className={`font-bold transition-colors duration-300 ${isDone ? 'text-emerald-400' : 'text-[#2997FF]'}`}>
          {Math.round(progress)}%
        </span>
      </div>

      <div className="relative w-full h-2 bg-black/50 rounded-full overflow-hidden p-0.5 border border-white/10 shadow-inner">
        <div
          className={`h-full rounded-full transition-all duration-300 ease-out relative ${
            isDone
              ? 'bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-300 shadow-[0_0_12px_rgba(52,211,153,0.8)]'
              : 'bg-gradient-to-r from-[#0071E3] via-[#2997FF] to-[#64D2FF] shadow-[0_0_12px_rgba(41,151,255,0.8)]'
          }`}
          style={{ width: `${progress}%` }}
        >
          {/* Leading edge glow bead */}
          <div className="absolute right-0 top-0 bottom-0 w-1.5 rounded-full bg-white/70 shadow-[0_0_6px_rgba(255,255,255,0.9)]" />
        </div>
      </div>
    </div>
  );
};

interface MoonWorkbenchOverlayProps {
  selectedScene: ScenePreset;
  onSelectScene?: (scene: ScenePreset) => void;
  layers: LayerVisibility;
  onLayerChange?: (layers: LayerVisibility) => void;
  options: PipelineOptions;
  onOptionsChange: (options: PipelineOptions) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetView: () => void;
  activeTab: '3d' | '2d' | 'results';
  onTabChange: (tab: '3d' | '2d' | 'results') => void;
  onBackToLanding: () => void;
  onOpenTargetPalette: () => void;
  onOpenEngineInspector: () => void;
  onOpenAddFiles?: () => void;
  pipelineStage: PipelineStage;
  onRunPipeline: () => void;
  isBackendOnline?: boolean;
  processingState?: ActiveProcessingState | null;
  onCompleteProcessing?: () => void;
  onDismissProcessing?: () => void;
}

export const MoonWorkbenchOverlay: React.FC<MoonWorkbenchOverlayProps> = ({
  selectedScene,
  layers,
  onLayerChange,
  options,
  onOptionsChange,
  onZoomIn,
  onZoomOut,
  onResetView,
  activeTab,
  onTabChange,
  onBackToLanding,
  onOpenTargetPalette,
  onOpenEngineInspector,
  onOpenAddFiles,
  pipelineStage,
  onRunPipeline,
  isBackendOnline = true,
  processingState,
  onCompleteProcessing,
  onDismissProcessing,
}) => {
  const isRunning = pipelineStage !== PipelineStage.Idle && pipelineStage !== PipelineStage.Done;
  const isDone = pipelineStage === PipelineStage.Done;

  return (
    <div className="absolute inset-0 w-full h-full pointer-events-none overflow-hidden select-none font-sans z-20">
      {/* ── 1. TOP-LEFT: BRAND & CRATER SELECTOR + ADD FILES (BESIDE) ── */}
      <div className="absolute top-4 left-4 sm:top-5 sm:left-5 pointer-events-auto flex items-center gap-2">
        <div className="flex items-center gap-2 sm:gap-2.5 bg-black/60 hover:bg-black/80 backdrop-blur-xl border border-white/15 px-3.5 py-1.5 sm:px-4 sm:py-2 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.6)] transition-all">
          {/* Brand: Voyage (Click goes to Home) */}
          <button
            onClick={onBackToLanding}
            className="flex items-center gap-2 group cursor-pointer focus:outline-none"
            title="Return to Home Overview"
          >
            <span className="font-logo text-base sm:text-lg font-extrabold tracking-[0.08em] text-white group-hover:text-[#2997FF] transition-colors">
              Voyage
            </span>
            <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-[#0F1117] via-slate-300 to-white shadow-[0_0_8px_rgba(255,255,255,0.35)] relative overflow-hidden flex items-center justify-center shrink-0 border border-white/30">
              <div className="absolute inset-0 bg-black/75 rounded-full translate-x-1 -translate-y-0.5" />
            </div>
          </button>

          <div className="w-px h-4 bg-white/20" />

          {/* Current Crater Selector (Opens Target Palette) */}
          <button
            onClick={onOpenTargetPalette}
            className="flex items-center gap-2 text-white/80 hover:text-white transition-colors cursor-pointer group"
            title="Open Lunar Target Palette"
          >
            <span className="w-2 h-2 rounded-full bg-[#2997FF] shadow-[0_0_8px_rgba(41,151,255,0.8)]" />
            <span className="text-xs font-semibold max-w-[105px] sm:max-w-[160px] md:max-w-none truncate">
              {selectedScene.name}
            </span>
            <ChevronDown size={13} className="text-white/40 group-hover:text-white transition-transform group-hover:translate-y-0.5" />
          </button>

          {/* Add Files Button (Right beside crater selector) */}
          {onOpenAddFiles && (
            <>
              <div className="w-px h-4 bg-white/20" />
              <button
                onClick={onOpenAddFiles}
                className="flex items-center gap-1.5 text-white/75 hover:text-white transition-all cursor-pointer group active:scale-95 text-xs font-semibold"
                title="Add Lunar Mission Imagery / PDS-4 Files"
              >
                <Upload size={12} className="text-[#2997FF] group-hover:scale-110 transition-transform" />
                <span>Add Files</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* ── 2. TOP-RIGHT: WORKBENCH MODE SWITCHER & ENGINE INSPECTOR ── */}
      <div className="absolute top-4 right-4 sm:top-5 sm:right-5 pointer-events-auto flex items-center bg-black/60 hover:bg-black/80 backdrop-blur-xl border border-white/15 p-1 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.6)] transition-all">
        <button
          onClick={() => onTabChange('3d')}
          className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-full font-semibold transition-all cursor-pointer text-xs ${
            activeTab === '3d'
              ? 'bg-white text-black shadow-md'
              : 'text-white/65 hover:text-white hover:bg-white/10'
          }`}
        >
          <Globe size={13} />
          <span>3D Globe</span>
        </button>

        <button
          onClick={() => onTabChange('2d')}
          className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-full font-semibold transition-all cursor-pointer text-xs ${
            activeTab === '2d'
              ? 'bg-white text-black shadow-md'
              : 'text-white/65 hover:text-white hover:bg-white/10'
          }`}
        >
          <GitCommit size={13} />
          <span>2D Alignment</span>
        </button>

        <button
          onClick={() => onTabChange('results')}
          className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-full font-semibold transition-all cursor-pointer text-xs ${
            activeTab === 'results'
              ? 'bg-white text-black shadow-md'
              : 'text-white/65 hover:text-white hover:bg-white/10'
          }`}
        >
          <FlaskConical size={13} />
          <span>Findings</span>
        </button>

        <div className="w-px h-3.5 bg-white/20 mx-1" />

        {/* Engine Inspector Trigger Button */}
        <button
          onClick={onOpenEngineInspector}
          className="flex items-center gap-1.5 px-3 sm:px-3.5 py-1.5 rounded-full font-semibold transition-all cursor-pointer text-xs text-white/65 hover:text-white hover:bg-white/10"
          title="Open Registration Engine & Dataset Inspector"
        >
          <Sliders size={13} />
          <span className="hidden sm:inline">Engine</span>
        </button>
      </div>



      {/* ── 3. BOTTOM-CENTER: PRIMARY ACTION BUTTON (STANDALONE RUN CO-REGISTRATION) ── */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 pointer-events-auto flex items-center bg-black/75 hover:bg-black/85 backdrop-blur-2xl border border-white/15 p-1.5 sm:p-2 rounded-full shadow-[0_12px_40px_rgba(0,0,0,0.8)] transition-all">
        {/* Primary Action Button: Run Co-Registration */}
        <button
          onClick={onRunPipeline}
          disabled={isRunning}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-full font-bold text-xs tracking-wide uppercase transition-all cursor-pointer active:scale-95 shadow-md ${
            isDone
              ? 'bg-white text-black shadow-lg hover:bg-white/90'
              : isRunning
              ? 'bg-white/10 text-white/40 cursor-not-allowed'
              : 'bg-[#0071E3] hover:bg-[#0077ED] text-white shadow-[0_0_20px_rgba(0,113,227,0.5)]'
          }`}
        >
          {isDone ? (
            <Check size={14} strokeWidth={3} className="text-emerald-600" />
          ) : (
            <Play size={14} className={isRunning ? 'animate-spin' : 'fill-current'} />
          )}
          <span>{isDone ? 'Co-Registered' : isRunning ? 'Aligning...' : 'Run Co-Registration'}</span>
        </button>
      </div>

      {/* ── 3b. MIDDLE-RIGHT: PIPELINE LIVE STATUS READOUT ── */}
      <div className="absolute right-5 top-1/2 -translate-y-1/2 pointer-events-auto z-20 flex flex-col items-end gap-2.5">
        <div className="flex items-center gap-2 bg-black/75 hover:bg-black/85 backdrop-blur-2xl border border-white/15 px-3.5 py-2 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.6)] font-mono text-xs text-white/80 transition-all">
          <span className={`w-2 h-2 rounded-full ${isBackendOnline ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)] animate-pulse' : 'bg-amber-400'}`} />
          <span>Pipeline: <strong className="text-white">{isDone ? 'Registered' : isRunning ? 'Processing' : 'Ready'}</strong></span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-emerald-300 font-semibold tracking-wider">
            {isBackendOnline ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>

        {/* Processing readout card below Pipeline Ready */}
        {processingState && processingState.status !== 'idle' && (
          <div className="w-80 bg-[#0B0D13]/92 hover:bg-[#0B0D13]/98 backdrop-blur-2xl border border-white/20 rounded-2xl p-4 shadow-[0_16px_40px_rgba(0,0,0,0.85)] flex flex-col gap-2.5 animate-in fade-in slide-in-from-right-3 duration-200 transition-all">
            {/* Top row: Status header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {processingState.status === 'processing' && (
                  <RefreshCw size={14} className="animate-spin text-[#2997FF]" />
                )}
                {processingState.status === 'completed' && (
                  <CheckCircle2 size={16} className="text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                )}
                {processingState.status === 'error' && (
                  <AlertTriangle size={15} className="text-red-400" />
                )}
                <span className="text-xs font-bold text-white tracking-wide">
                  {processingState.status === 'processing' && 'Processing Files...'}
                  {processingState.status === 'completed' && 'Processing Complete'}
                  {processingState.status === 'error' && 'Processing Error'}
                </span>
              </div>
              {onDismissProcessing && (
                <button
                  onClick={onDismissProcessing}
                  className="p-1 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
                  title="Dismiss"
                >
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Middle: Pair Details */}
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-white/90 truncate max-w-[190px]">
                  {processingState.pairName || 'Mission Pair'}
                </span>
                {processingState.fileCount !== undefined && processingState.fileCount > 0 && (
                  <span className="font-mono text-[10px] text-[#2997FF] bg-[#0071E3]/20 px-2 py-0.5 rounded-full border border-[#2997FF]/30">
                    {processingState.fileCount} {processingState.fileCount === 1 ? 'file' : 'files'}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-white/60">
                {processingState.errorMessage || processingState.stageMessage || 'Analyzing lunar telemetry...'}
              </p>
            </div>

            {/* Visual Dynamic Progress Bar */}
            <VisualProgressBar status={processingState.status} />

            {/* Complete Processing Action Button */}
            {processingState.status === 'completed' && onCompleteProcessing && (
              <button
                onClick={onCompleteProcessing}
                className="mt-1 w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-[#0071E3] to-[#2997FF] hover:from-[#0077ED] hover:to-[#409CFF] text-white font-bold text-xs tracking-wider uppercase flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(0,113,227,0.6)] hover:shadow-[0_0_28px_rgba(41,151,255,0.9)] transition-all cursor-pointer active:scale-95 group border border-white/20"
              >
                <span>Complete Processing</span>
                <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </button>
            )}

            {/* Error Dismiss Button */}
            {processingState.status === 'error' && onDismissProcessing && (
              <button
                onClick={onDismissProcessing}
                className="mt-1 w-full py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white/80 font-semibold text-xs transition-colors cursor-pointer"
              >
                Dismiss
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── 4. BOTTOM-LEFT: FLOATING LAYER CONTROL CAPSULE ── */}
      <div className="absolute bottom-5 left-5 pointer-events-auto">
        <MapLayerControl
          layers={layers}
          onLayerChange={onLayerChange || (() => {})}
          options={options}
          onOptionsChange={onOptionsChange}
        />
      </div>

      {/* ── 5. BOTTOM-RIGHT: CAMERA CONTROLS ONLY ── */}
      <div className="absolute bottom-5 right-5 pointer-events-auto flex items-center gap-1 bg-black/60 hover:bg-black/80 backdrop-blur-xl border border-white/15 p-1 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.6)] transition-all">
        <button
          onClick={onZoomIn}
          title="Zoom In"
          className="p-2 rounded-full text-white/70 hover:text-white hover:bg-white/15 transition-colors cursor-pointer active:scale-95"
        >
          <Plus size={14} />
        </button>
        <button
          onClick={onZoomOut}
          title="Zoom Out"
          className="p-2 rounded-full text-white/70 hover:text-white hover:bg-white/15 transition-colors cursor-pointer active:scale-95"
        >
          <Minus size={14} />
        </button>
        <div className="w-px h-3.5 bg-white/15 mx-0.5" />
        <button
          onClick={onResetView}
          title="Reset Camera View"
          className="p-2 rounded-full text-white/70 hover:text-white hover:bg-white/15 transition-colors cursor-pointer active:scale-95"
        >
          <RotateCcw size={13} />
        </button>
      </div>
    </div>
  );
};
