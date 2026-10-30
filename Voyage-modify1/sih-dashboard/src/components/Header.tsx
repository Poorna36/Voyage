import React from 'react';
import {
  Globe, GitCommit, FlaskConical, Play,
  Wifi, WifiOff, RefreshCw, PanelLeft, ArrowLeft
} from 'lucide-react';
import { PipelineStage, PIPELINE_STAGE_LABELS } from '../types';

interface HeaderProps {
  activeStage: PipelineStage;
  activeTab: '3d' | '2d' | 'results';
  onTabChange: (tab: '3d' | '2d' | 'results') => void;
  selectedScene?: string;
  isBackendOnline?: boolean;
  backendLatencyMs?: number | null;
  onRefreshBackend?: () => void;
  onBackToLanding?: () => void;
  isPanelOpen: boolean;
  onTogglePanel: () => void;
  onRunPipeline?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeStage,
  activeTab,
  onTabChange,
  isBackendOnline = false,
  backendLatencyMs = null,
  onRefreshBackend,
  onBackToLanding,
  isPanelOpen,
  onTogglePanel,
  onRunPipeline,
}) => {
  const isRunning = activeStage !== PipelineStage.Idle && activeStage !== PipelineStage.Done;
  const isDone = activeStage === PipelineStage.Done;

  return (
    <header className="h-14 bg-black/85 backdrop-blur-md border-b border-white/10 px-4 sm:px-6 flex items-center justify-between z-30 select-none text-white shrink-0 font-sans">
      {/* ── LEFT: BRANDING + BACK LINK + PANEL TOGGLE (ROUND BUTTONS) ── */}
      <div className="flex items-center gap-2.5 sm:gap-3">
        <div className="flex items-center gap-2 pr-1">
          <span className="font-logo text-lg sm:text-xl font-extrabold tracking-[0.08em] text-white">
            Voyage
          </span>
          <div className="w-3.5 h-3.5 rounded-full bg-gradient-to-tr from-[#0F1117] via-slate-300 to-white shadow-[0_0_8px_rgba(255,255,255,0.35)] relative overflow-hidden flex items-center justify-center shrink-0 border border-white/30">
            <div className="absolute inset-0 bg-black/75 rounded-full translate-x-1 -translate-y-0.5" />
          </div>
        </div>

        {onBackToLanding && (
          <button
            onClick={onBackToLanding}
            className="flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white text-xs font-semibold transition-all cursor-pointer active:scale-95 border border-white/15"
            title="Return to Public Overview"
          >
            <ArrowLeft size={13} />
            <span className="hidden sm:inline">Landing</span>
          </button>
        )}

        <button
          onClick={onTogglePanel}
          className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer border active:scale-95 ${
            isPanelOpen
              ? 'bg-[#0071E3] border-[#0071E3] text-white shadow-sm'
              : 'bg-white/10 hover:bg-white/15 border-white/15 text-white'
          }`}
          title="Toggle Mission & Data Panel"
        >
          <PanelLeft size={14} />
          <span className="hidden sm:inline">Data Panel</span>
        </button>
      </div>

      {/* ── CENTER: ROUND SEGMENTED TABS (NO SHARP EDGES) ── */}
      <div className="flex items-center bg-white/5 border border-white/10 rounded-full p-1 shadow-inner">
        <button
          onClick={() => onTabChange('3d')}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 cursor-pointer ${
            activeTab === '3d'
              ? 'bg-white text-black shadow-sm'
              : 'text-white/70 hover:text-white hover:bg-white/5'
          }`}
        >
          <Globe size={13} />
          <span>3D Globe</span>
        </button>

        <button
          onClick={() => onTabChange('2d')}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 cursor-pointer ${
            activeTab === '2d'
              ? 'bg-white text-black shadow-sm'
              : 'text-white/70 hover:text-white hover:bg-white/5'
          }`}
        >
          <GitCommit size={13} />
          <span className="hidden sm:inline">2D Alignment</span>
          <span className="sm:hidden">2D</span>
        </button>

        <button
          onClick={() => onTabChange('results')}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 cursor-pointer ${
            activeTab === 'results'
              ? 'bg-white text-black shadow-sm'
              : 'text-white/70 hover:text-white hover:bg-white/5'
          }`}
        >
          <FlaskConical size={13} />
          <span className="hidden sm:inline">Diagnostics</span>
          <span className="sm:hidden">Results</span>
        </button>
      </div>

      {/* ── RIGHT: STATUS BADGE + ROUND ACTION BUTTON ── */}
      <div className="flex items-center gap-3">
        {/* Backend Connectivity Status */}
        <div className="hidden lg:flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[11px] font-sans text-white/80">
          {isBackendOnline ? (
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <Wifi size={12} className="animate-pulse" />
              <span>ONLINE</span>
              {backendLatencyMs && <span className="text-[10px] text-white/50 font-mono">({backendLatencyMs}ms)</span>}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-amber-400 font-medium">
              <WifiOff size={12} />
              <span>OFFLINE</span>
            </span>
          )}

          {onRefreshBackend && (
            <button
              onClick={onRefreshBackend}
              title="Ping Backend API"
              className="text-white/40 hover:text-white transition-colors cursor-pointer ml-1"
            >
              <RefreshCw size={11} />
            </button>
          )}
        </div>

        {/* Pipeline Execution Button (Round Pill Button) */}
        {onRunPipeline && (
          <button
            onClick={onRunPipeline}
            disabled={isRunning}
            className={`flex items-center gap-2 px-5 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer shadow-md active:scale-95 ${
              isDone
                ? 'bg-emerald-500 hover:bg-emerald-400 text-white shadow-[0_0_16px_rgba(16,185,129,0.3)]'
                : isRunning
                ? 'bg-white/10 text-white/50 border border-white/15 cursor-not-allowed'
                : 'bg-[#0071E3] hover:bg-[#0077ED] text-white shadow-[0_2px_14px_rgba(0,113,227,0.35)]'
            }`}
          >
            <Play size={12} className={isRunning ? 'animate-spin' : 'fill-current'} />
            <span>{isDone ? 'Registered (Done)' : isRunning ? PIPELINE_STAGE_LABELS[activeStage] : 'Run Co-Registration'}</span>
          </button>
        )}
      </div>
    </header>
  );
};
