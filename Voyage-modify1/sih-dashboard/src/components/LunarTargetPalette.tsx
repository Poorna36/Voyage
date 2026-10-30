import React, { useState, useEffect, useRef } from 'react';
import { Search, X, ChevronLeft, ChevronRight, Compass } from 'lucide-react';
import type { ScenePreset, CraterDetail } from '../types';
import { getCraterCatalog, API_BASE } from '../services/api';
import ohrcThumb from '../assets/images/ohrc_orbital_fallback.jpg';
import iirsThumb from '../assets/images/iirs_hyperspectral_overlay_1788336834453.jpg';
import tmc2Thumb from '../assets/images/tmc2_terrain_context_1788336820221.jpg';
import lroThumb from '../assets/images/lro_reference_baseline_1788336850293.jpg';

interface LunarTargetPaletteProps {
  selectedScene: ScenePreset;
  onSelectScene: (scene: ScenePreset) => void;
  onClose: () => void;
  craters?: CraterDetail[];
}

const FALLBACK_THUMBNAILS: Record<string, string> = {
  boguslawsky: ohrcThumb,
  manzinus: lroThumb,
  shackleton: tmc2Thumb,
  cabeus: iirsThumb,
  clavius: ohrcThumb,
  tycho: lroThumb,
};

export const LunarTargetPalette: React.FC<LunarTargetPaletteProps> = ({
  selectedScene,
  onSelectScene,
  onClose,
  craters = [],
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<'all' | 'polar' | 'water' | 'highland'>('all');
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const targets = craters.map((c) => ({
    id: c.id,
    name: c.name,
    lat: c.lat,
    lon: c.lon,
    height: c.height,
    description: c.description,
    region: c.region,
    diameterKm: c.diameterKm ?? c.diameter_km ?? 50,
    depthKm: c.depthKm ?? c.depth_km ?? 2,
    waterAbsorptionDepth: c.waterAbsorptionDepthPct ?? c.water_absorption_depth_pct ?? 10.0,
    waterIceConcentration: c.waterIceConcentrationWtPct ?? c.water_ice_concentration_wt_pct ?? 3.5,
    solarIncidenceDeg: c.solarIncidenceDeg ?? c.solar_incidence_deg ?? 65,
    floorSlopeDeg: c.floorInclinationDeg ?? c.floor_inclination_deg ?? 4.5,
    psrStatus: c.psrStatus ?? c.psr_status ?? 'Micro-cold traps',
    thumbnail: `${API_BASE}/api/datasets/${c.id}/image/src`,
    fallbackThumbnail: FALLBACK_THUMBNAILS[c.id] || ohrcThumb,
  }));

  const filteredTargets = targets.filter((target) => {
    const q = searchQuery.toLowerCase().trim();
    const matches =
      !q ||
      target.name.toLowerCase().includes(q) ||
      target.region.toLowerCase().includes(q) ||
      `${target.lat}`.includes(q) ||
      `${target.lon}`.includes(q);

    if (!matches) return false;

    if (activeFilter === 'polar') return Math.abs(target.lat) > 65;
    if (activeFilter === 'water') return target.waterAbsorptionDepth > 8.0;
    if (activeFilter === 'highland') return target.region.toLowerCase().includes('highland');
    return true;
  });

  const handleSynthesizeCustomCrater = async () => {
    if (!searchQuery.trim()) return;
    try {
      const res = await getCraterCatalog(searchQuery.trim());
      if (res && res.length > 0) {
        const c = res[0];
        onSelectScene({
          id: c.id,
          name: c.name,
          lat: c.lat,
          lon: c.lon,
          height: c.height,
          description: c.description,
        });
        onClose();
      }
    } catch (e) {
      console.error('Failed to resolve crater:', e);
    }
  };

  const scroll = (direction: 'left' | 'right') => {
    if (scrollContainerRef.current) {
      const offset = direction === 'left' ? -320 : 320;
      scrollContainerRef.current.scrollBy({ left: offset, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-x-0 bottom-0 z-40 select-none font-sans pointer-events-auto flex flex-col justify-end"
      onPointerDown={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      {/* Non-intrusive bottom vignette — does not blur or darken the 3D Moon */}
      <div
        className="fixed inset-x-0 bottom-0 h-96 bg-gradient-to-t from-black/80 via-black/30 to-transparent pointer-events-none"
      />

      {/* ── PLANETARY EXPLORATION TRAY (NASA EYES & GOOGLE EARTH VOYAGER MODEL) ── */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 pb-4 animate-in slide-in-from-bottom-5 duration-200">
        <div className="bg-[#0A0C11]/95 backdrop-blur-2xl border border-white/15 rounded-3xl p-4 shadow-[0_24px_80px_rgba(0,0,0,0.95)] overflow-hidden">
          
          {/* ── TRAY HEADER ROW 1: TITLE + SEARCH + CLOSE ── */}
          <div className="flex items-center justify-between pb-3 border-b border-white/10">
            <div className="flex items-center gap-2.5">
              <div className="w-2 h-2 rounded-full bg-[#2997FF] shadow-[0_0_8px_rgba(41,151,255,0.8)] animate-pulse" />
              <span className="text-xs font-bold uppercase tracking-wider text-white">
                Lunar Exploration & Landing Catalog
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/10 text-white/60">
                {filteredTargets.length} Curated Sites · Global IAU Search
              </span>
            </div>

            <div className="flex items-center gap-2">
              {/* Compact Search Input */}
              <div className="relative w-48 sm:w-64">
                <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40" />
                <input
                  type="text"
                  autoFocus
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      if (filteredTargets.length > 0) {
                        const target = filteredTargets[0];
                        onSelectScene({
                          id: target.id,
                          name: target.name,
                          lat: target.lat,
                          lon: target.lon,
                          height: target.height,
                          description: target.description,
                        });
                        onClose();
                      } else {
                        handleSynthesizeCustomCrater();
                      }
                    }
                  }}
                  placeholder="Search crater (e.g. Copernicus, Tycho, Shackleton)..."
                  className="w-full bg-white/[0.06] border border-white/10 focus:border-[#2997FF] rounded-full pl-8 pr-7 py-1 text-xs text-white placeholder-white/30 focus:outline-none transition-all font-mono"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-white/40 hover:text-white cursor-pointer"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>

              {/* Close Tray Button */}
              <button
                onClick={onClose}
                className="p-1.5 rounded-full text-white/50 hover:text-white hover:bg-white/10 transition-colors cursor-pointer ml-1"
                title="Dismiss Tray (ESC)"
              >
                <X size={15} />
              </button>
            </div>
          </div>

          {/* ── TRAY HEADER ROW 2: CATEGORY SHELVES + CAROUSEL SCROLL ARROWS ── */}
          <div className="flex items-center justify-between pt-2.5 pb-1">
            <div className="flex items-center gap-1.5 overflow-x-auto">
              {[
                { id: 'all', label: 'All Craters' },
                { id: 'polar', label: 'South Pole PSRs' },
                { id: 'water', label: 'Volatiles & Ice' },
                { id: 'highland', label: 'Highland Corridors' },
              ].map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setActiveFilter(cat.id as any)}
                  className={`px-3 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer whitespace-nowrap ${
                    activeFilter === cat.id
                      ? 'bg-white text-black shadow-md'
                      : 'text-white/60 hover:text-white hover:bg-white/10'
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            {/* Carousel Scroll Arrows */}
            <div className="flex items-center gap-1.5 shrink-0">
              <button
                onClick={() => scroll('left')}
                className="p-1.5 rounded-full bg-white/5 hover:bg-white/15 border border-white/10 text-white/70 hover:text-white transition-colors cursor-pointer flex items-center justify-center"
                title="Scroll Left"
              >
                <ChevronLeft size={14} />
              </button>
              <button
                onClick={() => scroll('right')}
                className="p-1.5 rounded-full bg-white/5 hover:bg-white/15 border border-white/10 text-white/70 hover:text-white transition-colors cursor-pointer flex items-center justify-center"
                title="Scroll Right"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>

          {/* ── HORIZONTAL VISUAL CARDS CAROUSEL (ZERO OPERATING SYSTEM SCROLLBAR) ── */}
          <div
            ref={scrollContainerRef}
            className="flex items-center gap-3.5 pt-2 pb-1 overflow-x-auto [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
          >
            {filteredTargets.length === 0 ? (
              <div className="w-full flex flex-col items-center justify-center py-6 gap-2.5 text-center">
                <p className="text-white/60 text-xs font-mono">
                  &ldquo;{searchQuery}&rdquo; is not currently in quick-catalog presets.
                </p>
                <button
                  onClick={handleSynthesizeCustomCrater}
                  className="px-5 py-2 rounded-full bg-[#0071E3] hover:bg-[#0077ED] text-white text-xs font-semibold flex items-center gap-2 shadow-[0_0_20px_rgba(0,113,227,0.5)] cursor-pointer transition-all active:scale-95"
                >
                  <Compass size={14} className="animate-spin" />
                  <span>Compute Orbit & Rendezvous with &ldquo;{searchQuery}&rdquo;</span>
                </button>
              </div>
            ) : (
              filteredTargets.map((target) => {
                const isSelected = selectedScene.id === target.id;

                return (
                  <div
                    key={target.id}
                    onClick={() => {
                      onSelectScene({
                        id: target.id,
                        name: target.name,
                        lat: target.lat,
                        lon: target.lon,
                        height: target.height,
                        description: target.description,
                      });
                      onClose();
                    }}
                    className={`group relative w-64 h-36 shrink-0 rounded-2xl overflow-hidden cursor-pointer border transition-all duration-300 flex flex-col justify-between p-3 select-none ${
                      isSelected
                        ? 'border-[#2997FF] shadow-[0_0_24px_rgba(41,151,255,0.45)] ring-1 ring-[#2997FF]'
                        : 'border-white/15 hover:border-white/40 hover:shadow-[0_8px_24px_rgba(0,0,0,0.6)]'
                    }`}
                  >
                    {/* Full-bleed Authentic Lunar Terrain Imagery */}
                    <img
                      src={target.thumbnail}
                      onError={(e) => {
                        const img = e.target as HTMLImageElement;
                        if (img.src !== target.fallbackThumbnail) {
                          img.src = target.fallbackThumbnail;
                        }
                      }}
                      alt={target.name}
                      className="absolute inset-0 w-full h-full object-cover group-hover:scale-105 transition-transform duration-500 brightness-90"
                    />

                    {/* Gradient Overlay for Crisp Typography */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/95 via-black/40 to-black/20" />

                    {/* Top Row: Coordinates & Active Indicator */}
                    <div className="relative z-10 flex items-center justify-between gap-1">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-black/60 backdrop-blur-md border border-white/20 text-white/90">
                        {Math.abs(target.lat).toFixed(1)}°{target.lat < 0 ? 'S' : 'N'}, {Math.abs(target.lon).toFixed(1)}°{target.lon < 0 ? 'W' : 'E'}
                      </span>

                      {isSelected ? (
                        <span className="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#0071E3] text-white shadow-md flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                          <span>Active</span>
                        </span>
                      ) : (
                        <span className="text-[9px] text-white/50 font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                          Fly to site ➔
                        </span>
                      )}
                    </div>

                    {/* Bottom Row: Site Name, Region, Physical Telemetry */}
                    <div className="relative z-10">
                      <h3 className="text-xs sm:text-sm font-bold text-white group-hover:text-[#2997FF] transition-colors truncate drop-shadow">
                        {target.name}
                      </h3>
                      <p className="text-[10px] sm:text-[11px] text-white/60 truncate mt-0.5">
                        {target.region}
                      </p>

                      {/* Scientific Telemetry Metric Bar */}
                      <div className="flex items-center gap-2 mt-1.5 pt-1.5 border-t border-white/15 text-[10px] font-mono text-white/75">
                        <span>⌀ {target.diameterKm}km</span>
                        <span>·</span>
                        <span className="text-sky-300 font-semibold">{target.waterAbsorptionDepth}% H₂O</span>
                        <span>·</span>
                        <span>{target.floorSlopeDeg}° slope</span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* ── FOOTER SUB-BAR ── */}
          <div className="flex items-center justify-between pt-2 border-t border-white/10 text-[10px] text-white/40 font-mono">
            <span>Selecting a candidate zone executes real-time 3D orbital rendezvous</span>
            <span className="hidden sm:inline">Coordinate Reference: IAU Lunar Frame · Press ESC to dismiss</span>
          </div>
        </div>
      </div>
    </div>
  );
};
