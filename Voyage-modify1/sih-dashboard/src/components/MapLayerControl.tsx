import React, { useState, useRef, useEffect } from 'react';
import { Layers, Eye, Check, Mountain, Droplets, CircleDot, Grid3X3, Sliders, X } from 'lucide-react';
import type { LayerVisibility, PipelineOptions } from '../types';

interface MapLayerControlProps {
  layers: LayerVisibility;
  onLayerChange: (layers: LayerVisibility) => void;
  options: PipelineOptions;
  onOptionsChange: (options: PipelineOptions) => void;
}

export const MapLayerControl: React.FC<MapLayerControlProps> = ({
  layers,
  onLayerChange,
  options,
  onOptionsChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const toggleLayer = (key: keyof LayerVisibility) => {
    onLayerChange({ ...layers, [key]: !layers[key] });
  };

  const toggleOption = (key: keyof PipelineOptions) => {
    onOptionsChange({ ...options, [key]: !options[key] });
  };

  return (
    <div
      ref={menuRef}
      className="relative pointer-events-auto select-none font-sans"
      onPointerDown={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      {/* ── FLOATING TRIGGER CAPSULE ── */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3.5 py-2 rounded-full border backdrop-blur-xl transition-all cursor-pointer shadow-[0_8px_32px_rgba(0,0,0,0.6)] ${
          isOpen
            ? 'bg-white text-black border-white shadow-lg'
            : 'bg-black/70 hover:bg-black/90 text-white/80 hover:text-white border-white/15'
        }`}
        title="Toggle Surface Layers & Preprocessing"
      >
        <Layers size={14} />
        <span className="text-xs font-bold">Layers</span>
      </button>

      {/* ── FLOATING LAYERS POPOVER ── */}
      {isOpen && (
        <div className="absolute bottom-12 left-0 w-80 bg-[#0B0D13]/95 backdrop-blur-3xl border border-white/20 rounded-3xl p-4 shadow-[0_24px_80px_rgba(0,0,0,0.9)] animate-in fade-in slide-in-from-bottom-2 duration-150 z-30">
          <div className="flex items-center justify-between pb-3 border-b border-white/10 mb-3">
            <div className="flex items-center gap-2">
              <Layers size={14} className="text-[#2997FF]" />
              <span className="text-xs font-bold uppercase tracking-wider text-white">
                Terrain & Sensor Layers
              </span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            >
              <X size={13} />
            </button>
          </div>

          {/* Core Surface Layers */}
          <div className="space-y-1.5">
            {[
              { key: 'basemap' as const, label: 'LRO NAC Orbital Basemap', desc: '0.5m/px global surface mosaic', icon: Eye },
              { key: 'dem' as const, label: 'TMC-2 Stereo Elevation DEM', desc: '5m stereo topographic relief', icon: Mountain },
              { key: 'waterIce' as const, label: 'IIRS 3.0 µm Water-Ice Map', desc: 'Hydroxyl & volatile absorption bands', icon: Droplets },
              { key: 'craters' as const, label: 'Crater Neural Polygons', desc: 'YOLO-detected impact ring boundaries', icon: CircleDot },
              { key: 'grid' as const, label: 'Selenographic Lat/Lon Grid', desc: '5° interval coordinate meridians', icon: Grid3X3 },
            ].map((layer) => {
              const active = layers[layer.key];
              const Icon = layer.icon;
              return (
                <div
                  key={layer.key}
                  onClick={() => toggleLayer(layer.key)}
                  className={`p-2.5 rounded-2xl border transition-all cursor-pointer flex items-center justify-between ${
                    active
                      ? 'bg-white/[0.08] border-white/20 text-white'
                      : 'bg-white/[0.02] border-white/5 text-white/50 hover:bg-white/[0.04] hover:text-white'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0 pr-2">
                    <Icon size={14} className={active ? 'text-[#2997FF]' : 'text-white/40'} />
                    <div className="truncate">
                      <div className="text-xs font-bold truncate">{layer.label}</div>
                      <div className="text-[10px] text-white/40 truncate">{layer.desc}</div>
                    </div>
                  </div>
                  <div
                    className={`w-4 h-4 rounded-md flex items-center justify-center shrink-0 transition-colors ${
                      active ? 'bg-[#2997FF] text-white' : 'border border-white/25 bg-transparent'
                    }`}
                  >
                    {active && <Check size={10} strokeWidth={3} />}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Radiometric Pipeline Sub-section */}
          <div className="mt-4 pt-3 border-t border-white/10">
            <div className="text-[10px] font-bold uppercase tracking-wider text-white/40 mb-2 flex items-center gap-1.5">
              <Sliders size={11} />
              <span>Radiometric Preprocessing</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { key: 'clahe' as const, label: 'CLAHE' },
                { key: 'percentileClipping' as const, label: 'Percentile Clip' },
                { key: 'morphologicalGradients' as const, label: 'Morph Gradients' },
                { key: 'pcaBandReduction' as const, label: 'PCA Reduction' },
              ].map((opt) => {
                const checked = (options as any)[opt.key];
                return (
                  <button
                    key={opt.key}
                    onClick={() => toggleOption(opt.key)}
                    className={`p-2 rounded-xl text-[10px] font-bold border text-left transition-all cursor-pointer flex items-center justify-between ${
                      checked
                        ? 'bg-white/[0.08] border-white/20 text-white'
                        : 'bg-white/[0.02] border-white/5 text-white/50 hover:bg-white/[0.04] hover:text-white'
                    }`}
                  >
                    <span>{opt.label}</span>
                    <div
                      className={`w-3.5 h-3.5 rounded flex items-center justify-center shrink-0 ${
                        checked ? 'bg-[#0071E3] text-white' : 'border border-white/20'
                      }`}
                    >
                      {checked && <Check size={8} strokeWidth={3} />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
