import React, { useState } from 'react';
import {
  Layers,
  ArrowRight,
  Sparkles,
  CheckCircle2
} from 'lucide-react';

interface PipelineSectionProps {
  onNext: () => void;
  onLaunchWorkbench?: () => void;
}

interface StageDetail {
  id: string;
  number: string;
  title: string;
  subtitle: string;
  description: string;
  inputs: string[];
  outputs: string[];
  algorithms: string[];
  metrics: { label: string; val: string }[];
  keyInsight: string;
}

const STAGES: StageDetail[] = [
  {
    id: 'l1',
    number: '01',
    title: 'Radiometric Normalization & Filtering',
    subtitle: 'Overcoming Lunar Shadows & Solar Phase Discrepancy',
    description: 'Chandrayaan-2 imagery suffers from extreme contrast due to grazing sun-angles at the South Pole. This stage normalizes radiometry across optical, stereo, and infrared sensors.',
    inputs: ['Raw PDS4 .IMG (OHRC)', 'GeoTIFF DEM (TMC-2)', '256-band Spectral Cube (IIRS)'],
    outputs: ['8-bit Normalized Tile Patches', 'Binary Shadow & Crater Masks', 'PCA Band 1-3 Composite'],
    algorithms: ['Contrast Limited Adaptive Histogram Equalization (CLAHE)', '2nd/98th Percentile Dynamic Clipping', 'Morphological Gradient Filtering', 'Principal Component Analysis (PCA)'],
    metrics: [
      { label: 'Dynamic Range', val: '16-bit to 8-bit' },
      { label: 'Clip Threshold', val: '2.0% – 98.0%' },
      { label: 'Spectral Retention', val: '99.4% Variance' }
    ],
    keyInsight: 'Standard pixel matching fails when shadows invert between seasons. Radiometric normalization extracts structural edges invariant to illumination angle.'
  },
  {
    id: 'l2',
    number: '02',
    title: 'Multi-Engine Feature Matching',
    subtitle: 'Resolving the 320× Scale Jump (0.25m to 80m/px)',
    description: 'A hybrid ensemble of 4 complementary matching engines combines classical scale-space theory, frequency-domain phase congruency, and deep geometric attention.',
    inputs: ['Normalized Source Image', 'Reference Orthoimage / LROC NAC Baseline'],
    outputs: ['500+ Candidate Correspondence Pairs', 'Feature Descriptor Vectors', 'Confidence Heatmaps'],
    algorithms: [
      'Scale-Invariant Feature Transform (SIFT) for multi-octave pyramids',
      'Radiation-Invariant Feature Transform (RIFT-2) using log-Gabor phase congruency',
      'LightGlue Transformer with deep cross-attention layers',
      'Curvature-based Lunar Crater Rim Edge Matching'
    ],
    metrics: [
      { label: 'Candidate Matches', val: '500 – 1,200 pts' },
      { label: 'Phase Invariance', val: '100% Light-Blind' },
      { label: 'Scale Handling', val: '0.25m → 80m/px' }
    ],
    keyInsight: 'RIFT-2 calculates phase congruency rather than raw intensity, enabling flawless matching between visual optical images (OHRC) and thermal/infrared bands (IIRS).'
  },
  {
    id: 'l3',
    number: '03',
    title: 'Spatial Selection & Geometric Verification',
    subtitle: 'Eliminating Outliers with MAGSAC++ & ANMS',
    description: 'Raw correspondences are clustered and noisy. This stage guarantees homogeneous surface distribution and verifies that matches follow true lunar projective physics.',
    inputs: ['Candidate Match Coordinates (x, y)', 'Feature Descriptors'],
    outputs: ['Uniformly Distributed Tie Points', 'Inlier Mask (70-85% clean)', 'Estimated 3×3 Homography Matrix'],
    algorithms: [
      'Adaptive Non-Maximal Suppression (ANMS) for spatial spread',
      'MAGSAC++ (Marginalizing Sample Consensus) with sigma-consensus',
      'DEGENSAC for degenerate planar surface detection',
      'Epipolar Geometry Constraint Matrix'
    ],
    metrics: [
      { label: 'Spatial Coverage', val: '> 84% Tile Area' },
      { label: 'Inlier Ratio', val: '78.4% Clean' },
      { label: 'Outlier Rejection', val: '99.2% Robust' }
    ],
    keyInsight: 'Clusters of points around high-contrast crater rims can skew global transforms. ANMS enforces spatial dispersion across flat maria and rugged highlands alike.'
  },
  {
    id: 'l4',
    number: '04',
    title: 'Sub-Pixel Refinement & Fusion',
    subtitle: 'Achieving Sub-Quarter-Pixel Precision (< 0.28 px RMSE)',
    description: 'Final refinement optimizes correspondence coordinates using cross-correlation surface paraboloid fitting, followed by DEM-guided orthorectification and data fusion.',
    inputs: ['Verified Inlier Points', 'TMC-2 High-Resolution Elevation (DEM)'],
    outputs: ['Sub-Pixel Precision Coordinates', 'Thin-Plate Spline (TPS) Warped Surface', 'Multi-Layer Registered GeoTIFF'],
    algorithms: [
      'Phase-Only Correlation (POC) with 2D quadratic peak interpolation',
      'Thin Plate Spline (TPS) non-rigid deformation warping',
      'Ray-tracing DEM back-projection for relief displacement correction',
      'Bicubic Hydration Layer Overlay'
    ],
    metrics: [
      { label: 'Registration Error', val: '0.24 px RMSE' },
      { label: 'Spatial Distortion', val: '< 0.05% Area' },
      { label: 'Alignment Confidence', val: '99.8% Certified' }
    ],
    keyInsight: 'Sub-pixel paraboloid interpolation achieves sub-meter positional accuracy on the lunar surface, meeting strict NASA and ISRO landing guidance standards.'
  }
];

export const PipelineSection: React.FC<PipelineSectionProps> = ({ onNext, onLaunchWorkbench }) => {
  const [selectedStage, setSelectedStage] = useState<StageDetail>(STAGES[0]);

  return (
    <section
      id="pipeline"
      className="relative w-full py-20 sm:py-24 md:py-28 lg:py-32 px-4 sm:px-6 md:px-8 lg:px-12 bg-[#07090C] border-t border-subtle overflow-hidden select-none"
    >
      {/* Background Atmosphere */}
      <div className="absolute inset-0 pointer-events-none grid-lines-lunar opacity-30"></div>
      <div className="absolute inset-0 pointer-events-none lunar-noise-overlay opacity-40"></div>

      <div className="relative z-10 max-w-7xl mx-auto space-y-16 lg:space-y-20">
        
        {/* Section Header */}
        <div className="border-b border-subtle pb-8 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="w-2 h-2 rounded-full bg-[#D6C38B]"></span>
              <span className="font-mono-tech text-[10px] tracking-widest text-[#D6C38B] uppercase">
                SCIENTIFIC METHODOLOGY & ARCHITECTURE
              </span>
            </div>
            <h2 className="font-headline text-2xl sm:text-3xl md:text-4xl text-[#E7E3D9] uppercase tracking-wide">
              END-TO-END REGISTRATION PIPELINE
            </h2>
          </div>

          <p className="font-sans text-xs sm:text-sm text-[#B7B5AE] max-w-md leading-relaxed border-l border-[#D6C38B]/40 pl-4">
            How ECLIPSE bridges the gap between disparate Chandrayaan-2 payloads: turning heterogeneous, multi-angle telemetry into a unified sub-meter lunar cartographic coordinate space.
          </p>
        </div>

        {/* 4 Stage Interactive Tab Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {STAGES.map((stg) => {
            const isSelected = selectedStage.id === stg.id;
            return (
              <button
                key={stg.id}
                onClick={() => setSelectedStage(stg)}
                className={`p-4 rounded-xl border text-left transition-all duration-300 cursor-pointer relative overflow-hidden group ${
                  isSelected
                    ? 'bg-[#10141C] border-[#D6C38B] shadow-[0_0_24px_rgba(214,195,139,0.18)]'
                    : 'bg-[#0B0E13]/80 border-subtle hover:border-[#D6C38B]/40 hover:bg-[#0E1218]'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`font-mono-tech text-xs font-bold ${
                    isSelected ? 'text-[#D6C38B]' : 'text-[#8B908F]'
                  }`}>
                    STAGE {stg.number}
                  </span>
                  {isSelected && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#D6C38B] animate-pulse"></span>
                  )}
                </div>
                <h3 className={`font-sans font-bold text-sm leading-snug mb-1 transition-colors ${
                  isSelected ? 'text-white' : 'text-[#E7E3D9] group-hover:text-white'
                }`}>
                  {stg.title}
                </h3>
                <p className="font-sans text-[11px] text-[#8B908F] line-clamp-2">
                  {stg.subtitle}
                </p>

                {isSelected && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-[#D6C38B] to-[#988756]"></div>
                )}
              </button>
            );
          })}
        </div>

        {/* Selected Stage Deep-Dive Card */}
        <div className="rounded-2xl bg-gradient-to-b from-[#0F141A] to-[#0A0D12] border border-[#D6C38B]/30 p-6 sm:p-8 lg:p-10 shadow-2xl space-y-8">
          
          {/* Top Row: Stage Header & Key Insight */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-7 space-y-3">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#D6C38B]/10 border border-[#D6C38B]/25 text-[#D6C38B] font-mono-tech text-[10px]">
                <Layers size={12} />
                <span>ACTIVE STAGE: {selectedStage.number} / 04</span>
              </div>
              <h3 className="font-headline text-xl sm:text-2xl text-white uppercase tracking-wide">
                {selectedStage.title}
              </h3>
              <p className="font-subheading text-lg text-[#D6C38B] italic">
                {selectedStage.subtitle}
              </p>
              <p className="font-sans text-xs sm:text-sm text-[#B7B5AE] leading-relaxed">
                {selectedStage.description}
              </p>
            </div>

            {/* Key Engineering Insight Pill */}
            <div className="lg:col-span-5 bg-[#141A24]/70 border border-[#D6C38B]/20 rounded-xl p-4 sm:p-5 space-y-2">
              <div className="flex items-center gap-2 text-[#D6C38B] font-mono-tech text-[10px] uppercase font-bold">
                <Sparkles size={13} />
                <span>Crucial Mathematical Insight</span>
              </div>
              <p className="font-sans text-xs text-[#E7E3D9] leading-relaxed">
                {selectedStage.keyInsight}
              </p>
            </div>
          </div>

          {/* Middle Row: Metrics Gauges */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-subtle">
            {selectedStage.metrics.map((m, idx) => (
              <div
                key={idx}
                className="bg-[#0A0C10] p-4 rounded-xl border border-subtle/80 flex flex-col justify-between"
              >
                <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-wider">
                  {m.label}
                </span>
                <span className="font-headline text-xl lg:text-2xl font-bold text-[#D6C38B] mt-1">
                  {m.val}
                </span>
              </div>
            ))}
          </div>

          {/* Bottom Row: Inputs, Outputs & Algorithms */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-subtle text-xs">
            
            {/* Input Data Products */}
            <div className="space-y-2.5">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-wider block font-bold">
                Input Data Products
              </span>
              <ul className="space-y-1.5">
                {selectedStage.inputs.map((inp, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-slate-300">
                    <span className="text-[#D6C38B] font-mono-tech mt-0.5">↳</span>
                    <span>{inp}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Active Algorithms */}
            <div className="space-y-2.5">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-wider block font-bold">
                Active Algorithms & Filters
              </span>
              <ul className="space-y-1.5">
                {selectedStage.algorithms.map((alg, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-slate-300">
                    <CheckCircle2 size={13} className="text-[#D6C38B] shrink-0 mt-0.5" />
                    <span>{alg}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Verified Outputs */}
            <div className="space-y-2.5">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-wider block font-bold">
                Verified Output Artifacts
              </span>
              <ul className="space-y-1.5">
                {selectedStage.outputs.map((out, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-slate-300">
                    <span className="text-emerald-400 font-mono-tech mt-0.5">✓</span>
                    <span>{out}</span>
                  </li>
                ))}
              </ul>
            </div>

          </div>

        </div>

        {/* Section Navigation Footer Bar */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-subtle">
          <div className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-widest">
            CHANDRAYAAN • SECTION 02 / 05 — PIPELINE ARCHITECTURE
          </div>

          <div className="flex items-center gap-3">
            {onLaunchWorkbench && (
              <button
                onClick={onLaunchWorkbench}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#D6C38B] hover:bg-[#FAF6EB] text-black font-sans font-bold text-xs tracking-wider transition-all cursor-pointer shadow-[0_0_16px_rgba(214,195,139,0.3)]"
              >
                <span>TEST ON 3D MOON</span>
                <ArrowRight size={13} />
              </button>
            )}

            <button
              onClick={onNext}
              className="group inline-flex items-center gap-2.5 text-xs font-sans font-semibold tracking-wider text-[#E7E3D9] hover:text-[#D6C38B] py-2 px-4 rounded-xl border border-subtle hover:border-[#D6C38B]/50 bg-[#0D1116] transition-all cursor-pointer"
            >
              <span>NEXT: SENSORS</span>
              <ArrowRight className="w-3.5 h-3.5 text-[#D6C38B] group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>

      </div>
    </section>
  );
};

export default PipelineSection;
