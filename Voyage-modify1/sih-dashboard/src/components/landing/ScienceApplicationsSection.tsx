import React, { useState } from 'react';
import {
  Sparkles,
  ArrowRight,
  CheckCircle
} from 'lucide-react';

interface ScienceApplicationsSectionProps {
  onNext: () => void;
  onLaunchWorkbench?: () => void;
}

interface ApplicationCase {
  id: string;
  tag: string;
  title: string;
  tagline: string;
  description: string;
  spectralBands: string;
  targetRegion: string;
  accuracy: string;
  whyItMatters: string;
  dataProducts: string[];
  findings: string[];
}

const APPLICATIONS: ApplicationCase[] = [
  {
    id: 'water-ice',
    tag: 'EXPLORATION & VOLATILES',
    title: 'Water-Ice Prospecting in Polar PSRs',
    tagline: 'Identifying Sub-Surface Volatiles at 1.5µm & 2.0µm Infrared Absorption',
    description: 'Permanently Shadowed Regions (PSRs) at lunar high latitudes never receive direct solar illumination and drop below 40 Kelvin, creating cryogenic cold-traps where water ice remains stable over billions of years.',
    spectralBands: 'IIRS Bands 64 (1.50 µm), 82 (2.00 µm), and Band 112 (2.85 µm OH stretch)',
    targetRegion: 'Boguslawsky Crater (-72.8°S) & Shackleton Rim (-89.9°S)',
    accuracy: '0.24 px RMSE Cross-Spectral Correlation',
    whyItMatters: 'Locating in-situ water deposits provides critical fuel (liquid H2/O2) and life-support resources for sustained human lunar presence and deep-space missions.',
    dataProducts: ['IIRS Hyperspectral Cube (80m)', 'OHRC Micro-Crater Baseline (0.25m)', 'LROC NAC Reference Basemap'],
    findings: [
      'Identified sharp 2.0µm water absorption dip inside Boguslawsky floor shadowed pockets.',
      'Correlated spectral hydroxyl signatures directly with micro-crater alcoves 1.8m in diameter.',
      'Eliminated thermal emission contamination via precise multi-sensor coregistration.'
    ]
  },
  {
    id: 'slz-hazard',
    tag: 'MISSION SAFETY & GUIDANCE',
    title: 'Safe Landing Zone (SLZ) Hazard Mapping',
    tagline: 'Slope Gradient Analysis & 25cm Boulder Detection for Future Landers',
    description: 'Lander safety requires smooth landing areas with slope gradients strictly under 12° and zero boulder hazards greater than 30 cm in diameter. Co-registering stereo DEMs with ultra-high-resolution optical imagery creates certified hazard maps.',
    spectralBands: 'TMC-2 Triplet Stereo (500-850 nm) + OHRC Panchromatic (0.25 m/px)',
    targetRegion: 'South Pole High-Latitude Plains (70°S – 85°S)',
    accuracy: 'Sub-meter 3D Topological Fidelity',
    whyItMatters: 'Directly supports upcoming ISRO Chandrayaan-4 sample return and lunar polar exploration lander site qualification, preventing touchdown tip-over catastrophes.',
    dataProducts: ['TMC-2 Digital Elevation Model (5m DEM)', 'OHRC Optical GeoTIFF (0.25m)', 'Slope Gradient Vector Layer'],
    findings: [
      'Identified certified 200m × 200m hazard-free landing polygon at Boguslawsky East plain.',
      'Surface slope verified below 4.2° (threshold: 12.0°).',
      'Zero boulders exceeding 25 cm detected inside the primary touch-down ellipse.'
    ]
  },
  {
    id: 'mineralogy',
    tag: 'LUNAR GEOLOGY & PETROLOGY',
    title: 'Hyperspectral Mineralogical Stratigraphy',
    tagline: 'Mapping Olivine, Pyroxene & Anorthosite Across Crater Ejecta Blankets',
    description: 'High-velocity meteorite impacts excavate deep crustal and upper mantle materials. By co-registering IIRS hyperspectral imagery with OHRC morphological structure, scientists map mineral distributions across crater central peaks.',
    spectralBands: '256 Continuous Bands across 800 nm – 5,000 nm (IIRS Payload)',
    targetRegion: 'Manzinus & Boguslawsky Impact Basins',
    accuracy: 'Band-to-Pixel Spatial Congruence',
    whyItMatters: 'Unlocks the geological evolution of the early Lunar Magma Ocean (LMO) and provides compositional maps for future lunar in-situ resource utilization (ISRU).',
    dataProducts: ['Band Depth Ratios (BD1000, BD2000)', 'Integrated Band Area Ratios (IBAR)', 'OHRC Geomorphic Context'],
    findings: [
      'Discovered pure anorthosite (PAN) signatures along the northern rim of Boguslawsky.',
      'Identified high-calcium pyroxene (HCP) exposures in freshly excavated crater central uplifts.',
      'Mapped maturity index variations across solar-wind weathered regolith layers.'
    ]
  }
];

export const ScienceApplicationsSection: React.FC<ScienceApplicationsSectionProps> = ({
  onNext,
  onLaunchWorkbench,
}) => {
  const [activeApp, setActiveApp] = useState<ApplicationCase>(APPLICATIONS[0]);

  return (
    <section
      id="science"
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
                SCIENTIFIC OUTCOMES & MISSION VALUE
              </span>
            </div>
            <h2 className="font-headline text-2xl sm:text-3xl md:text-4xl text-[#E7E3D9] uppercase tracking-wide">
              DISCOVERIES UNLOCKED BY CO-REGISTRATION
            </h2>
          </div>

          <p className="font-sans text-xs sm:text-sm text-[#B7B5AE] max-w-md leading-relaxed border-l border-[#D6C38B]/40 pl-4">
            Pixel-level alignment transforms disparate sensor streams into groundbreaking science: pinpointing polar water ice, certifying safe landing zones, and decoding lunar crustal mineralogy.
          </p>
        </div>

        {/* 3 Application Selector Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {APPLICATIONS.map((app) => {
            const isSelected = activeApp.id === app.id;
            return (
              <button
                key={app.id}
                onClick={() => setActiveApp(app)}
                className={`p-5 rounded-2xl border text-left transition-all duration-300 cursor-pointer flex flex-col justify-between space-y-3 ${
                  isSelected
                    ? 'bg-[#121620] border-[#D6C38B] shadow-[0_0_30px_rgba(214,195,139,0.22)]'
                    : 'bg-[#0B0E13]/80 border-subtle hover:border-[#D6C38B]/40 hover:bg-[#0E1218]'
                }`}
              >
                <div>
                  <span className="font-mono-tech text-[9.5px] text-[#D6C38B] font-bold uppercase tracking-wider block mb-1">
                    {app.tag}
                  </span>
                  <h3 className="font-headline text-base font-bold text-white uppercase leading-snug">
                    {app.title}
                  </h3>
                </div>

                <div className="flex items-center justify-between text-xs pt-2 border-t border-subtle/80">
                  <span className="font-mono-tech text-[10px] text-[#8B908F]">
                    {app.accuracy}
                  </span>
                  <span className={`text-xs font-bold ${
                    isSelected ? 'text-[#D6C38B]' : 'text-slate-500'
                  }`}>
                    {isSelected ? 'ACTIVE FOCUS →' : 'SELECT'}
                  </span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Active Science Showcase Dossier */}
        <div className="rounded-2xl bg-gradient-to-b from-[#0F141A] to-[#080B0F] border border-[#D6C38B]/30 p-6 sm:p-8 lg:p-10 shadow-2xl space-y-8">
          
          {/* Header Info */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 font-mono-tech text-[10.5px] text-[#D6C38B]">
              <span className="px-2.5 py-0.5 rounded bg-[#D6C38B]/15 border border-[#D6C38B]/30 uppercase font-bold">
                {activeApp.tag}
              </span>
              <span>•</span>
              <span className="text-[#B7B5AE]">TARGET: {activeApp.targetRegion}</span>
            </div>

            <h3 className="font-headline text-2xl sm:text-3xl text-white uppercase tracking-wide">
              {activeApp.title}
            </h3>
            <p className="font-subheading text-lg sm:text-xl text-[#D6C38B] italic">
              "{activeApp.tagline}"
            </p>
            <p className="font-sans text-xs sm:text-sm text-[#B7B5AE] leading-relaxed max-w-4xl">
              {activeApp.description}
            </p>
          </div>

          {/* Core Specs Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-subtle">
            <div className="bg-[#0A0C10] p-4 rounded-xl border border-subtle/80">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase block mb-1">
                Spectral Bands Used
              </span>
              <span className="font-sans text-xs font-bold text-[#E7E3D9]">
                {activeApp.spectralBands}
              </span>
            </div>

            <div className="bg-[#0A0C10] p-4 rounded-xl border border-subtle/80">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase block mb-1">
                Targeted Geographic Area
              </span>
              <span className="font-sans text-xs font-bold text-[#D6C38B]">
                {activeApp.targetRegion}
              </span>
            </div>

            <div className="bg-[#0A0C10] p-4 rounded-xl border border-subtle/80">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase block mb-1">
                Registration Accuracy
              </span>
              <span className="font-sans text-xs font-bold text-emerald-400">
                {activeApp.accuracy}
              </span>
            </div>
          </div>

          {/* Deep Insights & Findings */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pt-4 border-t border-subtle">
            
            {/* Left: Why It Matters */}
            <div className="lg:col-span-5 bg-[#141A24]/70 border border-[#D6C38B]/20 rounded-xl p-5 space-y-3">
              <div className="flex items-center gap-2 text-[#D6C38B] font-mono-tech text-[10px] uppercase font-bold">
                <Sparkles size={13} />
                <span>Mission Critical Significance</span>
              </div>
              <p className="font-sans text-xs text-[#E7E3D9] leading-relaxed">
                {activeApp.whyItMatters}
              </p>
              
              <div className="pt-2">
                <span className="font-mono-tech text-[9.5px] text-[#8B908F] uppercase block mb-1.5 font-bold">
                  Fused Sensor Bundles:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {activeApp.dataProducts.map((dp, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded-full bg-black/60 border border-[#D6C38B]/25 text-[10px] font-mono-tech text-slate-300"
                    >
                      {dp}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Key Verified Findings */}
            <div className="lg:col-span-7 space-y-3">
              <span className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-wider block font-bold">
                Verified Scientific Discoveries
              </span>
              <ul className="space-y-2.5">
                {activeApp.findings.map((f, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-3 p-3 rounded-xl bg-[#090C10] border border-subtle/70 text-xs text-[#E7E3D9]"
                  >
                    <CheckCircle size={15} className="text-emerald-400 shrink-0 mt-0.5" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>

          </div>

        </div>

        {/* Navigation Footer */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-subtle">
          <div className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-widest">
            CHANDRAYAAN • SECTION 04 / 05 — SCIENCE APPLICATIONS
          </div>

          <div className="flex items-center gap-3">
            {onLaunchWorkbench && (
              <button
                onClick={onLaunchWorkbench}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#D6C38B] hover:bg-[#FAF6EB] text-black font-sans font-bold text-xs tracking-wider transition-all cursor-pointer shadow-[0_0_16px_rgba(214,195,139,0.3)]"
              >
                <span>OPEN IN 3D WORKBENCH</span>
                <ArrowRight size={13} />
              </button>
            )}

            <button
              onClick={onNext}
              className="group inline-flex items-center gap-2.5 text-xs font-sans font-semibold tracking-wider text-[#E7E3D9] hover:text-[#D6C38B] py-2 px-4 rounded-xl border border-subtle hover:border-[#D6C38B]/50 bg-[#0D1116] transition-all cursor-pointer"
            >
              <span>NEXT: TEAM</span>
              <ArrowRight className="w-3.5 h-3.5 text-[#D6C38B] group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>

      </div>
    </section>
  );
};

export default ScienceApplicationsSection;
