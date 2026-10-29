import React from 'react';
import { ChevronRight } from 'lucide-react';

interface HeroProps {
  onNext: () => void;
  onLaunchWorkbench?: () => void;
  isLaunching?: boolean;
}

export const Hero: React.FC<HeroProps> = ({ onNext, onLaunchWorkbench, isLaunching = false }) => {
  return (
    <section
      id="overview"
      className="relative w-full min-h-[100vh] bg-transparent overflow-hidden flex flex-col justify-between pt-20 md:pt-24 pb-8 px-6 sm:px-10 lg:px-16 border-b border-white/[0.08]"
    >
      {/* Left Column: Clean Typography (Selectable text, does not rotate Moon!) */}
      <div className="relative z-10 max-w-7xl mx-auto w-full my-auto flex-1 flex flex-col justify-center py-6 pointer-events-none">
        <div
          className={`w-full max-w-[420px] sm:max-w-[480px] lg:max-w-xl flex flex-col items-start text-left pointer-events-auto select-text transition-all duration-700 ${
            isLaunching ? 'opacity-0 -translate-x-12 pointer-events-none' : 'opacity-100 translate-x-0'
          }`}
        >
          {/* Grand Problem Statement Headline */}
          <h1 className="font-headline tracking-[-0.035em] text-3xl sm:text-4xl md:text-5xl lg:text-[56px] font-bold text-white mb-4 leading-[1.08] select-text">
            Multi-Sensor Lunar <br />
            <span className="text-[#86868b]">Image Registration.</span>
          </h1>

          {/* Descriptive Copy */}
          <p className="font-sans text-sm sm:text-base md:text-lg text-[#A1A1A6] leading-relaxed max-w-[340px] sm:max-w-md lg:max-w-lg mb-8 font-normal select-text">
            Autonomous alignment connecting Chandrayaan-2 high-resolution payloads (OHRC, TMC-2, IIRS) with global LRO baselines. Built to withstand extreme polar illumination and scale disparities.
          </p>

          {/* Action Group */}
          <div className="flex items-center gap-5 mb-10">
            <button
              onClick={onLaunchWorkbench}
              disabled={isLaunching}
              className="px-6 py-3 rounded-full bg-white text-black font-semibold text-sm hover:bg-[#E5E5EA] active:scale-[0.98] transition-all cursor-pointer shadow-sm shrink-0"
            >
              {isLaunching ? 'Entering Orbit...' : 'Launch 3D Workbench'}
            </button>
            <button
              onClick={onNext}
              disabled={isLaunching}
              className="inline-flex items-center gap-1 text-sm font-medium text-[#2997FF] hover:text-[#70B4FF] transition-colors cursor-pointer group shrink-0"
            >
              <span>Explore the pipeline</span>
              <ChevronRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
            </button>
          </div>

          {/* Minimal Stats Grid */}
          <div className="flex items-baseline gap-6 sm:gap-10 pt-6 border-t border-white/10 w-full max-w-md select-text">
            <div>
              <div className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight text-white select-text">&lt; 0.4 px</div>
              <div className="text-xs text-[#86868b] mt-1 font-sans select-text">Spatial Precision</div>
            </div>
            <div className="h-8 w-px bg-white/10" />
            <div>
              <div className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight text-white select-text">MAGSAC++</div>
              <div className="text-xs text-[#86868b] mt-1 font-sans select-text">Robust Consensus</div>
            </div>
            <div className="h-8 w-px bg-white/10" />
            <div>
              <div className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight text-white select-text">42 ms</div>
              <div className="text-xs text-[#86868b] mt-1 font-sans select-text">Pipeline latency</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Bar: Clean Subtle Footer */}
      <div className="relative z-10 max-w-7xl mx-auto w-full pt-4 flex items-center justify-between border-t border-white/[0.08] text-xs text-[#86868b] font-sans">
        <div>Problem Statement 26166 • Autonomous Precision Engine</div>
        <button
          onClick={onNext}
          disabled={isLaunching}
          className="hover:text-white transition-colors cursor-pointer flex items-center gap-1 pointer-events-auto"
        >
          <span>Next: Pipeline Architecture</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </section>
  );
};
