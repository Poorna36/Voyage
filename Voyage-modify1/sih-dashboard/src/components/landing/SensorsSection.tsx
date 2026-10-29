import React, { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { ohrcImg, tmc2Img, iirsImg } from '../../data/lunarisDatasets';

interface SensorsSectionProps {
  onNext: () => void;
}

export const SensorsSection: React.FC<SensorsSectionProps> = ({ onNext }) => {
  const [activeIirsBand, setActiveIirsBand] = useState<'mineral' | 'water' | 'continuum'>('mineral');

  return (
    <section
      id="sensors"
      className="relative w-full py-20 sm:py-24 md:py-28 lg:py-36 px-4 sm:px-6 md:px-8 lg:px-12 bg-[#07090C] border-t border-subtle overflow-hidden select-none"
    >
      {/* Background Atmosphere */}
      <div className="absolute inset-0 pointer-events-none grid-lines-lunar opacity-30"></div>
      <div className="absolute inset-0 pointer-events-none lunar-noise-overlay opacity-40"></div>

      <div className="relative z-10 max-w-7xl mx-auto space-y-20 lg:space-y-28">
        
        {/* Section Editorial Header */}
        <div className="border-b border-subtle pb-8">
          <h2 className="font-headline text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold uppercase tracking-[-0.02em] text-[#E7E3D9]">
            CHANDRAYAAN-2<br />
            <span className="text-[#D6C38B]">SENSORS.</span>
          </h2>
        </div>

        {/* ========================================================================= */}
        {/* SENSOR 01: OHRC (Orbiter High Resolution Camera) */}
        {/* ========================================================================= */}
        <div id="sensor-ohrc" className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-start pt-4">
          
          {/* Left Column: Detailed High-Res Imagery Crop & Zoom Inset */}
          <div className="lg:col-span-7 space-y-4">
            <div className="relative rounded-xl overflow-hidden border border-[#D6C38B]/40 bg-[#0D1116] shadow-2xl group">
              <div className="h-[360px] sm:h-[420px] md:h-[480px] w-full overflow-hidden relative">
                <img
                  src={ohrcImg}
                  alt="OHRC High Resolution Lunar Crater Detail"
                  className="w-full h-full object-cover filter contrast-125 brightness-95 group-hover:scale-105 transition-transform duration-700 ease-out"
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#07090C] via-transparent to-transparent"></div>
                
                {/* Visual Geometry Reticle */}
                <div className="absolute top-6 left-6 font-mono-tech text-[10px] text-[#D6C38B] bg-[#07090C]/85 backdrop-blur-sm px-2.5 py-1 rounded border border-[#D6C38B]/40">
                  FOV: 3.0 km • GSD: 0.25 m/PX
                </div>

                {/* Sub-Meter Boulder Magnifier Inset */}
                <div className="absolute bottom-6 left-6 right-6 sm:right-auto sm:w-72 bg-[#07090C]/90 backdrop-blur-md p-3 rounded-lg border border-[#D6C38B]/50 font-mono-tech text-xs space-y-1.5">
                  <div className="flex items-center justify-between text-[#D6C38B] font-bold text-[10px]">
                    <span>SUB-METER RESOLUTION</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                  </div>
                  <div className="text-[11px] text-[#E7E3D9] font-mono">
                    Resolves meter-scale boulders & micro-craters down to 25 cm.
                  </div>
                  <div className="text-[9px] text-[#8B908F]">
                    LAT: 72.91° S • LON: 53.28° E • TDI CCD SENSOR
                  </div>
                </div>

                {/* Scale reference bar */}
                <div className="absolute bottom-6 right-6 hidden sm:flex flex-col items-end font-mono-tech text-[9px] text-[#D6C38B]">
                  <div className="w-24 h-[2px] bg-[#D6C38B] mb-1"></div>
                  <span>100 METERS</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Narrative & Technical Specifications */}
          <div className="lg:col-span-5 space-y-6 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-baseline gap-3">
                <span className="font-headline text-3xl md:text-4xl text-[#D6C38B] font-bold">01</span>
                <div>
                  <span className="font-mono-tech text-xs text-[#8B908F] uppercase tracking-widest block font-medium">
                    PRIMARY OPTICAL CAMERA
                  </span>
                  <h3 className="font-headline text-2xl sm:text-3xl font-bold uppercase text-[#E7E3D9] tracking-tight">
                    OHRC
                  </h3>
                </div>
              </div>

              <div className="font-serif-italic text-lg text-[#D6C38B]">
                Orbiter High Resolution Camera
              </div>

              {/* Exact user requested resolution */}
              <div className="bg-[#0D1116] p-3 rounded border border-subtle font-mono-tech text-xs flex items-center justify-between">
                <span className="text-[#8B908F] uppercase">Resolution:</span>
                <span className="text-[#D6C38B] text-sm font-bold">0.25 m / pixel</span>
              </div>

              {/* Exact user requested description */}
              <p className="font-sans text-sm md:text-base text-[#E7E3D9] leading-relaxed border-l-2 border-[#D6C38B] pl-4">
                High-resolution optical imagery used to observe detailed lunar surface features over relatively small areas.
              </p>
            </div>

            {/* Scientific Architecture Specs */}
            <div className="pt-4 border-t border-subtle grid grid-cols-2 gap-4 font-mono-tech text-xs">
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SWATH DIMENSION</span>
                <span className="text-[#E7E3D9]">3.0 km (at 100 km orbit)</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SPECTRAL PROFILE</span>
                <span className="text-[#E7E3D9]">450 – 900 nm (Panchromatic)</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">DETECTOR TYPE</span>
                <span className="text-[#E7E3D9]">TDI Mode Linear Array</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">PRIMARY SCIENTIFIC APPLICATION</span>
                <span className="text-[#D6C38B]">Lander Hazard & Boulder Mapping</span>
              </div>
            </div>
          </div>

        </div>


        {/* ========================================================================= */}
        {/* SENSOR 02: TMC-2 (Terrain Mapping Camera-2) */}
        {/* ========================================================================= */}
        <div id="sensor-tmc2" className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-start pt-12 border-t border-subtle">
          
          {/* Left Column: Narrative & Technical Specifications */}
          <div className="lg:col-span-5 space-y-6 flex flex-col justify-between order-2 lg:order-1">
            <div className="space-y-4">
              <div className="flex items-baseline gap-3">
                <span className="font-headline text-3xl md:text-4xl text-[#E7E3D9] font-bold">02</span>
                <div>
                  <span className="font-mono-tech text-xs text-[#8B908F] uppercase tracking-widest block font-medium">
                    STEREO TOPOGRAPHIC CAMERA
                  </span>
                  <h3 className="font-headline text-2xl sm:text-3xl font-bold uppercase text-[#E7E3D9] tracking-tight">
                    TMC-2
                  </h3>
                </div>
              </div>

              <div className="font-serif-italic text-lg text-[#E7E3D9]">
                Terrain Mapping Camera-2
              </div>

              {/* Exact user requested resolution */}
              <div className="bg-[#0D1116] p-3 rounded border border-subtle font-mono-tech text-xs flex items-center justify-between">
                <span className="text-[#8B908F] uppercase">Resolution:</span>
                <span className="text-[#E7E3D9] text-sm font-bold">5 m / pixel</span>
              </div>

              {/* Exact user requested description */}
              <p className="font-sans text-sm md:text-base text-[#E7E3D9] leading-relaxed border-l-2 border-[#E7E3D9]/60 pl-4">
                Medium-resolution stereo imagery that provides broader terrain and topographic context.
              </p>
            </div>

            {/* Stereo Triplet Look Angle Geometry */}
            <div className="p-4 bg-[#0D1116] rounded-lg border border-subtle space-y-3 font-mono-tech text-xs">
              <span className="text-[10px] text-[#8B908F] uppercase tracking-wider block font-bold">
                TRIPLET STEREO VIEW GEOMETRY
              </span>
              <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                <div className="bg-[#07090C] p-2 rounded border border-subtle">
                  <span className="text-[#8B908F] block">FORE VIEW</span>
                  <span className="text-[#E7E3D9] font-bold">+26° Look</span>
                </div>
                <div className="bg-[#07090C] p-2 rounded border border-subtle border-[#D6C38B]/40">
                  <span className="text-[#D6C38B] block">NADIR VIEW</span>
                  <span className="text-[#D6C38B] font-bold">0° Vertical</span>
                </div>
                <div className="bg-[#07090C] p-2 rounded border border-subtle">
                  <span className="text-[#8B908F] block">AFT VIEW</span>
                  <span className="text-[#E7E3D9] font-bold">-26° Look</span>
                </div>
              </div>
            </div>

            {/* Scientific Architecture Specs */}
            <div className="pt-2 grid grid-cols-2 gap-4 font-mono-tech text-xs">
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SWATH WIDTH</span>
                <span className="text-[#E7E3D9]">20.0 km across track</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SPECTRAL BAND</span>
                <span className="text-[#E7E3D9]">500 – 850 nm</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">ELEVATION POSTING</span>
                <span className="text-[#E7E3D9]">Digital Elevation Model (DEM)</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">PRIMARY SCIENTIFIC APPLICATION</span>
                <span className="text-[#E7E3D9]">Topographic & Crater Slope Modeling</span>
              </div>
            </div>
          </div>

          {/* Right Column: Wide Terrain Imagery & Stereo DEM Overlay */}
          <div className="lg:col-span-7 space-y-4 order-1 lg:order-2">
            <div className="relative rounded-xl overflow-hidden border border-[#E7E3D9]/30 bg-[#0D1116] shadow-2xl group">
              <div className="h-[360px] sm:h-[420px] md:h-[480px] w-full overflow-hidden relative">
                <img
                  src={tmc2Img}
                  alt="TMC-2 Broad Lunar Geological Terrain Context"
                  className="w-full h-full object-cover filter contrast-115 brightness-90 group-hover:scale-105 transition-transform duration-700 ease-out"
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#07090C] via-transparent to-transparent"></div>

                {/* Top Overlay Badge */}
                <div className="absolute top-6 left-6 font-mono-tech text-[10px] text-[#E7E3D9] bg-[#07090C]/85 backdrop-blur-sm px-2.5 py-1 rounded border border-subtle">
                  SWATH: 20 km • STEREO BASELINE B/H: ~1.0
                </div>

                {/* Elevation contour simulation tag */}
                <div className="absolute bottom-6 left-6 bg-[#07090C]/90 backdrop-blur-md p-3 rounded-lg border border-subtle font-mono-tech text-xs space-y-1">
                  <div className="text-[10px] text-[#E7E3D9] font-bold">
                    REGIONAL TOPOGRAPHY CONTEXT
                  </div>
                  <div className="text-[9px] text-[#8B908F]">
                    Generates 3D Digital Elevation Models linking OHRC patches to global Selenodesy.
                  </div>
                </div>

                {/* Scale reference bar */}
                <div className="absolute bottom-6 right-6 hidden sm:flex flex-col items-end font-mono-tech text-[9px] text-[#E7E3D9]">
                  <div className="w-24 h-[2px] bg-[#E7E3D9] mb-1"></div>
                  <span>2.0 KILOMETERS</span>
                </div>
              </div>
            </div>
          </div>

        </div>


        {/* ========================================================================= */}
        {/* SENSOR 03: IIRS (Imaging Infrared Spectrometer) */}
        {/* ========================================================================= */}
        <div id="sensor-iirs" className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-start pt-12 border-t border-subtle">
          
          {/* Left Column: Hyperspectral Image Crop & Controlled Spectral Channels */}
          <div className="lg:col-span-7 space-y-4">
            <div className="relative rounded-xl overflow-hidden border border-[#B7B5AE]/40 bg-[#0D1116] shadow-2xl group">
              <div className="h-[360px] sm:h-[420px] md:h-[480px] w-full overflow-hidden relative">
                <img
                  src={iirsImg}
                  alt="IIRS Mineralogical Hyperspectral Regolith Data"
                  className="w-full h-full object-cover filter contrast-120 brightness-90 group-hover:scale-105 transition-transform duration-700 ease-out"
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#07090C] via-transparent to-transparent"></div>

                {/* Spectral Band Selector Overlay (Scientifically subtle) */}
                <div className="absolute top-6 left-6 flex items-center gap-2 bg-[#07090C]/90 backdrop-blur-md p-1.5 rounded border border-subtle font-mono-tech text-[10px]">
                  <button
                    onClick={() => setActiveIirsBand('mineral')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      activeIirsBand === 'mineral'
                        ? 'bg-[#D6C38B] text-[#07090C] font-bold'
                        : 'text-[#8B908F] hover:text-[#E7E3D9]'
                    }`}
                  >
                    1.0 µm PYROXENE
                  </button>
                  <button
                    onClick={() => setActiveIirsBand('water')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      activeIirsBand === 'water'
                        ? 'bg-[#D6C38B] text-[#07090C] font-bold'
                        : 'text-[#8B908F] hover:text-[#E7E3D9]'
                    }`}
                  >
                    2.8–3.0 µm OH/H₂O
                  </button>
                  <button
                    onClick={() => setActiveIirsBand('continuum')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      activeIirsBand === 'continuum'
                        ? 'bg-[#D6C38B] text-[#07090C] font-bold'
                        : 'text-[#8B908F] hover:text-[#E7E3D9]'
                    }`}
                  >
                    CONTINUUM (0.8–5.0 µm)
                  </button>
                </div>

                {/* Spectral Signature Readout */}
                <div className="absolute bottom-6 left-6 right-6 sm:right-auto sm:w-80 bg-[#07090C]/95 backdrop-blur-md p-3.5 rounded-lg border border-[#B7B5AE]/40 font-mono-tech text-xs space-y-2">
                  <div className="flex items-center justify-between text-[#B7B5AE] font-bold text-[10px]">
                    <span>256 CONTIGUOUS INFRARED CHANNELS</span>
                    <span className="text-[#D6C38B]">~15 nm FWHM</span>
                  </div>
                  <div className="text-[11px] text-[#E7E3D9]">
                    {activeIirsBand === 'mineral' && 'Maps olivine and pyroxene crystal field absorption bands across the lunar crust.'}
                    {activeIirsBand === 'water' && 'Detects 2.8–3.0 µm hydroxyl (OH) and adsorbed water absorption features in polar regolith.'}
                    {activeIirsBand === 'continuum' && 'Spans full NIR to thermal infrared spectrum (0.8 to 5.0 µm) for thermal albedo unmixing.'}
                  </div>
                </div>

                {/* Scale reference bar */}
                <div className="absolute bottom-6 right-6 hidden sm:flex flex-col items-end font-mono-tech text-[9px] text-[#B7B5AE]">
                  <div className="w-24 h-[2px] bg-[#B7B5AE] mb-1"></div>
                  <span>10 KILOMETERS</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Narrative & Technical Specifications */}
          <div className="lg:col-span-5 space-y-6 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-baseline gap-3">
                <span className="font-headline text-3xl md:text-4xl text-[#B7B5AE] font-bold">03</span>
                <div>
                  <span className="font-mono-tech text-xs text-[#8B908F] uppercase tracking-widest block font-medium">
                    HYPERSPECTRAL INFRARED SENSOR
                  </span>
                  <h3 className="font-headline text-2xl sm:text-3xl font-bold uppercase text-[#E7E3D9] tracking-tight">
                    IIRS
                  </h3>
                </div>
              </div>

              <div className="font-serif-italic text-lg text-[#B7B5AE]">
                Imaging Infrared Spectrometer
              </div>

              {/* Exact user requested resolution and spectral bands */}
              <div className="grid grid-cols-2 gap-3 font-mono-tech text-xs">
                <div className="bg-[#0D1116] p-3 rounded border border-subtle">
                  <span className="text-[#8B908F] text-[10px] uppercase block">Resolution:</span>
                  <span className="text-[#B7B5AE] text-sm font-bold">80 m</span>
                </div>
                <div className="bg-[#0D1116] p-3 rounded border border-subtle">
                  <span className="text-[#8B908F] text-[10px] uppercase block">Spectral Bands:</span>
                  <span className="text-[#D6C38B] text-sm font-bold">256</span>
                </div>
              </div>

              {/* Exact user requested description */}
              <p className="font-sans text-sm md:text-base text-[#E7E3D9] leading-relaxed border-l-2 border-[#B7B5AE] pl-4">
                Hyperspectral imagery containing information across hundreds of spectral bands beyond visible light.
              </p>
            </div>

            {/* Scientific Architecture Specs */}
            <div className="pt-4 border-t border-subtle grid grid-cols-2 gap-4 font-mono-tech text-xs">
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SPECTRAL RANGE</span>
                <span className="text-[#E7E3D9]">0.8 – 5.0 µm</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SPECTRAL RESOLUTION</span>
                <span className="text-[#E7E3D9]">~15 nm contiguous</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">SWATH WIDTH</span>
                <span className="text-[#E7E3D9]">20.0 km</span>
              </div>
              <div className="space-y-1">
                <span className="text-[#8B908F] text-[9px] uppercase block">PRIMARY SCIENTIFIC APPLICATION</span>
                <span className="text-[#D6C38B]">Mineralogy & Polar Hydroxyl Detection</span>
              </div>
            </div>
          </div>

        </div>

        {/* Sensors Page Bottom Bar with elegant NEXT button navigating to SCIENCE */}
        <div className="pt-10 flex items-center justify-between border-t border-subtle">
          <div className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-widest hidden sm:block">
            CHANDRAYAAN • SECTION 03 / 05 — SENSOR SUITE
          </div>

          <button
            id="sensors-next-button"
            onClick={onNext}
            className="group inline-flex items-center gap-3 text-xs md:text-sm font-sans font-semibold tracking-[0.18em] text-[#E7E3D9] hover:text-[#D6C38B] py-2 px-4 rounded-full border border-subtle hover:border-[#D6C38B]/50 bg-[#0D1116]/80 hover:bg-[#0D1116] transition-all ml-auto focus:outline-none cursor-pointer"
            aria-label="Navigate to Science Applications page"
          >
            <span>NEXT: SCIENCE</span>
            <ArrowRight className="w-4 h-4 text-[#D6C38B] group-hover:translate-x-1.5 transition-transform duration-300" />
          </button>
        </div>

      </div>
    </section>
  );
};
