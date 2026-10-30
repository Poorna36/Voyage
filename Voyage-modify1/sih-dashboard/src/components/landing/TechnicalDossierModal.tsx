import React from 'react';
import { X, BookOpen, Calculator, Shield, Atom, Layers } from 'lucide-react';

interface TechnicalDossierModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLaunchWorkbench?: () => void;
}

export const TechnicalDossierModal: React.FC<TechnicalDossierModalProps> = ({
  isOpen,
  onClose,
  onLaunchWorkbench,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-xl animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-[#0A0D12] border border-[#D6C38B]/40 rounded-2xl shadow-[0_20px_60px_rgba(0,0,0,0.95)] flex flex-col overflow-hidden text-[#E7E3D9]">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-subtle bg-[#0D1117]/90">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#D6C38B]/15 border border-[#D6C38B]/30 flex items-center justify-center text-[#D6C38B]">
              <BookOpen size={16} />
            </div>
            <div>
              <span className="font-mono-tech text-[10px] text-[#D6C38B] uppercase tracking-wider block font-bold">
                SIH26166 COMPREHENSIVE ARCHITECTURE
              </span>
              <h2 className="font-headline text-lg sm:text-xl text-white uppercase tracking-wide">
                ECLIPSE SCIENTIFIC & MATHEMATICAL DOSSIER
              </h2>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            aria-label="Close Dossier"
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-8 sidebar-scroll text-xs sm:text-sm">
          
          {/* Section 1: The Core Scientific Challenge */}
          <section className="space-y-3">
            <div className="flex items-center gap-2 text-[#D6C38B] font-mono-tech text-xs uppercase font-bold">
              <Atom size={14} />
              <span>01. The Lunar Registration Problem</span>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Standard earth-observation algorithms break down in the lunar polar environment. Chandrayaan-2 orbits at 100 km altitude over rugged polar highlands characterized by:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
              <div className="p-3 rounded-xl bg-black/40 border border-subtle">
                <span className="text-[#D6C38B] font-bold block mb-1">320× Scale Jump</span>
                <span className="text-[11.5px] text-slate-400">
                  OHRC at 0.25 m/px resolves rocks, while IIRS at 80 m/px spans whole impact craters within a single pixel.
                </span>
              </div>
              <div className="p-3 rounded-xl bg-black/40 border border-subtle">
                <span className="text-[#D6C38B] font-bold block mb-1">Grazing Solar Phase</span>
                <span className="text-[11.5px] text-slate-400">
                  Incidence angles exceed 75°–88° near the poles, creating long black shadows that change orientation radically between passes.
                </span>
              </div>
              <div className="p-3 rounded-xl bg-black/40 border border-subtle">
                <span className="text-[#D6C38B] font-bold block mb-1">Cross-Modal Radiometry</span>
                <span className="text-[11.5px] text-slate-400">
                  Monochromatic CCD pixels (OHRC) do not directly correlate with 256-band continuous hyperspectral radiance (IIRS).
                </span>
              </div>
            </div>
          </section>

          {/* Section 2: Mathematical Formulations */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-[#D6C38B] font-mono-tech text-xs uppercase font-bold">
              <Calculator size={14} />
              <span>02. Mathematical Foundations</span>
            </div>

            {/* Formula 1: Log-Gabor Phase Congruency */}
            <div className="p-4 rounded-xl bg-[#080B0F] border border-[#D6C38B]/25 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono-tech text-[10.5px] text-[#D6C38B] font-bold">
                  A. Radiation-Invariant Phase Congruency (RIFT-2)
                </span>
                <span className="text-[10px] font-mono-tech text-slate-400">Frequency Domain</span>
              </div>
              <div className="p-3 rounded-lg bg-black/60 font-mono-tech text-xs text-[#E7E3D9] overflow-x-auto text-center">
                PC(x,y) = ∑θ ∑n W_θ(x,y) ⌊A_nθ(x,y) ΔΦ_nθ(x,y) - T⌋ / [ ε + ∑θ ∑n A_nθ(x,y) ]
              </div>
              <p className="text-[11.5px] text-slate-400">
                Phase congruency calculates structural edges where Fourier frequency components are in phase, making features entirely independent of image brightness, contrast, and solar incidence angles.
              </p>
            </div>

            {/* Formula 2: Sub-Pixel Paraboloid Surface Fitting */}
            <div className="p-4 rounded-xl bg-[#080B0F] border border-[#D6C38B]/25 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono-tech text-[10.5px] text-[#D6C38B] font-bold">
                  B. Sub-Pixel Paraboloid Peak Interpolation
                </span>
                <span className="text-[10px] font-mono-tech text-slate-400">&lt; 0.28 px RMSE</span>
              </div>
              <div className="p-3 rounded-lg bg-black/60 font-mono-tech text-xs text-[#E7E3D9] overflow-x-auto text-center">
                C(u,v) ≈ c₀ + c₁ u + c₂ v + c₃ u² + c₄ uv + c₅ v²  ⟹  [u*, v*]ᵀ = -½ [c₃, ½c₄; ½c₄, c₅]⁻¹ [c₁, c₂]ᵀ
              </div>
              <p className="text-[11.5px] text-slate-400">
                By fitting a continuous bivariate second-order paraboloid across the 3×3 normalized cross-correlation peak, ECLIPSE resolves tie points to quarter-pixel precision.
              </p>
            </div>

            {/* Formula 3: MAGSAC++ Scoring */}
            <div className="p-4 rounded-xl bg-[#080B0F] border border-[#D6C38B]/25 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono-tech text-[10.5px] text-[#D6C38B] font-bold">
                  C. MAGSAC++ Marginalized Outlier Rejection
                </span>
                <span className="text-[10px] font-mono-tech text-slate-400">Robust Estimator</span>
              </div>
              <div className="p-3 rounded-lg bg-black/60 font-mono-tech text-xs text-[#E7E3D9] overflow-x-auto text-center">
                Score(H) = ∑ᵢ ∫₀^σ_max L(rᵢ(H), σ) P(σ) dσ
              </div>
              <p className="text-[11.5px] text-slate-400">
                Rather than using an arbitrary threshold like standard RANSAC, MAGSAC++ marginalizes over all noise levels σ, preventing false model rejections on undulating crater terrains.
              </p>
            </div>
          </section>

          {/* Section 3: Sensor Payload Parameters */}
          <section className="space-y-3">
            <div className="flex items-center gap-2 text-[#D6C38B] font-mono-tech text-xs uppercase font-bold">
              <Layers size={14} />
              <span>03. Chandrayaan-2 Payload Specifications</span>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono-tech text-[11px] border border-subtle">
                <thead className="bg-[#0D1117] text-[#D6C38B]">
                  <tr className="border-b border-subtle">
                    <th className="p-2.5">Payload</th>
                    <th className="p-2.5">Spatial Resolution</th>
                    <th className="p-2.5">Spectral Coverage</th>
                    <th className="p-2.5">Swath Width</th>
                    <th className="p-2.5">Core Function</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-subtle/50 text-slate-300">
                  <tr>
                    <td className="p-2.5 font-bold text-white">OHRC</td>
                    <td className="p-2.5 text-[#D6C38B]">0.25 m / px</td>
                    <td className="p-2.5">450 – 900 nm (Panchromatic)</td>
                    <td className="p-2.5">3.0 km</td>
                    <td className="p-2.5">Hazard identification & micro-craters</td>
                  </tr>
                  <tr>
                    <td className="p-2.5 font-bold text-white">TMC-2</td>
                    <td className="p-2.5 text-[#D6C38B]">5.0 m / px</td>
                    <td className="p-2.5">Triplet stereo (+26°, 0°, -26°)</td>
                    <td className="p-2.5">20.0 km</td>
                    <td className="p-2.5">3D Digital Elevation Models (DEM)</td>
                  </tr>
                  <tr>
                    <td className="p-2.5 font-bold text-white">IIRS</td>
                    <td className="p-2.5 text-[#D6C38B]">80.0 m / px</td>
                    <td className="p-2.5">256 bands (800 – 5000 nm)</td>
                    <td className="p-2.5">20.0 km</td>
                    <td className="p-2.5">Hydration & water-ice absorption</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Section 4: Data Standard & Validation */}
          <section className="space-y-3">
            <div className="flex items-center gap-2 text-[#D6C38B] font-mono-tech text-xs uppercase font-bold">
              <Shield size={14} />
              <span>04. PDS4 Compliance & Ground Truth Validation</span>
            </div>
            <p className="text-slate-300 leading-relaxed text-xs">
              Every data product processed by ECLIPSE complies with the Planetary Data System (PDS4) XML schema mandated by ISRO ISSDC and NASA planetary missions. Ground-truth cross-checks are validated against Lunar Reconnaissance Orbiter Camera (LROC NAC) calibrated GDRs and SELENE (Kaguya) stereo baselines.
            </p>
          </section>

        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-subtle bg-[#0D1117] flex items-center justify-between">
          <span className="font-mono-tech text-[10px] text-[#8B908F]">
            ISRO CHANDRAYAAN-2 • SIH26166 DOSSIER
          </span>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-xl border border-subtle hover:border-slate-500 text-slate-300 text-xs font-sans cursor-pointer transition-colors"
            >
              Close
            </button>
            {onLaunchWorkbench && (
              <button
                onClick={() => {
                  onClose();
                  onLaunchWorkbench();
                }}
                className="px-4 py-1.5 rounded-xl bg-[#D6C38B] hover:bg-[#FAF6EB] text-black font-sans font-bold text-xs tracking-wider cursor-pointer transition-all shadow-[0_0_12px_rgba(214,195,139,0.3)]"
              >
                Launch
              </button>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default TechnicalDossierModal;
