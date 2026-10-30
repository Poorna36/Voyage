// sih-dashboard/src/components/quantum/HonestyBanner.tsx
import type { FC } from 'react';
import { AlertTriangle, Cpu } from 'lucide-react';

interface HonestyBannerProps {
  backendKind?: string;
}

export const HonestyBanner: FC<HonestyBannerProps> = ({ backendKind = 'simulator' }) => {
  const isHardware = backendKind === 'hardware';

  return (
    <div className="w-full bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 rounded-xl p-3.5 sm:p-4 mb-5 shadow-lg backdrop-blur-md">
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400 shrink-0 mt-0.5">
          {isHardware ? <Cpu size={18} /> : <AlertTriangle size={18} />}
        </div>
        <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed">
          <div className="font-semibold text-amber-300 mb-0.5 flex items-center gap-2">
            <span>Rigorous Scientific Framing Notice</span>
            <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40">
              {isHardware ? 'Hardware QPU Target' : 'Classical CPU Simulation'}
            </span>
          </div>
          <p className="text-zinc-400 text-xs">
            These modules simulate quantum algorithms on a classical CPU unless a real hardware backend is explicitly
            selected. Speed comparisons in simulation reflect algorithmic complexity and oracle query counts (e.g.
            Grover&apos;s O(&radic;N) vs O(N)), not wall-clock speedup. Loading image raster data into quantum statevectors
            is classically bounded; every result is mirrored against its exact classical twin.
          </p>
        </div>
      </div>
    </div>
  );
};
