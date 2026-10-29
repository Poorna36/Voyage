// sih-dashboard/src/components/quantum/CircuitSummary.tsx
import type { FC } from 'react';
import { Layers, Activity } from 'lucide-react';

interface CircuitSummaryProps {
  qubits?: number;
  depth?: number;
  gateCounts?: Record<string, number>;
  gateSequence?: Array<{ gate: string; target?: number; targets?: number[]; phi?: number; param?: number }>;
}

export const CircuitSummary: FC<CircuitSummaryProps> = ({
  qubits = 8,
  depth = 12,
  gateCounts = {},
  gateSequence = [],
}) => {
  return (
    <div className="w-full bg-zinc-900/80 border border-white/10 rounded-xl overflow-hidden shadow-xl backdrop-blur-xl mb-6">
      <div className="px-5 py-3.5 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="text-purple-400" size={16} />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Quantum Circuit Architecture & Complexity
          </h3>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-zinc-400">
            Qubits: <strong className="text-cyan-300">{qubits}</strong>
          </span>
          <span className="text-zinc-400">
            Depth: <strong className="text-purple-300">{depth}</strong>
          </span>
        </div>
      </div>

      <div className="p-5">
        {/* Gate Counts Pills */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <span className="text-xs text-zinc-400 flex items-center gap-1 mr-2">
            <Layers size={13} /> Gates:
          </span>
          {Object.entries(gateCounts).length > 0 ? (
            Object.entries(gateCounts).map(([gate, count]) => (
              <span
                key={gate}
                className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-purple-950/40 border border-purple-500/30 text-purple-200"
              >
                <strong>{gate}</strong>: {count}
              </span>
            ))
          ) : (
            <span className="text-xs text-zinc-500">Hadamard, Controlled-Phase, Entanglement</span>
          )}
        </div>

        {/* Visual Gate Flow Diagram */}
        <div className="p-4 rounded-lg bg-black/60 border border-white/5 overflow-x-auto">
          <div className="text-[11px] font-mono text-zinc-400 mb-2 flex items-center justify-between">
            <span>Gate Sequence Timeline (First {Math.min(gateSequence.length, 16)} Operations)</span>
            <span className="text-[10px] text-zinc-500">Qubit Wire View</span>
          </div>

          <div className="flex items-center gap-2 py-2 min-w-[500px]">
            {gateSequence.slice(0, 16).map((g, idx) => (
              <div
                key={idx}
                className="flex flex-col items-center justify-center p-2 rounded bg-zinc-800/80 border border-white/10 shrink-0 min-w-[56px] text-center"
              >
                <span className="text-[10px] font-bold text-cyan-400 uppercase">{g.gate}</span>
                <span className="text-[9px] font-mono text-zinc-400 mt-0.5">
                  {g.target !== undefined
                    ? `q[${g.target}]`
                    : g.targets
                    ? `q[${g.targets.join(',')}]`
                    : 'all'}
                </span>
                {g.phi !== undefined && (
                  <span className="text-[8px] font-mono text-amber-300/80 mt-0.5">&phi;={g.phi}</span>
                )}
              </div>
            ))}
            {gateSequence.length > 16 && (
              <div className="text-xs font-mono text-zinc-500 px-3">
                +{gateSequence.length - 16} more
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
