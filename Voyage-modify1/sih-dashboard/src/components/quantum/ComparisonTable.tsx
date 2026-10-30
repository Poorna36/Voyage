// sih-dashboard/src/components/quantum/ComparisonTable.tsx
import type { FC } from 'react';
import { AlertCircle, Clock, Zap } from 'lucide-react';
import type { QuantumRunRecord } from '../../types/quantum';

interface ComparisonTableProps {
  record: QuantumRunRecord;
}

export const ComparisonTable: FC<ComparisonTableProps> = ({ record }) => {
  const qMetrics = record.metrics || {};
  const cTwin = record.classical_twin || {};
  const fallback = record.fallback || { used: false };

  return (
    <div className="w-full bg-zinc-900/80 border border-white/10 rounded-xl overflow-hidden shadow-xl backdrop-blur-xl mb-6">
      <div className="px-5 py-3.5 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Zap className="text-cyan-400" size={16} />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Paired Comparison: Quantum vs Classical Twin
          </h3>
        </div>
        <span className="text-[11px] font-mono text-zinc-400">
          Run ID: <span className="text-cyan-300 font-bold">{record.run_id}</span> | Seed:{' '}
          <span className="text-white">{record.seed}</span>
        </span>
      </div>

      <div className="p-5">
        {fallback.used && (
          <div className="mb-4 p-3 rounded-lg bg-orange-500/10 border border-orange-500/30 flex items-center gap-2 text-xs text-orange-300">
            <AlertCircle size={15} className="shrink-0" />
            <span>
              <strong>Fallback Activated:</strong> {fallback.reason || 'Solver fell back to classical baseline'}
            </span>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Quantum Side */}
          <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-500/30">
            <div className="flex items-center justify-between mb-3 border-b border-cyan-500/20 pb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                Quantum Module ({record.module})
              </span>
              <span className="text-[11px] font-mono text-cyan-300/80 bg-cyan-500/10 px-2 py-0.5 rounded">
                Backend: {record.backend}
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-zinc-400">Execution Mode</span>
                <span className="font-semibold text-white uppercase">{record.mode}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-zinc-400">Wall-Clock Runtime</span>
                <span className="font-mono text-cyan-300 flex items-center gap-1">
                  <Clock size={12} /> {record.runtime_s.toFixed(3)}s
                </span>
              </div>

              {/* Dynamic Module Metrics */}
              {Object.entries(qMetrics).map(([key, val]) => (
                <div key={key} className="flex justify-between items-center py-1 border-b border-white/5">
                  <span className="text-zinc-400 capitalize">{key.replace(/_/g, ' ')}</span>
                  <span className="font-mono font-semibold text-white">
                    {typeof val === 'number'
                      ? Number.isInteger(val)
                        ? val
                        : val.toFixed(4)
                      : typeof val === 'boolean'
                      ? val ? 'Yes' : 'No'
                      : Array.isArray(val)
                      ? `[${val.length} items]`
                      : String(val)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Classical Twin Side */}
          <div className="p-4 rounded-xl bg-zinc-800/40 border border-white/10">
            <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-zinc-400" />
                Classical Twin ({cTwin.name || 'Baseline'})
              </span>
              <span className="text-[11px] font-mono text-zinc-400 bg-white/5 px-2 py-0.5 rounded">
                Same Seed & Input
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-zinc-400">Baseline Algorithm</span>
                <span className="font-semibold text-white">{cTwin.name || 'Standard Baseline'}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-zinc-400">Wall-Clock Runtime</span>
                <span className="font-mono text-zinc-300 flex items-center gap-1">
                  <Clock size={12} /> {(cTwin.runtime_s || 0.001).toFixed(4)}s
                </span>
              </div>

              {/* Dynamic Classical Metrics */}
              {Object.entries(cTwin)
                .filter(([k]) => k !== 'name' && k !== 'runtime_s')
                .map(([key, val]) => (
                  <div key={key} className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-zinc-400 capitalize">{key.replace(/_/g, ' ')}</span>
                    <span className="font-mono font-semibold text-white">
                      {typeof val === 'number'
                        ? Number.isInteger(val)
                          ? val
                          : val.toFixed(4)
                        : String(val)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
