// sih-dashboard/src/components/quantum/BenchmarkBoard.tsx
import { useEffect, useState, type FC } from 'react';
import { Award } from 'lucide-react';
import { fetchLatestBenchmarks } from '../../services/quantumApi';

export const BenchmarkBoard: FC = () => {
  const [benchmarks, setBenchmarks] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchLatestBenchmarks()
      .then((data) => setBenchmarks(data))
      .catch((err) => console.error('Benchmark fetch error:', err))
      .finally(() => setIsLoading(false));
  }, []);

  if (isLoading) {
    return (
      <div className="p-8 text-center text-xs text-zinc-400 font-mono">
        Loading 12-condition benchmark evaluation matrix...
      </div>
    );
  }

  const modules = benchmarks?.modules || {};

  return (
    <div className="w-full bg-zinc-900/80 border border-white/10 rounded-xl overflow-hidden shadow-xl backdrop-blur-xl mb-6">
      <div className="px-5 py-3.5 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Award className="text-amber-400" size={17} />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Benchmark Matrix: 12-Condition Evaluation vs Classical Twins
          </h3>
        </div>
        <span className="text-[11px] font-mono text-zinc-400">
          Seeds: 10 per condition | Promotion Gate Status
        </span>
      </div>

      <div className="p-5 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-white/10 text-zinc-400 font-mono text-[11px] uppercase tracking-wider">
              <th className="pb-3 pl-2">Module ID</th>
              <th className="pb-3">Target Layer</th>
              <th className="pb-3">Benchmark Metric</th>
              <th className="pb-3">Quantum Output</th>
              <th className="pb-3">Classical Twin</th>
              <th className="pb-3">Algorithmic Twin</th>
              <th className="pb-3 pr-2 text-right">Promotion Gate</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 font-mono">
            {Object.entries(modules).map(([mid, data]: [string, any]) => (
              <tr key={mid} className="hover:bg-white/5 transition-colors">
                <td className="py-3 pl-2 font-bold text-cyan-300">{mid}</td>
                <td className="py-3 text-zinc-400 text-[11px]">
                  {mid.startsWith('Q0')
                    ? 'L2 Matcher'
                    : mid.startsWith('Q1')
                    ? 'L3 Selection'
                    : mid.startsWith('Q2')
                    ? 'L1.5 / L4'
                    : mid.startsWith('Q3')
                    ? 'L4 Verification'
                    : mid.startsWith('Q4')
                    ? 'L5 Refinement'
                    : mid.startsWith('Q5')
                    ? 'L1 Imaging'
                    : 'L6 Security'}
                </td>
                <td className="py-3 text-zinc-300">{data.metric}</td>
                <td className="py-3 font-semibold text-white">{data.quantum}</td>
                <td className="py-3 text-zinc-300">{data.classical_twin}</td>
                <td className="py-3 text-zinc-400">{data.twin_name}</td>
                <td className="py-3 pr-2 text-right">
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                      mid === 'Q7_signing'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-zinc-800 text-zinc-400 border border-white/10'
                    }`}
                  >
                    {mid === 'Q7_signing' ? 'Active' : 'Shadow Mode'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
