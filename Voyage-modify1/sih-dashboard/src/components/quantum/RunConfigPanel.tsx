// sih-dashboard/src/components/quantum/RunConfigPanel.tsx
import { useState, type FC } from 'react';
import { Play, Settings2, RefreshCw } from 'lucide-react';
import type { ModuleId, BackendInfo, QuantumRunRequest } from '../../types/quantum';

interface RunConfigPanelProps {
  selectedPairId: string;
  onRun: (request: QuantumRunRequest) => void;
  isRunning: boolean;
  backends: BackendInfo[];
}

const MODULES_LIST: Array<{ id: ModuleId; title: string; desc: string }> = [
  { id: 'Q0_filter', title: 'Q0: QUBO Matcher Consistency Filter', desc: 'Ising Hamiltonian geometric filter on raw candidate matches' },
  { id: 'Q1_selection', title: 'Q1: QAOA Spatial Subset Selection', desc: 'Per-cell QAOA cardinality & dispersion optimization' },
  { id: 'Q2_router', title: 'Q2a: Quantum Kernel Matcher Router', desc: '8-qubit feature map predicting optimal matcher from MSM vector' },
  { id: 'Q2_inlier', title: 'Q2b: Quantum Kernel Inlier Scorer', desc: '8-qubit feature map classifying matches as inlier/outlier' },
  { id: 'Q3_consensus', title: 'Q3: Grover Consensus Hypotheses Search', desc: 'O(sqrt(N)) amplitude amplification finding minimal 4-point homography' },
  { id: 'Q4_phase_corr', title: 'Q4: Gate-Level 2D QFT Phase Correlation', desc: 'Explicit gate-level QFT on amplitude-encoded tiles for sub-pixel shift' },
  { id: 'Q5_encoding', title: 'Q5: FRQI/NEQR Image Encoding & QHED', desc: 'Quantum image encodings & Hadamard edge detection demo' },
  { id: 'Q7_signing', title: 'Q7: Post-Quantum Cryptographic Signing', desc: 'Hybrid ML-DSA-65 + Ed25519 tamper-proof export signatures' },
];

export const RunConfigPanel: FC<RunConfigPanelProps> = ({
  selectedPairId,
  onRun,
  isRunning,
  backends,
}) => {
  const [selectedModule, setSelectedModule] = useState<ModuleId>('Q1_selection');
  const [selectedBackend, setSelectedBackend] = useState<string>('numpy_statevector');
  const [mode, setMode] = useState<'shadow' | 'active'>('shadow');
  const [seed, setSeed] = useState<number>(42);

  const handleExecute = () => {
    onRun({
      pair_id: selectedPairId,
      module: selectedModule,
      backend: selectedBackend,
      mode,
      seed,
    });
  };

  return (
    <div className="w-full bg-zinc-900/90 border border-white/10 rounded-xl p-5 mb-6 shadow-xl backdrop-blur-xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <Settings2 className="text-cyan-400" size={17} />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Quantum Lab Job Configuration
          </h3>
        </div>
        <div className="text-xs text-zinc-400 font-mono">
          Target Pair: <strong className="text-cyan-300 font-bold">{selectedPairId}</strong>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        {/* Module Picker */}
        <div>
          <label className="block text-xs font-medium text-zinc-300 mb-1.5">
            Quantum Module (Q0–Q7)
          </label>
          <select
            value={selectedModule}
            onChange={(e) => setSelectedModule(e.target.value as ModuleId)}
            className="w-full px-3 py-2 rounded-lg bg-black/60 border border-white/15 text-xs text-white focus:outline-none focus:border-cyan-400"
          >
            {MODULES_LIST.map((m) => (
              <option key={m.id} value={m.id}>
                {m.title}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-zinc-500 mt-1">
            {MODULES_LIST.find((m) => m.id === selectedModule)?.desc}
          </p>
        </div>

        {/* Backend Picker */}
        <div>
          <label className="block text-xs font-medium text-zinc-300 mb-1.5 flex items-center justify-between">
            <span>Execution Backend</span>
            <span className="text-[10px] text-zinc-400">Q6 Abstraction</span>
          </label>
          <select
            value={selectedBackend}
            onChange={(e) => setSelectedBackend(e.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-black/60 border border-white/15 text-xs text-white focus:outline-none focus:border-cyan-400"
          >
            {backends.map((b) => (
              <option key={b.id} value={b.id}>
                {b.id} ({b.kind}, max {b.max_qubits}q) {b.available ? '✓' : '(offline)'}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-zinc-500 mt-1">
            NumPy statevector default (up to 16 qubits on CPU).
          </p>
        </div>

        {/* Execution Mode & Seed */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs font-medium text-zinc-300 mb-1.5">Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as any)}
              className="w-full px-3 py-2 rounded-lg bg-black/60 border border-white/15 text-xs text-white focus:outline-none focus:border-cyan-400"
            >
              <option value="shadow">Shadow (Safe / Log Only)</option>
              <option value="active">Active (Requires Gate)</option>
            </select>
          </div>

          <div className="w-24">
            <label className="block text-xs font-medium text-zinc-300 mb-1.5">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
              className="w-full px-3 py-2 rounded-lg bg-black/60 border border-white/15 text-xs text-white font-mono focus:outline-none focus:border-cyan-400"
            />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-end">
        <button
          onClick={handleExecute}
          disabled={isRunning}
          className={`px-6 py-2.5 rounded-full font-bold text-xs uppercase tracking-wider transition-all flex items-center gap-2 shadow-lg cursor-pointer ${
            isRunning
              ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
              : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-[0_0_20px_rgba(6,182,212,0.4)] active:scale-95'
          }`}
        >
          {isRunning ? (
            <>
              <RefreshCw size={14} className="animate-spin" />
              <span>Simulating Quantum Circuit...</span>
            </>
          ) : (
            <>
              <Play size={14} />
              <span>Execute {selectedModule} Job</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
