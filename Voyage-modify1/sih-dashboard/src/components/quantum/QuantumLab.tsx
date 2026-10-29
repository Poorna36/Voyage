// sih-dashboard/src/components/quantum/QuantumLab.tsx
import { useState, useEffect, type FC } from 'react';
import {
  Atom,
  Cpu,
  Layers,
  Shield,
  Activity,
  Award,
  Sparkles,
  GitCommit,
  Radio,
} from 'lucide-react';
import { HonestyBanner } from './HonestyBanner';
import { RunConfigPanel } from './RunConfigPanel';
import { ComparisonTable } from './ComparisonTable';
import { CircuitSummary } from './CircuitSummary';
import { SecurityPanel } from './SecurityPanel';
import { BenchmarkBoard } from './BenchmarkBoard';
import { useQuantumRun } from '../../hooks/useQuantumRun';
import { fetchQuantumStatus, fetchCircuitData } from '../../services/quantumApi';
import type { QuantumStatusResponse, BackendInfo, CircuitData } from '../../types/quantum';

interface QuantumLabProps {
  selectedPairId?: string;
}

export const QuantumLab: FC<QuantumLabProps> = ({ selectedPairId = 'boguslawsky' }) => {
  const [activeTab, setActiveTab] = useState<
    'overview' | 'matching' | 'selection' | 'classifier' | 'consensus' | 'fourier' | 'imaging' | 'security' | 'benchmarks'
  >('overview');

  const [statusData, setStatusData] = useState<QuantumStatusResponse | null>(null);
  const [circuitData, setCircuitData] = useState<CircuitData | null>(null);

  const { isRunning, currentRecord, errorMessage, triggerRun } = useQuantumRun();

  useEffect(() => {
    fetchQuantumStatus()
      .then((data) => setStatusData(data))
      .catch((err) => console.error('Failed to load quantum status:', err));
  }, []);

  // Fetch circuit structure when switching tabs
  useEffect(() => {
    let mod = 'Q1_selection';
    if (activeTab === 'matching') mod = 'Q0_filter';
    if (activeTab === 'classifier') mod = 'Q2_router';
    if (activeTab === 'consensus') mod = 'Q3_consensus';
    if (activeTab === 'fourier') mod = 'Q4_phase_corr';

    fetchCircuitData(mod)
      .then((data) => setCircuitData(data))
      .catch((err) => console.error('Circuit fetch error:', err));
  }, [activeTab]);

  const backends: BackendInfo[] = statusData?.backends || [
    { id: 'numpy_statevector', kind: 'simulator', max_qubits: 16, available: true },
    { id: 'qiskit_aer', kind: 'simulator', max_qubits: 24, available: false, reason: 'Optional extra' },
    { id: 'dwave_neal', kind: 'annealer', max_qubits: 2048, available: false },
    { id: 'ibm_runtime', kind: 'hardware', max_qubits: 127, available: false },
    { id: 'dwave_leap', kind: 'hardware', max_qubits: 5000, available: false },
  ];

  return (
    <div className="w-full h-full flex flex-col font-sans select-none overflow-y-auto sidebar-scroll p-4 md:p-8">
      {/* Header Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 pb-4 border-b border-white/10">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-[0_0_15px_rgba(6,182,212,0.5)]">
              <Atom size={20} className="animate-spin-slow" />
            </div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
              Voyage Quantum Lab & Simulation Deck
            </h1>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Hybrid Classical–Quantum Lunar Registration Engine (PS-26166) | Q0–Q7 Modules
          </p>
        </div>

        {/* Global status badge */}
        <div className="flex items-center gap-3 text-xs">
          <span className="px-3 py-1.5 rounded-full bg-cyan-950/40 border border-cyan-500/30 text-cyan-300 font-mono flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>Mode: {statusData?.global_mode?.toUpperCase() || 'SHADOW'}</span>
          </span>
          <span className="px-3 py-1.5 rounded-full bg-zinc-800 border border-white/10 text-zinc-300 font-mono">
            Statevector Limit: 16 Qubits
          </span>
        </div>
      </div>

      {/* Mandatory Honesty Notice Banner */}
      <HonestyBanner backendKind={currentRecord?.backend} />

      {/* Navigation Sub-Tabs */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-3 mb-6 border-b border-white/10 text-xs font-medium">
        {[
          { id: 'overview', label: 'Overview & Status', icon: Layers },
          { id: 'matching', label: 'Matching (Q0)', icon: GitCommit },
          { id: 'selection', label: 'Selection (Q1)', icon: Sparkles },
          { id: 'classifier', label: 'Kernel Classifier (Q2)', icon: Activity },
          { id: 'consensus', label: 'Grover Consensus (Q3)', icon: Atom },
          { id: 'fourier', label: 'QFT Correlation (Q4)', icon: Radio },
          { id: 'imaging', label: 'Quantum Imaging (Q5)', icon: Cpu },
          { id: 'security', label: 'Post-Quantum (Q7)', icon: Shield },
          { id: 'benchmarks', label: 'Benchmark Board', icon: Award },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition-all shrink-0 cursor-pointer ${
                isActive
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <Icon size={14} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Configuration Panel */}
      <RunConfigPanel
        selectedPairId={selectedPairId}
        onRun={triggerRun}
        isRunning={isRunning}
        backends={backends}
      />

      {/* Error Display */}
      {errorMessage && (
        <div className="mb-6 p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-200 text-xs">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}

      {/* Live Run Result Comparison */}
      {currentRecord && <ComparisonTable record={currentRecord} />}

      {/* Interactive Circuit Summary */}
      {circuitData && (
        <CircuitSummary
          qubits={circuitData.qubits}
          depth={circuitData.total_gates}
          gateCounts={circuitData.gate_counts}
          gateSequence={circuitData.gate_sequence}
        />
      )}

      {/* Tab Specific Extra Panes */}
      {activeTab === 'security' && <SecurityPanel />}
      {activeTab === 'benchmarks' && <BenchmarkBoard />}

      {/* Module Overview Cards Grid (when on overview tab) */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
          {[
            { id: 'Q0', title: 'Q0: Matcher & Filter', layer: 'L2 Correspondence', desc: 'Ising QUBO consistency filter rejecting topological outliers.' },
            { id: 'Q1', title: 'Q1: QAOA Selection', layer: 'L3 Spatial Selection', desc: '8x8 cell QAOA selecting confident spread-out keypoint subsets.' },
            { id: 'Q2', title: 'Q2: Kernel Classifier', layer: 'L1.5 Router & L4 Inlier', desc: '8-qubit quantum feature map predicting optimal matchers and inliers.' },
            { id: 'Q3', title: 'Q3: Grover Consensus', layer: 'L4 Verification', desc: 'O(sqrt(N)) amplitude amplification finding minimal 4-point homographies.' },
            { id: 'Q4', title: 'Q4: QFT Refinement', layer: 'L5 Sub-Pixel Phase', desc: 'Explicit gate-level 2D Quantum Fourier Transform on image tiles.' },
            { id: 'Q5', title: 'Q5: Image Encoding', layer: 'L1 Preprocessing', desc: 'FRQI, NEQR basis encodings and Quantum Hadamard Edge Detection.' },
            { id: 'Q6', title: 'Q6: Backend Layer', layer: 'Cross-Cutting', desc: 'Statevector simulator, Qiskit Aer, D-Wave Neal, and hardware adapters.' },
            { id: 'Q7', title: 'Q7: Post-Quantum', layer: 'L6 Security & Export', desc: 'Hybrid ML-DSA-65 + Ed25519 signing protecting products against Q-day.' },
          ].map((card) => (
            <div
              key={card.id}
              className="p-4 rounded-xl bg-zinc-900/60 border border-white/10 hover:border-cyan-500/30 transition-all shadow-md"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold font-mono text-cyan-400">{card.id}</span>
                <span className="text-[10px] text-zinc-500 font-mono">{card.layer}</span>
              </div>
              <h4 className="text-xs font-semibold text-white mb-1">{card.title}</h4>
              <p className="text-[11px] text-zinc-400 leading-relaxed">{card.desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default QuantumLab;
