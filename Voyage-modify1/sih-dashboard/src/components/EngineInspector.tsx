import React, { useState, useRef } from 'react';
import {
  X, Cpu, Sliders, Database, Check, FolderOpen, Save,
  RefreshCw, Upload, FolderUp, FileText
} from 'lucide-react';
import type {
  PipelineOptions,
  MatcherParameters,
  UploadedFile
} from '../types';
import {
  PairSummary as DatasetPair,
  DatasetStats,
  uploadMissionFiles
} from '../services/api';

interface EngineInspectorProps {
  isOpen: boolean;
  onClose: () => void;
  options: PipelineOptions;
  onOptionsChange: (opts: PipelineOptions) => void;
  matcherParams: MatcherParameters;
  onUpdateMatcherParams: (params: MatcherParameters) => void;
  onSaveMatcherConfig: (params: any) => Promise<boolean>;
  pairs: DatasetPair[];
  stats: DatasetStats | null;
  selectedPairId?: string;
  onSelectPair?: (pairId: string) => void;
}

/**
 * Recursively scans all files and directories from drag-and-drop DataTransfer.
 */
const scanFilesFromDataTransfer = async (dataTransfer: DataTransfer): Promise<File[]> => {
  const files: File[] = [];
  const items = dataTransfer.items;

  if (items && items.length > 0) {
    const traverseEntry = async (entry: any): Promise<void> => {
      if (!entry) return;
      if (entry.isFile) {
        await new Promise<void>((resolve) => {
          entry.file(
            (file: File) => {
              if (entry.fullPath && entry.fullPath !== `/${file.name}`) {
                try {
                  Object.defineProperty(file, 'webkitRelativePath', {
                    value: entry.fullPath.replace(/^\//, ''),
                    writable: true,
                  });
                } catch {
                  // Ignore
                }
              }
              files.push(file);
              resolve();
            },
            () => resolve()
          );
        });
      } else if (entry.isDirectory) {
        const dirReader = entry.createReader();
        const readBatch = async (): Promise<any[]> => {
          return new Promise((resolve) => {
            dirReader.readEntries(
              (entries: any[]) => resolve(entries || []),
              () => resolve([])
            );
          });
        };
        let batch = await readBatch();
        while (batch && batch.length > 0) {
          for (const child of batch) {
            await traverseEntry(child);
          }
          batch = await readBatch();
        }
      }
    };

    const promises: Promise<void>[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.webkitGetAsEntry) {
        const entry = item.webkitGetAsEntry();
        if (entry) {
          promises.push(traverseEntry(entry));
          continue;
        }
      }
      const f = item.getAsFile();
      if (f) files.push(f);
    }
    await Promise.all(promises);
  } else if (dataTransfer.files && dataTransfer.files.length > 0) {
    files.push(...Array.from(dataTransfer.files));
  }
  return files;
};

export const EngineInspector: React.FC<EngineInspectorProps> = ({
  isOpen,
  onClose,
  options,
  onOptionsChange,
  matcherParams,
  onUpdateMatcherParams,
  onSaveMatcherConfig,
  pairs,
  stats,
  selectedPairId,
  onSelectPair,
}) => {
  const [activeTab, setActiveTab] = useState<'engine' | 'ingest'>('engine');
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [rawFiles, setRawFiles] = useState<File[]>([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [ingestSuccess, setIngestSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleSave = async () => {
    setIsSaving(true);
    setSaveSuccess(false);
    try {
      const ok = await onSaveMatcherConfig(matcherParams);
      if (ok) {
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 2500);
      }
    } finally {
      setIsSaving(false);
    }
  };

  const handleFilesAdded = (files: FileList | File[]) => {
    const arr = Array.from(files);
    setRawFiles((prev) => [...prev, ...arr]);
    const newFiles: UploadedFile[] = arr.map((f) => {
      const relPath = (f as any).webkitRelativePath || f.name;
      const lower = relPath.toLowerCase();
      return {
        name: relPath,
        size: f.size,
        sensor: lower.includes('ohr') ? 'OHRC' : lower.includes('iirs') ? 'IIRS' : 'TMC-2',
        status: 'ready',
      };
    });
    setUploadedFiles((prev) => [...prev, ...newFiles]);
    setIngestError(null);
  };

  const handleBatchIngest = async () => {
    if (rawFiles.length === 0) {
      setIngestError('Please select at least one mission file or folder to ingest.');
      return;
    }
    setIsIngesting(true);
    setIngestError(null);
    setIngestSuccess(null);
    try {
      const res = await uploadMissionFiles(rawFiles);
      if (res && res.status === 'success') {
        setIngestSuccess(res.message);
        setUploadedFiles([]);
        setRawFiles([]);
        if (onSelectPair && res.pair_id) {
          onSelectPair(res.pair_id);
        }
      } else {
        setIngestError(res?.message || 'Ingestion failed.');
      }
    } catch (err: any) {
      setIngestError(err?.message || 'Failed to ingest batch files. Ensure backend is running.');
    } finally {
      setIsIngesting(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    try {
      const extractedFiles = await scanFilesFromDataTransfer(e.dataTransfer);
      if (extractedFiles.length > 0) {
        handleFilesAdded(extractedFiles);
      }
    } catch {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFilesAdded(e.dataTransfer.files);
      }
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/50 backdrop-blur-sm animate-in fade-in duration-150 select-none font-sans"
      onClick={onClose}
      onPointerDown={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full sm:w-[440px] max-w-[92vw] h-full bg-[#0A0C10]/95 backdrop-blur-3xl border-l border-white/15 shadow-[0_0_80px_rgba(0,0,0,0.9)] flex flex-col animate-in slide-in-from-right duration-200"
      >
        {/* ── HEADER ── */}
        <div className="p-4 border-b border-white/10 shrink-0">
          <div className="flex items-center justify-between pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#2997FF] shadow-[0_0_8px_rgba(41,151,255,0.8)]" />
              <span className="text-xs font-bold uppercase tracking-wider text-white">
                Registration Engine & Data Inspector
              </span>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-full text-white/50 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            >
              <X size={15} />
            </button>
          </div>

          {/* Tab Switcher */}
          <div className="flex items-center p-1 bg-white/[0.04] border border-white/10 rounded-2xl">
            <button
              onClick={() => setActiveTab('engine')}
              className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                activeTab === 'engine'
                  ? 'bg-white text-black shadow-md'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              <Sliders size={13} />
              <span>Engine Tuning</span>
            </button>
            <button
              onClick={() => setActiveTab('ingest')}
              className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                activeTab === 'ingest'
                  ? 'bg-white text-black shadow-md'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              <Database size={13} />
              <span>Manifest & Ingest</span>
            </button>
          </div>
        </div>

        {/* ── TAB 1: REGISTRATION ENGINE TUNING ── */}
        {activeTab === 'engine' && (
          <div className="flex-1 overflow-y-auto p-4 space-y-4 sidebar-scroll">
            {/* Adaptive MSM Meta-Selector */}
            <div className="p-3 bg-white/[0.04] border border-white/10 rounded-2xl flex items-center justify-between">
              <div>
                <div className="font-bold text-xs text-white flex items-center gap-1.5">
                  <Cpu size={14} className="text-[#2997FF]" />
                  <span>Adaptive MSM Meta-Selector</span>
                </div>
                <div className="text-[10px] text-white/50 mt-0.5">
                  Trained Gradient Boosting model (models/msm_v1.pkl)
                </div>
              </div>
              <button
                onClick={() => onOptionsChange({ ...options, adaptiveMsm: !options.adaptiveMsm })}
                className={`w-11 h-6 rounded-full transition-colors relative cursor-pointer ${
                  options.adaptiveMsm ? 'bg-[#0071E3]' : 'bg-white/20'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-1 ${
                    options.adaptiveMsm ? 'left-6' : 'left-1'
                  }`}
                />
              </button>
            </div>

            {/* Algorithm Selection */}
            <div>
              <div className="text-[10px] font-bold text-white/40 uppercase tracking-wider mb-2">
                Primary Matcher Algorithm
              </div>
              <div className="space-y-1.5">
                {[
                  { id: 'sift', name: 'SIFT + MAGSAC++', desc: 'Scale-space Gaussian keypoints with robust homography' },
                  { id: 'rift2', name: 'RIFT2 (Phase Congruency)', desc: 'Invariant to extreme solar incidence & cross-sensor disparities' },
                  { id: 'lightglue', name: 'SuperPoint + LightGlue', desc: 'Deep learned keypoints & attention-based graph neural matching' },
                  { id: 'crater', name: 'Crater Ring Topology (YOLO)', desc: 'Geometric ellipse fitting for cratered polar terrain' },
                ].map((algo) => {
                  const isSelected = options.activeMatcher === algo.id;
                  return (
                    <div
                      key={algo.id}
                      onClick={() => onOptionsChange({ ...options, activeMatcher: algo.id as any })}
                      className={`p-2.5 rounded-2xl border transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-[#0071E3]/20 border-[#2997FF] text-white'
                          : 'bg-white/[0.02] border-white/5 text-white/60 hover:text-white hover:bg-white/[0.04]'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-3.5 h-3.5 rounded-full flex items-center justify-center border ${
                            isSelected ? 'border-[#2997FF] bg-[#2997FF]' : 'border-white/30'
                          }`}
                        >
                          {isSelected && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                        </div>
                        <span className="text-xs font-bold text-white">{algo.name}</span>
                      </div>
                      <p className="text-[10px] text-white/45 mt-1 pl-5">{algo.desc}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Live Hyperparameters Sliders */}
            <div className="space-y-3 pt-2 border-t border-white/10">
              <div className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
                Algorithm Hyperparameters (configs/matchers.yaml)
              </div>

              {/* SIFT Lowe's Ratio */}
              <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-white/80 font-medium">SIFT Lowe&apos;s Ratio</span>
                  <span className="font-mono font-bold text-[#2997FF]">{matcherParams.sift.ratio_thresh}</span>
                </div>
                <input
                  type="range"
                  min="0.50"
                  max="0.95"
                  step="0.01"
                  value={matcherParams.sift.ratio_thresh}
                  onChange={(e) =>
                    onUpdateMatcherParams({
                      ...matcherParams,
                      sift: { ...matcherParams.sift, ratio_thresh: parseFloat(e.target.value) },
                    })
                  }
                  className="w-full accent-[#2997FF] cursor-pointer"
                />
              </div>

              {/* ANMS Keypoint Budget */}
              <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-white/80 font-medium">ANMS Spatial Keypoint Budget</span>
                  <span className="font-mono font-bold text-[#2997FF]">{matcherParams.sift.n_features} pts</span>
                </div>
                <input
                  type="range"
                  min="500"
                  max="4000"
                  step="100"
                  value={matcherParams.sift.n_features}
                  onChange={(e) =>
                    onUpdateMatcherParams({
                      ...matcherParams,
                      sift: { ...matcherParams.sift, n_features: parseInt(e.target.value) },
                    })
                  }
                  className="w-full accent-[#2997FF] cursor-pointer"
                />
              </div>

              {/* SuperPoint Keypoints */}
              <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-white/80 font-medium">SuperPoint Max Keypoints</span>
                  <span className="font-mono font-bold text-[#2997FF]">{matcherParams.lightglue.max_num_keypoints}</span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="3072"
                  step="128"
                  value={matcherParams.lightglue.max_num_keypoints}
                  onChange={(e) =>
                    onUpdateMatcherParams({
                      ...matcherParams,
                      lightglue: { ...matcherParams.lightglue, max_num_keypoints: parseInt(e.target.value) },
                    })
                  }
                  className="w-full accent-[#2997FF] cursor-pointer"
                />
              </div>

              {/* Crater Neural Detector Weights Dropdown */}
              <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                <div className="text-xs text-white/80 font-medium">Crater Ring Neural Detector</div>
                <select
                  value={matcherParams.crater.model_path}
                  onChange={(e) =>
                    onUpdateMatcherParams({
                      ...matcherParams,
                      crater: { ...matcherParams.crater, model_path: e.target.value },
                    })
                  }
                  className="w-full bg-black/60 border border-white/15 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#2997FF] font-mono cursor-pointer"
                >
                  <option value="models/crater_real_best.pt">crater_real_best.pt (YOLOv8x - Highest Accuracy)</option>
                  <option value="models/crater_yolov9.pt">crater_yolov9.pt (YOLOv9-e - Balanced)</option>
                  <option value="weights/yolo26n.pt">yolo26n.pt (Nano - High Speed 42ms)</option>
                </select>
              </div>

              {/* Save Configuration Button */}
              <button
                onClick={handleSave}
                disabled={isSaving}
                className={`w-full py-2.5 rounded-2xl font-bold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer shadow-lg ${
                  saveSuccess
                    ? 'bg-emerald-600 text-white'
                    : isSaving
                    ? 'bg-white/10 text-white/40 cursor-not-allowed'
                    : 'bg-[#0071E3] hover:bg-[#0077ED] text-white'
                }`}
              >
                {saveSuccess ? (
                  <>
                    <Check size={14} strokeWidth={3} />
                    <span>Configuration Saved to Backend</span>
                  </>
                ) : isSaving ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" />
                    <span>Writing to configs/matchers.yaml...</span>
                  </>
                ) : (
                  <>
                    <Save size={14} />
                    <span>Save Configuration to Backend</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* ── TAB 2: DATASET MANIFEST & PDS-4 INGESTION ── */}
        {activeTab === 'ingest' && (
          <div className="flex-1 overflow-y-auto p-4 space-y-4 sidebar-scroll">
            {/* Manifest Summary Strip */}
            <div className="grid grid-cols-3 gap-2">
              <div className="p-3 rounded-2xl bg-white/[0.04] border border-white/10 text-center">
                <div className="text-xl font-bold font-mono text-white">{stats?.total_pairs ?? pairs.length}</div>
                <div className="text-[10px] text-white/40 uppercase mt-0.5">Total Pairs</div>
              </div>
              <div className="p-3 rounded-2xl bg-white/[0.04] border border-white/10 text-center">
                <div className="text-xl font-bold font-mono text-emerald-400">{stats?.train_pairs ?? pairs.filter((p) => p.split === 'train').length}</div>
                <div className="text-[10px] text-white/40 uppercase mt-0.5">Train Set</div>
              </div>
              <div className="p-3 rounded-2xl bg-white/[0.04] border border-white/10 text-center">
                <div className="text-xl font-bold font-mono text-[#2997FF]">{stats?.test_pairs ?? pairs.filter((p) => p.split === 'test').length}</div>
                <div className="text-[10px] text-white/40 uppercase mt-0.5">Test Set</div>
              </div>
            </div>

            {/* Live Backend Pairs List */}
            <div>
              <div className="text-[10px] font-bold text-white/40 uppercase tracking-wider mb-2">
                Registered Multi-Sensor Pairs ({pairs.length})
              </div>
              <div className="space-y-1.5 max-h-56 overflow-y-auto sidebar-scroll pr-1">
                {pairs.map((pair) => (
                  <div
                    key={pair.pair_id}
                    onClick={() => onSelectPair && onSelectPair(pair.pair_id)}
                    className={`p-2 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                      selectedPairId === pair.pair_id
                        ? 'bg-[#0071E3]/20 border-[#2997FF] text-white'
                        : 'bg-white/[0.02] border-white/5 text-white/60 hover:bg-white/[0.04] hover:text-white'
                    }`}
                  >
                    <div className="truncate pr-2">
                      <div className="text-xs font-mono font-bold truncate text-white">{pair.pair_id}</div>
                      <div className="text-[10px] text-white/40">
                        {pair.src.sensor} ({pair.src.gsd_m}m) ➔ {pair.ref.type}
                      </div>
                    </div>
                    <span className="text-[9px] font-bold px-2 py-0.5 rounded-full bg-white/10 text-white shrink-0">
                      {pair.split}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* PDS-4 File Dropzone */}
            <div>
              <div className="text-[10px] font-bold text-white/40 uppercase tracking-wider mb-2">
                Ingest Custom Mission Data
              </div>

              {/* Native inputs for individual files and folder bundles */}
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".xml,.img,.tif,.tiff,.qub,.png,.jpg,.jpeg,.zip"
                onChange={(e) => {
                  if (e.target.files) handleFilesAdded(e.target.files);
                }}
                className="hidden"
              />
              <input
                ref={folderInputRef}
                type="file"
                {...({ webkitdirectory: '', directory: '' } as any)}
                multiple
                onChange={(e) => {
                  if (e.target.files) handleFilesAdded(e.target.files);
                }}
                className="hidden"
              />

              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={`p-5 rounded-3xl border border-dashed text-center transition-all ${
                  dragOver
                    ? 'border-[#0071E3] bg-[#0071E3]/15 scale-[1.01]'
                    : 'border-white/15 hover:border-white/30 bg-white/[0.02] hover:bg-white/[0.04]'
                }`}
              >
                <div className="flex items-center justify-center gap-2 mb-2 text-[#2997FF]">
                  <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center">
                    <Upload size={15} />
                  </div>
                  <div className="w-8 h-8 rounded-full bg-[#0071E3]/20 border border-[#2997FF]/30 flex items-center justify-center text-white">
                    <FolderUp size={15} />
                  </div>
                </div>
                <p className="font-bold text-white text-xs">Drop files or mission folders here</p>
                <p className="text-[10px] text-white/40 mt-1">Supports .IMG, .TIF, .QUB, .XML, .ZIP & Directory Bundles</p>

                <div className="flex items-center justify-center gap-2 mt-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="px-2.5 py-1.5 rounded-xl bg-white/10 hover:bg-white/15 text-white text-[11px] font-medium border border-white/10 flex items-center gap-1.5 transition-all cursor-pointer"
                  >
                    <FileText size={12} className="text-[#2997FF]" />
                    <span>Files</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => folderInputRef.current?.click()}
                    className="px-2.5 py-1.5 rounded-xl bg-[#0071E3]/20 hover:bg-[#0071E3]/30 text-white text-[11px] font-bold border border-[#2997FF]/40 flex items-center gap-1.5 transition-all cursor-pointer"
                  >
                    <FolderOpen size={12} className="text-[#2997FF]" />
                    <span>Folder Bundle</span>
                  </button>
                </div>
              </div>

              {uploadedFiles.length > 0 && (
                <div className="space-y-2 mt-3">
                  <div className="flex items-center justify-between text-[10px] text-white/40 px-1 font-mono">
                    <span>UPLOADED BUNDLE FILES ({uploadedFiles.length})</span>
                    <button onClick={() => setUploadedFiles([])} className="hover:text-white cursor-pointer">
                      Clear
                    </button>
                  </div>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto sidebar-scroll pr-1">
                    {uploadedFiles.map((f, i) => (
                      <div key={i} className="p-2 rounded-xl bg-white/[0.02] border border-white/5 flex items-center justify-between">
                        <span className="text-white text-xs font-mono truncate pr-2">{f.name}</span>
                        <span className="text-[10px] text-white/40 font-mono shrink-0">
                          {(f.size / 1024).toFixed(0)} KB
                        </span>
                      </div>
                    ))}
                  </div>
                  {ingestError && (
                    <div className="p-2 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
                      {ingestError}
                    </div>
                  )}
                  {ingestSuccess && (
                    <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs">
                      {ingestSuccess}
                    </div>
                  )}
                  <button
                    onClick={handleBatchIngest}
                    disabled={isIngesting}
                    className={`w-full py-2.5 rounded-2xl font-bold text-xs flex items-center justify-center gap-2 shadow-md transition-all active:scale-95 ${
                      isIngesting
                        ? 'bg-white/10 text-white/50 cursor-not-allowed'
                        : 'bg-[#0071E3] hover:bg-[#0077ED] text-white cursor-pointer'
                    }`}
                  >
                    {isIngesting ? (
                      <>
                        <RefreshCw size={13} className="animate-spin" />
                        <span>Ingesting & Registering...</span>
                      </>
                    ) : (
                      <span>Ingest & Process Batch ({uploadedFiles.length} Files)</span>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
