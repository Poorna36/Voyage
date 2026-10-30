import React, { useState, useRef } from 'react';
import { X, Upload, FileText, CheckCircle2, RefreshCw, Layers, FolderOpen, FolderUp } from 'lucide-react';
import { uploadMissionFiles } from '../services/api';
import type { ScenePreset } from '../types';

interface AddFilesModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPairCreated?: (newScene: ScenePreset) => void;
  onStartUpload?: (
    files: File[],
    pairName: string,
    sensor: string,
    roles: ('src' | 'ref')[]
  ) => void;
}

/**
 * Recursively scans all files and directories from drag-and-drop DataTransfer.
 * Traverses nested directory hierarchies (e.g. ISRO PDS-4 bundles).
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
                  // Ignore if property is non-configurable
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

export const AddFilesModal: React.FC<AddFilesModalProps> = ({
  isOpen,
  onClose,
  onPairCreated,
  onStartUpload,
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [fileRoles, setFileRoles] = useState<('src' | 'ref')[]>([]);
  const [pairName, setPairName] = useState('');
  const [sensor, setSensor] = useState('OHRC');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFilesAdded = (files: FileList | File[]) => {
    const arr = Array.from(files);
    const newRoles: ('src' | 'ref')[] = arr.map((f, idx) => {
      const overallIdx = selectedFiles.length + idx;
      const lower = ((f as any).webkitRelativePath || f.name).toLowerCase();
      if (lower.includes('ref') || lower.includes('lro') || lower.includes('nac') || overallIdx === 1) {
        return 'ref';
      }
      return 'src';
    });
    setSelectedFiles((prev) => [...prev, ...arr]);
    setFileRoles((prev) => [...prev, ...newRoles]);
    setUploadError(null);
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

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFileRoles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRoleChange = (index: number, role: 'src' | 'ref') => {
    setFileRoles((prev) => prev.map((r, i) => (i === index ? role : r)));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setUploadError('Please select at least one lunar mission file or folder to ingest.');
      return;
    }

    if (onStartUpload) {
      onStartUpload(selectedFiles, pairName, sensor, fileRoles);
      setSelectedFiles([]);
      setFileRoles([]);
      setPairName('');
      setUploadError(null);
      setUploadSuccess(null);
      onClose();
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const res = await uploadMissionFiles(
        selectedFiles,
        pairName,
        sensor,
        fileRoles
      );

      if (res && res.status === 'success') {
        setUploadSuccess(res.message);

        // Notify parent to select the newly created pair
        if (onPairCreated && res.pair_id) {
          const newScene: ScenePreset = {
            id: res.pair_id,
            name: res.name || res.pair_id.replace(/_/g, ' ').toUpperCase(),
            lat: res.pair?.latitude_center_deg ?? -70.0,
            lon: res.pair?.longitude_center_deg ?? 35.0,
            height: 80000,
            terrainClass: (res.pair?.terrain_class as any) ?? 'polar_highland',
            solarIncidenceDeg: 66.0,
            solarAzimuthDeg: 175.0,
            gsdM: sensor === 'OHRC' ? 0.31 : 0.50,
            description: `User-ingested mission pair (${selectedFiles.length} files, verified ${sensor})`,
          };

          setTimeout(() => {
            onPairCreated(newScene);
            onClose();
            setSelectedFiles([]);
            setFileRoles([]);
            setPairName('');
            setUploadSuccess(null);
          }, 1000);
        }
      } else {
        setUploadError('Failed to upload files. Ensure FastAPI backend is online.');
      }
    } catch (err: any) {
      setUploadError(err?.message || 'Network error during file upload.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in duration-150 select-none font-sans"
      onClick={onClose}
      onPointerDown={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-xl bg-[#0B0D13]/95 backdrop-blur-3xl border border-white/20 rounded-3xl p-6 shadow-[0_24px_80px_rgba(0,0,0,0.95)] flex flex-col gap-4 animate-in zoom-in-95 duration-150 text-white"
      >
        {/* Header */}
        <div className="flex items-start justify-between pb-3 border-b border-white/10">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#2997FF] shadow-[0_0_8px_rgba(41,151,255,0.8)]" />
              <span className="text-[11px] uppercase font-bold tracking-widest text-[#2997FF]">
                Mission Data Ingestion
              </span>
            </div>
            <h2 className="text-lg font-bold text-white mt-1">Add Lunar Mission Data</h2>
            <p className="text-xs text-white/50">
              Upload individual files or entire mission directory bundles (ISRO PDS-4, TIFF, PNG, JPG).
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Inputs row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="text-[11px] font-bold text-white/60 uppercase tracking-wider block mb-1.5">
              Target / Corridor Name
            </label>
            <input
              type="text"
              value={pairName}
              onChange={(e) => setPairName(e.target.value)}
              placeholder="e.g. South Pole Crater A"
              className="w-full bg-white/[0.04] border border-white/10 focus:border-[#2997FF] rounded-xl px-3 py-2 text-xs text-white placeholder-white/25 focus:outline-none transition-all font-mono"
            />
          </div>

          <div>
            <label className="text-[11px] font-bold text-white/60 uppercase tracking-wider block mb-1.5">
              Instrument / Sensor Type
            </label>
            <select
              value={sensor}
              onChange={(e) => setSensor(e.target.value)}
              className="w-full bg-black/80 border border-white/10 focus:border-[#2997FF] rounded-xl px-3 py-2 text-xs text-white focus:outline-none transition-all cursor-pointer font-sans"
            >
              <option value="OHRC">Chandrayaan-2 OHRC (0.3m/px)</option>
              <option value="TMC-2">Chandrayaan-2 TMC-2 (5m Stereo)</option>
              <option value="IIRS">Chandrayaan-2 IIRS (Hyperspectral)</option>
              <option value="LRO_NAC">LRO NAC (0.5m Baseline)</option>
            </select>
          </div>
        </div>

        {/* Automatic Geolocation & Metadata Banner */}
        <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/10 flex items-start gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-[#2997FF]/20 border border-[#2997FF]/40 flex items-center justify-center text-[#2997FF] shrink-0 mt-0.5">
            <Layers size={13} />
          </div>
          <div className="text-[11px] text-white/70 leading-relaxed">
            <span className="font-semibold text-white">Automated Footprint & Metadata Ingestion: </span>
            Footprint coordinates, resolution (GSD), and solar angles are extracted directly from the mission label (<span className="text-amber-300 font-mono">.xml</span> / <span className="text-cyan-300 font-mono">.html</span>).
          </div>
        </div>

        {/* Hidden native inputs for Individual Files and Full Directory Bundles */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".xml,.html,.htm,.img,.tif,.tiff,.qub,.png,.jpg,.jpeg,.zip"
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

        {/* Drag and Drop Zone with Dual Browse Options */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`p-5 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center text-center transition-all ${
            dragOver
              ? 'border-[#2997FF] bg-[#2997FF]/15 scale-[1.01]'
              : 'border-white/15 hover:border-white/30 bg-white/[0.02] hover:bg-white/[0.04]'
          }`}
        >
          <div className="flex items-center gap-2 mb-2 text-[#2997FF]">
            <div className="w-9 h-9 rounded-full bg-white/5 flex items-center justify-center">
              <Upload size={16} />
            </div>
            <div className="w-9 h-9 rounded-full bg-[#0071E3]/20 border border-[#2997FF]/30 flex items-center justify-center text-white">
              <FolderUp size={16} />
            </div>
          </div>
          <p className="text-xs font-bold text-white">
            Drag & drop files or entire mission folders here
          </p>
          <p className="text-[11px] text-white/40 mt-1 font-mono">
            Automatically scans nested folders for .XML / .HTML labels & .IMG binary rasters
          </p>

          <div className="flex items-center gap-2 mt-3.5">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-medium border border-white/10 flex items-center gap-1.5 transition-all cursor-pointer"
            >
              <FileText size={13} className="text-[#2997FF]" />
              <span>Browse Files</span>
            </button>
            <button
              type="button"
              onClick={() => folderInputRef.current?.click()}
              className="px-3 py-1.5 rounded-xl bg-[#0071E3]/20 hover:bg-[#0071E3]/30 text-white text-xs font-bold border border-[#2997FF]/40 flex items-center gap-1.5 transition-all cursor-pointer shadow-sm"
            >
              <FolderOpen size={13} className="text-[#2997FF]" />
              <span>Browse Folder / Bundle</span>
            </button>
          </div>
        </div>

        {/* Selected files list with Role Assignment and Directory Paths */}
        {selectedFiles.length > 0 && (
          <div className="space-y-1.5 max-h-48 overflow-y-auto sidebar-scroll pr-1">
            <div className="flex items-center justify-between text-[10px] text-white/40 font-mono px-1">
              <span>SELECTED FILES & BUNDLE ITEMS ({selectedFiles.length})</span>
              <button
                onClick={() => {
                  setSelectedFiles([]);
                  setFileRoles([]);
                }}
                className="text-red-400 hover:text-red-300 cursor-pointer"
              >
                Clear All
              </button>
            </div>
            {selectedFiles.map((file, i) => {
              const relPath = (file as any).webkitRelativePath || '';
              const ext = file.name.split('.').pop()?.toLowerCase() || '';
              const isXml = ext === 'xml';
              const isHtml = ext === 'html' || ext === 'htm';
              const isImg = ext === 'img' || ext === 'qub';
              const isZip = ext === 'zip';
              const isRaster = ['png', 'jpg', 'jpeg', 'tif', 'tiff'].includes(ext);

              return (
                <div
                  key={i}
                  className="flex items-center justify-between gap-2 p-2 rounded-xl bg-white/[0.03] border border-white/10 text-xs"
                >
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {relPath ? (
                      <FolderOpen size={14} className="text-[#2997FF] shrink-0" />
                    ) : (
                      <FileText size={14} className="text-[#2997FF] shrink-0" />
                    )}
                    <div className="min-w-0 flex-1 truncate">
                      <div className="font-mono text-xs text-white truncate">
                        {relPath || file.name}
                      </div>
                    </div>
                    <span className="text-[10px] text-white/40 font-mono shrink-0">
                      ({(file.size / 1024).toFixed(0)} KB)
                    </span>
                    {isXml && (
                      <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[9px] font-mono shrink-0">
                        PDS-4 Label (.xml)
                      </span>
                    )}
                    {isHtml && (
                      <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-[9px] font-mono shrink-0">
                        Edge PDS-4 (.html)
                      </span>
                    )}
                    {isImg && (
                      <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30 text-[9px] font-mono shrink-0">
                        Raw .IMG
                      </span>
                    )}
                    {isZip && (
                      <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30 text-[9px] font-mono shrink-0">
                        Archive Package
                      </span>
                    )}
                    {isRaster && (
                      <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[9px] font-mono shrink-0">
                        Calibrated Image
                      </span>
                    )}
                  </div>

                  {/* Role Selector */}
                  <div className="flex items-center gap-1.5 shrink-0">
                    <select
                      value={fileRoles[i] || (i === 1 ? 'ref' : 'src')}
                      onChange={(e) => handleRoleChange(i, e.target.value as 'src' | 'ref')}
                      className={`rounded-lg px-2 py-1 text-[10px] font-semibold uppercase tracking-wider border focus:outline-none transition-all cursor-pointer ${
                        (fileRoles[i] || (i === 1 ? 'ref' : 'src')) === 'src'
                          ? 'bg-[#0071E3]/20 text-[#2997FF] border-[#2997FF]/40'
                          : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      }`}
                    >
                      <option value="src" className="bg-[#0B0D13] text-white">Source (Mission)</option>
                      <option value="ref" className="bg-[#0B0D13] text-white">Reference (Baseline)</option>
                    </select>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveFile(i);
                      }}
                      className="p-1 rounded-md text-white/40 hover:text-white hover:bg-white/10 cursor-pointer"
                      title="Remove file"
                    >
                      <X size={13} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Ingestion Progress Stages during upload */}
        {isUploading && (
          <div className="p-3 rounded-2xl bg-[#0071E3]/10 border border-[#0071E3]/30 flex flex-col gap-2">
            <div className="flex items-center justify-between text-[11px] text-[#2997FF] font-semibold">
              <span className="flex items-center gap-2">
                <RefreshCw size={12} className="animate-spin" />
                Pipeline Ingestion in Progress...
              </span>
              <span className="font-mono text-[10px] text-white/60">Stage S1 ➔ S5</span>
            </div>
            <div className="grid grid-cols-3 gap-1.5 text-[9px] font-mono text-center">
              <div className="p-1.5 rounded-lg bg-white/5 border border-white/10 text-white/80">
                1. Parse XML & Extract GSD
              </div>
              <div className="p-1.5 rounded-lg bg-white/5 border border-white/10 text-white/80">
                2. Memmap .IMG Pixels
              </div>
              <div className="p-1.5 rounded-lg bg-[#0071E3]/20 border border-[#2997FF]/40 text-[#2997FF]">
                3. Sub-Pixel Matching
              </div>
            </div>
          </div>
        )}

        {/* Error / Success Feedback */}
        {uploadError && (
          <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300">
            {uploadError}
          </div>
        )}

        {uploadSuccess && (
          <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 size={15} className="shrink-0" />
            <span>{uploadSuccess}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5 pt-1">
          <button
            onClick={onClose}
            className="flex-1 py-2.5 rounded-2xl bg-white/5 hover:bg-white/10 text-white/70 hover:text-white font-semibold text-xs border border-white/10 transition-all cursor-pointer"
          >
            Cancel
          </button>

          <button
            onClick={handleUpload}
            disabled={isUploading || selectedFiles.length === 0}
            className={`flex-1 py-2.5 rounded-2xl font-bold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer shadow-lg active:scale-95 ${
              isUploading || selectedFiles.length === 0
                ? 'bg-white/10 text-white/30 cursor-not-allowed'
                : 'bg-[#0071E3] hover:bg-[#0077ED] text-white shadow-[0_0_20px_rgba(0,113,227,0.5)]'
            }`}
          >
            {isUploading ? (
              <>
                <RefreshCw size={14} className="animate-spin" />
                <span>Ingesting & Registering...</span>
              </>
            ) : (
              <>
                <Layers size={14} />
                <span>Upload & Register Pair</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
