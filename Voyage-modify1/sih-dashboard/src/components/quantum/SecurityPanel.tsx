// sih-dashboard/src/components/quantum/SecurityPanel.tsx
import { useState, type FC } from 'react';
import { ShieldCheck, ShieldAlert, Key, CheckCircle, FileText } from 'lucide-react';
import { signProductArtifact, verifyProductArtifact } from '../../services/quantumApi';

export const SecurityPanel: FC = () => {
  const [samplePayload, setSamplePayload] = useState('CH2_OHRC_NAC_BOGUSLAWSKY_COREGISTERED_PRODUCT_2026');
  const [signatureData, setSignatureData] = useState<any>(null);
  const [verificationResult, setVerificationResult] = useState<{ valid: boolean; message: string } | null>(null);
  const [isSigning, setIsSigning] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);

  const handleSign = async () => {
    setIsSigning(true);
    setVerificationResult(null);
    try {
      const res = await signProductArtifact({ payload: samplePayload });
      setSignatureData(res);
    } catch (err: any) {
      alert(`Signing failed: ${err.message}`);
    } finally {
      setIsSigning(false);
    }
  };

  const handleVerify = async (tamper: boolean = false) => {
    if (!signatureData) return;
    setIsVerifying(true);
    try {
      const dataToVerify = tamper ? `${samplePayload}_TAMPERED_BYTE` : samplePayload;
      const res = await verifyProductArtifact({
        payload: dataToVerify,
        signature_b64: signatureData.sig_b64,
      });
      setVerificationResult(res);
    } catch (err: any) {
      setVerificationResult({ valid: false, message: err.message });
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="w-full bg-zinc-900/80 border border-white/10 rounded-xl overflow-hidden shadow-xl backdrop-blur-xl mb-6">
      <div className="px-5 py-3.5 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck className="text-emerald-400" size={17} />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Q7 — Post-Quantum Cryptographic Verification (ML-DSA-65 + Ed25519)
          </h3>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
          FIPS 204 Ready
        </span>
      </div>

      <div className="p-5 space-y-4">
        <div>
          <label className="block text-xs font-medium text-zinc-300 mb-1.5">
            Cartographic Export File / Manifest Payload
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={samplePayload}
              onChange={(e) => setSamplePayload(e.target.value)}
              className="flex-1 px-3 py-2 rounded-lg bg-black/60 border border-white/15 text-xs text-white font-mono focus:outline-none focus:border-cyan-400"
            />
            <button
              onClick={handleSign}
              disabled={isSigning}
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-all shadow-md flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              <Key size={13} />
              <span>{isSigning ? 'Signing...' : 'Sign with PQC'}</span>
            </button>
          </div>
        </div>

        {signatureData && (
          <div className="p-4 rounded-xl bg-black/60 border border-white/10 space-y-3">
            <div className="flex items-center justify-between text-xs border-b border-white/10 pb-2">
              <span className="font-semibold text-zinc-300 flex items-center gap-1.5">
                <FileText size={14} className="text-cyan-400" />
                Active Signature Descriptor
              </span>
              <span className="font-mono text-zinc-400 text-[11px]">
                Key ID: <strong className="text-white">{signatureData.pubkey_id}</strong>
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-2.5 rounded bg-zinc-900 border border-white/5">
                <span className="text-zinc-500 block text-[10px] uppercase">SHA3-256 Digest</span>
                <span className="text-cyan-300 break-all">{signatureData.sha3_256}</span>
              </div>
              <div className="p-2.5 rounded bg-zinc-900 border border-white/5">
                <span className="text-zinc-500 block text-[10px] uppercase">Classical SHA-256 Twin</span>
                <span className="text-zinc-400 break-all">{signatureData.classical_sha256}</span>
              </div>
            </div>

            <div>
              <span className="text-[10px] font-mono text-zinc-500 uppercase block mb-1">
                PQC Signature (Base64)
              </span>
              <div className="p-2 rounded bg-zinc-900 border border-white/5 font-mono text-[11px] text-emerald-400 break-all max-h-16 overflow-y-auto">
                {signatureData.sig_b64}
              </div>
            </div>

            <div className="pt-2 flex flex-wrap gap-2">
              <button
                onClick={() => handleVerify(false)}
                disabled={isVerifying}
                className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white font-medium text-xs transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <CheckCircle size={13} className="text-emerald-400" />
                <span>Verify Authentic Data</span>
              </button>

              <button
                onClick={() => handleVerify(true)}
                disabled={isVerifying}
                className="px-3.5 py-1.5 rounded-lg bg-red-950/40 hover:bg-red-900/50 border border-red-500/30 text-red-200 font-medium text-xs transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <ShieldAlert size={13} className="text-red-400" />
                <span>Test Tamper Rejection (+1 Byte Flip)</span>
              </button>
            </div>

            {verificationResult && (
              <div
                className={`mt-2 p-3 rounded-lg border text-xs flex items-center gap-2 ${
                  verificationResult.valid
                    ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
                    : 'bg-red-950/30 border-red-500/40 text-red-200'
                }`}
              >
                {verificationResult.valid ? <ShieldCheck size={16} /> : <ShieldAlert size={16} />}
                <span>
                  <strong>Result:</strong> {verificationResult.message}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
