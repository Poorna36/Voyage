// sih-dashboard/src/hooks/useQuantumRun.ts
import { useState, useCallback, useRef } from 'react';
import {
  startQuantumRun,
  getQuantumRunStatus,
} from '../services/quantumApi';
import type {
  QuantumRunRequest,
  QuantumRunRecord,
} from '../types/quantum';

export function useQuantumRun() {
  const [isRunning, setIsRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'queued' | 'running' | 'completed' | 'failed'>('idle');
  const [currentRecord, setCurrentRecord] = useState<QuantumRunRecord | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const pollTimerRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const triggerRun = useCallback(
    async (request: QuantumRunRequest) => {
      stopPolling();
      setIsRunning(true);
      setStatus('queued');
      setErrorMessage(null);
      setCurrentRecord(null);

      try {
        const { run_id } = await startQuantumRun(request);
        setRunId(run_id);
        setStatus('running');

        // Poll every 800ms
        const startTime = Date.now();
        pollTimerRef.current = window.setInterval(async () => {
          try {
            const statusRes = await getQuantumRunStatus(run_id);
            if (statusRes.status === 'completed' && statusRes.record) {
              setCurrentRecord(statusRes.record as QuantumRunRecord);
              setStatus('completed');
              setIsRunning(false);
              stopPolling();
            } else if (statusRes.status === 'failed') {
              setStatus('failed');
              setErrorMessage(statusRes.error || 'Execution failed on server');
              setIsRunning(false);
              stopPolling();
            } else if (Date.now() - startTime > 70000) {
              // 70s client timeout
              setStatus('failed');
              setErrorMessage('Module run timed out on client after 70s');
              setIsRunning(false);
              stopPolling();
            }
          } catch (pollErr: any) {
            console.error('Polling error:', pollErr);
          }
        }, 800);
      } catch (err: any) {
        setIsRunning(false);
        setStatus('failed');
        setErrorMessage(err?.message || 'Failed to submit quantum task');
      }
    },
    [stopPolling]
  );

  return {
    isRunning,
    runId,
    status,
    currentRecord,
    errorMessage,
    triggerRun,
    reset: () => {
      stopPolling();
      setIsRunning(false);
      setRunId(null);
      setStatus('idle');
      setCurrentRecord(null);
      setErrorMessage(null);
    },
  };
}
