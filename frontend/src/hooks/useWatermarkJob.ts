import { useCallback, useEffect, useRef, useState } from "react";
import { watermarkApi, type JobStatus } from "../api/client";

interface UseWatermarkJobReturn {
  status: JobStatus | null;
  isPolling: boolean;
  error: string | null;
  startPolling: (jobId: string) => void;
  reset: () => void;
}

/**
 * Poll GET /job/{id} every `intervalMs` until status is "success" or "failure".
 */
export function useWatermarkJob(intervalMs = 2000): UseWatermarkJobReturn {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const jobIdRef = useRef<string | null>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsPolling(false);
  }, []);

  const poll = useCallback(async () => {
    if (!jobIdRef.current) return;
    try {
      const res = await watermarkApi.jobStatus(jobIdRef.current);
      setStatus(res.data);
      if (res.data.status === "success" || res.data.status === "failure") {
        cleanup();
        if (res.data.status === "failure") {
          setError(res.data.error_message || "Job failed");
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Polling error");
      cleanup();
    }
  }, [cleanup]);

  const startPolling = useCallback(
    (jobId: string) => {
      cleanup();
      jobIdRef.current = jobId;
      setStatus(null);
      setError(null);
      setIsPolling(true);

      // Immediate first poll
      poll();

      timerRef.current = setInterval(poll, intervalMs);
    },
    [cleanup, poll, intervalMs],
  );

  const reset = useCallback(() => {
    cleanup();
    setStatus(null);
    setError(null);
    jobIdRef.current = null;
  }, [cleanup]);

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup]);

  return { status, isPolling, error, startPolling, reset };
}
