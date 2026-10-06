import React, { useRef, useEffect } from 'react';
import {
  PipelineStage,
  ActivityLogItem,
  StageStatus
} from '../types';
import {
  Play,
  StopCircle,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Wrench,
  FileText,
  Terminal,
  MinusCircle
} from 'lucide-react';

interface Step3RunAnalysisProps {
  isRunning: boolean;
  canRun: boolean;
  disabledReason?: string;
  errorMessage?: string | null;
  onStartAnalysis: () => void;
  onCancelAnalysis: () => void;
  stages: PipelineStage[];
  activityLogs: ActivityLogItem[];
  elapsedSeconds: number;
  streamingReportDraft: string;
  onProceedToReview?: () => void;
  hasFinalResults: boolean;
}

export const Step3RunAnalysis: React.FC<Step3RunAnalysisProps> = ({
  isRunning,
  canRun,
  disabledReason,
  errorMessage,
  onStartAnalysis,
  onCancelAnalysis,
  stages,
  activityLogs,
  elapsedSeconds,
  streamingReportDraft,
  onProceedToReview,
  hasFinalResults
}) => {
  const logContainerRef = useRef<HTMLDivElement>(null);
  const draftContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom as entries arrive
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [activityLogs]);

  // Auto-scroll draft to bottom as tokens arrive
  useEffect(() => {
    if (draftContainerRef.current && isRunning) {
      draftContainerRef.current.scrollTop = draftContainerRef.current.scrollHeight;
    }
  }, [streamingReportDraft, isRunning]);

  const formatElapsed = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = (sec % 60).toFixed(1);
    const paddedSecs = Number(secs) < 10 ? `0${secs}` : secs;
    return `${mins < 10 ? `0${mins}` : mins}:${paddedSecs}`;
  };

  const getStageStatusBadge = (status: StageStatus) => {
    switch (status) {
      case 'running':
        return (
          <span className="flex items-center gap-1.5 rounded-xl border border-sky-300/80 bg-gradient-to-r from-sky-500/20 via-cyan-500/15 to-indigo-500/15 px-2.5 py-0.5 text-[11px] font-bold text-sky-800 shadow-xs backdrop-blur-md dark:border-sky-500/40 dark:text-sky-300">
            <Loader2 className="h-3 w-3 animate-spin text-sky-600 dark:text-sky-400" />
            <span>Running</span>
          </span>
        );
      case 'done':
        return (
          <span className="flex items-center gap-1 rounded-xl border border-emerald-300/80 bg-gradient-to-r from-emerald-500/15 to-teal-500/15 px-2.5 py-0.5 text-[11px] font-bold text-emerald-800 shadow-xs backdrop-blur-md dark:border-emerald-500/40 dark:text-emerald-300">
            <CheckCircle2 className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
            <span>Done</span>
          </span>
        );
      case 'not_needed':
        return (
          <span className="flex items-center gap-1 rounded-xl border border-white/60 bg-white/40 px-2.5 py-0.5 text-[11px] font-medium text-slate-500 shadow-xs backdrop-blur-xs dark:border-white/10 dark:bg-slate-800/40 dark:text-slate-400">
            <MinusCircle className="h-3 w-3 text-slate-400" />
            <span>Not needed</span>
          </span>
        );
      case 'failed':
        return (
          <span className="flex items-center gap-1 rounded-xl border border-rose-300/80 bg-gradient-to-r from-rose-500/20 to-red-500/15 px-2.5 py-0.5 text-[11px] font-bold text-rose-800 shadow-xs backdrop-blur-md dark:border-rose-500/40 dark:text-rose-300">
            <AlertCircle className="h-3 w-3 text-rose-600 dark:text-rose-400" />
            <span>Failed</span>
          </span>
        );
      case 'waiting':
      default:
        return (
          <span className="rounded-xl border border-white/60 bg-white/40 px-2.5 py-0.5 text-[11px] font-medium text-slate-500 shadow-xs backdrop-blur-xs dark:border-white/10 dark:bg-slate-800/40 dark:text-slate-400">
            Waiting
          </span>
        );
    }
  };

  return (
    <section aria-labelledby="step3-heading" className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-white/50 pb-4 dark:border-white/10">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-tr from-sky-600 to-indigo-600 text-xs font-bold text-white shadow-sm shadow-sky-500/30">
              3
            </span>
            <h2 id="step3-heading" className="text-base font-bold bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-900 dark:from-white dark:via-sky-100 dark:to-indigo-200 bg-clip-text text-transparent sm:text-lg">
              Run Multi-Agent Analysis
            </h2>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            A supervisor agent sends work to four specialists (technical, news risk, macro,
            disclosures). Each stage and each data request shows here as it happens.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          {isRunning ? (
            <button
              type="button"
              onClick={onCancelAnalysis}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-rose-600 via-red-600 to-amber-600 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-rose-600/25 hover:from-rose-500 hover:to-red-500 transition-all cursor-pointer"
            >
              <StopCircle className="h-4 w-4" />
              <span>Cancel</span>
            </button>
          ) : hasFinalResults ? (
            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={onStartAnalysis}
                disabled={!canRun}
                className="flex items-center gap-2 rounded-xl border border-white/70 bg-white/70 px-4 py-2 text-xs font-semibold text-slate-700 shadow-xs hover:bg-white dark:border-white/10 dark:bg-slate-800/70 dark:text-slate-200 dark:hover:bg-slate-800 transition-all cursor-pointer"
              >
                <Play className="h-3.5 w-3.5 text-sky-600 dark:text-sky-400" />
                <span>Run again</span>
              </button>
              <button
                type="button"
                onClick={onProceedToReview}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-sky-600/25 hover:from-sky-500 hover:to-indigo-500 transition-all cursor-pointer"
              >
                <span>View the results</span>
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-end">
              <button
                type="button"
                onClick={onStartAnalysis}
                disabled={!canRun}
                className={`flex items-center gap-2 rounded-xl px-6 py-2.5 text-xs font-semibold shadow-md transition-all ${
                  canRun
                    ? 'bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 text-white shadow-sky-600/25 hover:from-sky-500 hover:to-indigo-500 hover:shadow-sky-600/35 cursor-pointer'
                    : 'bg-slate-200/60 text-slate-400 dark:bg-slate-800/60 dark:text-slate-500 cursor-not-allowed'
                }`}
              >
                <Play className="h-4 w-4" />
                <span>Start analysis</span>
              </button>
              {!canRun && disabledReason && (
                <span className="mt-1 text-[11px] font-medium text-rose-600 dark:text-rose-400">
                  {disabledReason}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="rounded-2xl border border-rose-300/60 bg-rose-500/10 p-4 text-xs font-medium text-rose-800 dark:border-rose-500/30 dark:text-rose-200"
        >
          {errorMessage}
        </div>
      )}

      {/* Timer and Pipeline Overview Card */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column (5 cols): 8-Stage Vertical Pipeline */}
        <div className="glass-panel rounded-2xl p-5 shadow-sm lg:col-span-5">
          <div className="flex items-center justify-between border-b border-white/50 pb-3.5 dark:border-white/10">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                8-Stage Multi-Agent Pipeline
              </h3>
              <p className="mt-0.5 text-xs text-slate-400">
                Agent handoffs and data requests
              </p>
            </div>

            {/* Elapsed Timer */}
            <div className="flex items-center gap-1.5 rounded-xl border border-white/60 bg-white/70 px-3 py-1 text-xs font-mono font-medium shadow-xs backdrop-blur-md dark:border-white/10 dark:bg-slate-800/70">
              <Clock className="h-3.5 w-3.5 text-sky-500" />
              <span className="tabular-nums font-bold text-slate-900 dark:text-white">
                {formatElapsed(elapsedSeconds)}
              </span>
            </div>
          </div>

          {/* Vertical Pipeline Items */}
          <ol className="mt-4 space-y-2.5" aria-label="Research pipeline stages">
            {stages.map((stage, idx) => {
              const isCurrent = stage.status === 'running';

              return (
                <li
                  key={stage.id}
                  className={`relative flex flex-col justify-between rounded-xl border p-3.5 transition-all ${
                    isCurrent
                      ? 'border-sky-400/80 bg-gradient-to-r from-sky-500/15 via-indigo-500/10 to-teal-500/10 shadow-md shadow-sky-500/15 backdrop-blur-md dark:border-sky-400/50 dark:from-sky-500/20 dark:via-indigo-500/15 ring-1 ring-sky-400/40'
                      : stage.status === 'done'
                      ? 'border-white/70 bg-white/60 dark:border-white/10 dark:bg-slate-800/50 backdrop-blur-sm'
                      : stage.status === 'not_needed'
                      ? 'border-white/30 bg-white/30 opacity-60 dark:border-white/5 dark:bg-slate-900/30'
                      : 'border-white/40 bg-white/40 dark:border-white/5 dark:bg-slate-900/40 backdrop-blur-xs'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <span className="font-mono text-xs font-bold bg-gradient-to-r from-sky-600 to-indigo-600 bg-clip-text text-transparent dark:from-sky-400 dark:to-indigo-300">
                        0{idx + 1}
                      </span>
                      <div>
                        <span className="text-xs font-bold text-slate-900 dark:text-slate-100">
                          {stage.name}
                        </span>
                        {stage.agent && (
                          <span className="ml-1.5 font-mono text-[10px] text-slate-400">
                            ({stage.agent})
                          </span>
                        )}
                      </div>
                    </div>
                    {getStageStatusBadge(stage.status)}
                  </div>

                  {/* Stage Tools Used */}
                  {stage.toolsUsed.length > 0 && (
                    <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-slate-200/40 pt-2 dark:border-white/10">
                      <span className="flex items-center gap-1 text-[10px] font-medium text-slate-400">
                        <Wrench className="h-3 w-3" />
                        <span>Tools:</span>
                      </span>
                      {stage.toolsUsed.map((tool) => (
                        <span
                          key={tool}
                          className="rounded-md border border-white/40 bg-white/80 px-2 py-0.5 font-mono text-[10px] font-medium text-slate-700 shadow-xs dark:border-white/10 dark:bg-slate-800/80 dark:text-slate-300"
                        >
                          {tool}
                        </span>
                      ))}
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        </div>

        {/* Right Column (7 cols): Live Activity Log & Live Report Draft */}
        <div className="flex flex-col gap-5 lg:col-span-7">
          {/* Timestamped Activity Log (24h clock) in Dark Glass Console */}
          <div className="flex flex-1 flex-col rounded-2xl border border-slate-800/80 bg-slate-950/85 p-5 text-slate-100 shadow-xl backdrop-blur-xl min-h-[220px]">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-sky-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  Real-Time Activity Log (24-Hour Clock)
                </h3>
              </div>
              <span className="font-mono text-[11px] text-slate-400">
                {activityLogs.length} events
              </span>
            </div>

            <div
              ref={logContainerRef}
              role="log"
              aria-live="polite"
              className="mt-3 flex-1 max-h-[200px] overflow-y-auto space-y-1.5 font-mono text-xs pr-1"
            >
              {activityLogs.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500 font-sans">
                  Ready. Click &ldquo;Execute Analysis&rdquo; to begin stream dispatch.
                </div>
              ) : (
                activityLogs.map((log) => (
                  <div key={log.id} className="flex items-start gap-2.5 leading-relaxed">
                    <span className="shrink-0 text-[11px] text-slate-500 tabular-nums">
                      {log.timestamp}
                    </span>
                    <span
                      className={`break-words ${
                        log.type === 'error'
                          ? 'text-rose-400 font-semibold'
                          : log.type === 'warning'
                          ? 'text-amber-400'
                          : log.type === 'tool'
                          ? 'text-sky-300 font-medium'
                          : log.type === 'success'
                          ? 'text-emerald-400 font-semibold'
                          : 'text-slate-300'
                      }`}
                    >
                      {log.text}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Live Report Draft Stream in Frosted Glass Panel */}
          <div className="glass-panel flex flex-1 flex-col rounded-2xl p-5 shadow-sm min-h-[220px]">
            <div className="flex items-center justify-between border-b border-white/50 pb-3 dark:border-white/10">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-sky-600 dark:text-sky-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Supervisor Stream Draft
                </h3>
              </div>
              {isRunning && (
                <span className="flex items-center gap-1.5 text-xs font-medium text-sky-600 dark:text-sky-400">
                  <span className="h-2 w-2 rounded-full bg-sky-500 animate-ping" />
                  <span>Synthesizing...</span>
                </span>
              )}
            </div>

            <div
              ref={draftContainerRef}
              className="mt-3 flex-1 max-h-[220px] overflow-y-auto rounded-xl border border-white/60 bg-white/60 p-4 font-mono text-xs leading-relaxed text-slate-800 shadow-inner backdrop-blur-md dark:border-white/10 dark:bg-slate-900/60 dark:text-slate-200"
            >
              {streamingReportDraft ? (
                <div className="whitespace-pre-wrap">{streamingReportDraft}</div>
              ) : (
                <div className="py-8 text-center text-xs text-slate-400 font-sans">
                  The supervisor&apos;s streamed report draft will appear here in real time.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
