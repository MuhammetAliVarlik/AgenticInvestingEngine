import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  ActivityLogItem,
  AuthError,
  Instrument,
  PipelineStage,
  PipelineStageId,
  StreamEvent,
  StreamFinalEvent,
  UploadedDataState,
  UsageStats,
  UserSession,
} from './types';
import {
  ApiError,
  MOCK_MODE,
  fetchCurrentUser,
  fetchInstruments,
  fetchUsage,
  streamAnalysis,
  uploadDataset,
  uploadDocument,
} from './api';
import { Header } from './components/Header';
import { AuthScreens } from './components/AuthScreens';
import { WorkflowStepper, type StepItem } from './components/WorkflowStepper';
import { Step1Instruments } from './components/Step1Instruments';
import { Step2DataUpload } from './components/Step2DataUpload';
import { Step3RunAnalysis } from './components/Step3RunAnalysis';
import { Step4ReviewResults } from './components/Step4ReviewResults';
import { Footer } from './components/Footer';

const MAX_SYMBOLS = 3;

const INITIAL_STAGES: PipelineStage[] = [
  { id: 'verify_inputs', name: 'Verify inputs', status: 'waiting', toolsUsed: [] },
  { id: 'plan_analysis', name: 'Plan the analysis', agent: 'supervisor', status: 'waiting', toolsUsed: [] },
  { id: 'technical_analysis', name: 'Technical analysis', agent: 'technical_analyst', status: 'waiting', toolsUsed: [] },
  { id: 'news_risk', name: 'News risk', agent: 'news_analyst', status: 'waiting', toolsUsed: [] },
  { id: 'macro_backdrop', name: 'Macro backdrop', agent: 'macro_analyst', status: 'waiting', toolsUsed: [] },
  { id: 'disclosures', name: 'Disclosures', agent: 'disclosure_analyst', status: 'waiting', toolsUsed: [] },
  { id: 'write_report', name: 'Write the report', agent: 'supervisor', status: 'waiting', toolsUsed: [] },
  { id: 'quality_checks', name: 'Quality checks', status: 'waiting', toolsUsed: [] },
];

/** Specialist agent -> pipeline stage. */
const AGENT_STAGE: Record<string, PipelineStageId> = {
  technical_analyst: 'technical_analysis',
  news_analyst: 'news_risk',
  macro_analyst: 'macro_backdrop',
  disclosure_analyst: 'disclosures',
};

const freshStages = (): PipelineStage[] => INITIAL_STAGES.map((s) => ({ ...s, toolsUsed: [] }));

/** Start `id` and close whichever planning or specialist stage was running before it. */
function activate(stages: PipelineStage[], id: PipelineStageId): PipelineStage[] {
  return stages.map((s) => {
    if (s.id === id) return s.status === 'done' ? s : { ...s, status: 'running' };
    if (s.status === 'running' && s.id !== 'verify_inputs') return { ...s, status: 'done' };
    return s;
  });
}

function addTool(stages: PipelineStage[], id: PipelineStageId, tool: string): PipelineStage[] {
  return stages.map((s) =>
    s.id === id && !s.toolsUsed.includes(tool) ? { ...s, toolsUsed: [...s.toolsUsed, tool] } : s,
  );
}

export default function App() {
  const { t } = useTranslation();
  const agentLabel = (agent: string) => t(`agents.${agent}`, { defaultValue: t('agents.supervisor') });
  const toolLabel = (tool: string) => t(`tools.${tool}`, { defaultValue: tool });
  const [currentUser, setCurrentUser] = useState<UserSession | null>(null);
  const [authError, setAuthError] = useState<AuthError | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingInitial, setLoadingInitial] = useState(true);

  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [loadingInstruments, setLoadingInstruments] = useState(false);
  const [usage, setUsage] = useState<UsageStats | null>(null);

  const [activeStep, setActiveStep] = useState<number>(1);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['XU100']);
  const [dataState, setDataState] = useState<Record<string, UploadedDataState>>({});

  const [isRunning, setIsRunning] = useState(false);
  const [stages, setStages] = useState<PipelineStage[]>(freshStages);
  const [activityLogs, setActivityLogs] = useState<ActivityLogItem[]>([]);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [streamingReportDraft, setStreamingReportDraft] = useState('');
  const [finalResult, setFinalResult] = useState<StreamFinalEvent | null>(null);
  const [resultSymbols, setResultSymbols] = useState<string[]>([]);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const timerIntervalRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const documentsEnabled = currentUser?.features?.documents ?? true;

  async function loadSession() {
    setLoadingInitial(true);
    setLoadError(null);
    try {
      const user = await fetchCurrentUser();
      setCurrentUser(user);
      setAuthError(null);
      setLoadingInstruments(true);
      const [list, usageStats] = await Promise.all([
        fetchInstruments(),
        fetchUsage().catch(() => null),
      ]);
      setInstruments(list);
      setUsage(usageStats);
    } catch (err: unknown) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        setAuthError({
          status: err.status,
          detail: err.detail,
          login_url: err.login_url,
          logout_url: err.logout_url,
          method: err.method,
        });
      } else {
        setLoadError(err instanceof Error ? err.message : t('errors.serviceUnavailable'));
      }
    } finally {
      setLoadingInitial(false);
      setLoadingInstruments(false);
    }
  }

  useEffect(() => {
    void loadSession();
  }, []);

  const refreshAfterRun = () => {
    fetchUsage().then(setUsage).catch(() => null);
    fetchCurrentUser().then(setCurrentUser).catch(() => null);
  };

  const addLog = (text: string, type: ActivityLogItem['type'] = 'status') => {
    setActivityLogs((prev) => [
      ...prev,
      {
        id: `log_${prev.length}_${Date.now()}`,
        timestamp: new Date().toTimeString().slice(0, 8),
        text,
        type,
      },
    ]);
  };

  const handleToggleSymbol = (symbol: string) => {
    setSelectedSymbols((prev) => {
      if (prev.includes(symbol)) return prev.filter((s) => s !== symbol);
      if (prev.length >= MAX_SYMBOLS) return prev;
      return [...prev, symbol];
    });
  };

  const patchData = (symbol: string, patch: Partial<UploadedDataState>) =>
    setDataState((prev) => ({ ...prev, [symbol]: { ...prev[symbol], ...patch } }));

  const handleUploadDataset = async (symbol: string, file: File) => {
    patchData(symbol, { datasetUploading: true, datasetError: undefined, dataset: undefined });
    try {
      const res = await uploadDataset(symbol, file);
      patchData(symbol, { datasetUploading: false, dataset: res });
    } catch (err: unknown) {
      patchData(symbol, {
        datasetUploading: false,
        datasetError: err instanceof Error ? err.message : t('errors.uploadFailed'),
      });
    }
  };

  const handleUploadDocument = async (symbol: string, file: File) => {
    patchData(symbol, { documentUploading: true, documentError: undefined, document: undefined });
    try {
      const res = await uploadDocument(symbol, file);
      patchData(symbol, { documentUploading: false, document: res });
    } catch (err: unknown) {
      patchData(symbol, {
        documentUploading: false,
        documentError: err instanceof Error ? err.message : t('errors.uploadFailed'),
      });
    }
  };

  const needsFile = (sym: string) => {
    const inst = instruments.find((i) => i.symbol === sym);
    return Boolean(inst && !inst.public_prices);
  };

  const codeQuotaUsed = currentUser?.code ? currentUser.code.used >= currentUser.code.quota : false;

  const getDisabledReason = (): string | undefined => {
    if (selectedSymbols.length === 0) return t('run.reason.noSelection');
    const missing = selectedSymbols.filter((sym) => needsFile(sym) && !dataState[sym]?.dataset);
    if (missing.length > 0) return t('run.reason.missingFile', { symbols: missing.join(', ') });
    const uploading = selectedSymbols.some(
      (sym) => dataState[sym]?.datasetUploading || dataState[sym]?.documentUploading,
    );
    if (uploading) return t('run.reason.uploading');
    if (codeQuotaUsed) return t('run.reason.codeQuota');
    if (usage && usage.analyses >= usage.analyses_limit) return t('run.reason.dailyQuota');
    return undefined;
  };

  const disabledReason = getDisabledReason();
  const canRun = !disabledReason && !isRunning;

  const handleStreamEvent = (event: StreamEvent) => {
    switch (event.type) {
      case 'status': {
        const target = event.text.replace(/^Consulting /, '');
        const stage = AGENT_STAGE[target];
        if (stage) {
          setStages((prev) => activate(prev, stage));
          addLog(t('log.handover', { agent: agentLabel(target) }));
        } else if (event.text === 'Writing the report') {
          setStages((prev) => activate(prev, 'write_report'));
          addLog(t('log.writing'));
        } else {
          addLog(event.text);
        }
        break;
      }
      case 'tool_call': {
        const stage = AGENT_STAGE[event.agent] ?? 'plan_analysis';
        setStages((prev) => {
          const writing = prev.find((s) => s.id === 'write_report')?.status === 'running';
          const target: PipelineStageId = stage === 'plan_analysis' && writing ? 'write_report' : stage;
          const current = prev.find((s) => s.id === target);
          const next = current?.status === 'running' ? prev : activate(prev, target);
          return addTool(next, target, event.tool);
        });
        addLog(
          t('log.requests', { agent: agentLabel(event.agent), tool: toolLabel(event.tool) }),
          'tool',
        );
        break;
      }
      case 'tool_result':
        addLog(
          t('log.received', { agent: agentLabel(event.agent), tool: toolLabel(event.tool) }),
          'success',
        );
        break;
      case 'token':
        if (event.agent === 'supervisor') {
          setStages((prev) =>
            prev.find((s) => s.id === 'write_report')?.status === 'running'
              ? prev
              : activate(prev, 'write_report'),
          );
          setStreamingReportDraft((prev) => prev + event.text);
        }
        break;
      case 'error':
        addLog(event.message, 'error');
        setAnalysisError(event.message);
        break;
      case 'final': {
        const succeeded = Boolean(event.report);
        setStages((prev) =>
          prev.map((s): PipelineStage => {
            if (s.id === 'verify_inputs') return s;
            if (!succeeded) return s.status === 'running' ? { ...s, status: 'failed' } : s;
            if (event.cached) return { ...s, status: 'done' };
            if (s.status === 'running' || s.id === 'write_report' || s.id === 'quality_checks') {
              return { ...s, status: 'done' };
            }
            return s.status === 'waiting' ? { ...s, status: 'not_needed' } : s;
          }),
        );
        if (succeeded) {
          setFinalResult(event);
          addLog(
            event.cached
              ? t('log.cached')
              : t('log.complete'),
            'success',
          );
        } else {
          setAnalysisError((prev) => prev ?? t('errors.analysisIncomplete'));
          addLog(t('log.incomplete'), 'error');
        }
        break;
      }
    }
  };

  const handleStartAnalysis = async () => {
    if (!canRun) return;
    const symbols = [...selectedSymbols];
    setIsRunning(true);
    setAnalysisError(null);
    setFinalResult(null);
    setResultSymbols(symbols);
    setStreamingReportDraft('');
    setActivityLogs([]);
    setStages(
      activate(
        freshStages().map((s): PipelineStage => (s.id === 'verify_inputs' ? { ...s, status: 'done' } : s)),
        'plan_analysis',
      ),
    );
    setElapsedSeconds(0);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    startTimeRef.current = Date.now();
    timerIntervalRef.current = window.setInterval(() => {
      setElapsedSeconds((Date.now() - startTimeRef.current) / 1000);
    }, 100);

    const datasets: Record<string, string> = {};
    const documents: Record<string, string> = {};
    for (const sym of symbols) {
      const st = dataState[sym];
      if (st?.dataset && needsFile(sym)) datasets[sym] = st.dataset.dataset_id;
      if (st?.document && documentsEnabled) documents[sym] = st.document.document_id;
    }
    const files = Object.keys(datasets).length;
    const docs = Object.keys(documents).length;
    addLog(
      t('log.inputs', { symbols: symbols.join(', ') }) +
        (files ? t('log.inputFiles', { count: files }) : '') +
        (docs ? t('log.inputDocs', { count: docs }) : ''),
    );

    try {
      await streamAnalysis({ symbols, datasets, documents }, handleStreamEvent, controller.signal);
    } catch (err: unknown) {
      const message = controller.signal.aborted
        ? t('log.cancelled')
        : err instanceof Error
          ? err.message
          : t('errors.analysisFailed');
      setAnalysisError(message);
      addLog(message, controller.signal.aborted ? 'warning' : 'error');
      setStages((prev) =>
        prev.map((s): PipelineStage => (s.status === 'running' ? { ...s, status: 'failed' } : s)),
      );
    } finally {
      if (timerIntervalRef.current !== null) window.clearInterval(timerIntervalRef.current);
      setIsRunning(false);
      abortControllerRef.current = null;
      refreshAfterRun();
    }
  };

  const handleCancelAnalysis = () => abortControllerRef.current?.abort();

  const hasSelectedInstruments = selectedSymbols.length > 0;
  const hasUploadedRequiredData = selectedSymbols.every(
    (sym) => !needsFile(sym) || Boolean(dataState[sym]?.dataset),
  );
  const hasCompletedRun = finalResult !== null;

  const stepperSteps: StepItem[] = [
    {
      id: 1,
      title: t('stepper.step1.title'),
      instruction: t('stepper.step1.instruction', { count: selectedSymbols.length, max: MAX_SYMBOLS }),
      state: hasSelectedInstruments && activeStep > 1 ? 'complete' : 'active',
    },
    {
      id: 2,
      title: t('stepper.step2.title'),
      instruction: hasUploadedRequiredData ? t('stepper.step2.done') : t('stepper.step2.todo'),
      state: !hasSelectedInstruments
        ? 'locked'
        : hasUploadedRequiredData && activeStep > 2
          ? 'complete'
          : 'active',
    },
    {
      id: 3,
      title: t('stepper.step3.title'),
      instruction: isRunning
        ? t('stepper.step3.running')
        : hasCompletedRun
          ? t('stepper.step3.done')
          : t('stepper.step3.todo'),
      state:
        !hasSelectedInstruments || !hasUploadedRequiredData
          ? 'locked'
          : hasCompletedRun && activeStep > 3
            ? 'complete'
            : 'active',
    },
    {
      id: 4,
      title: t('stepper.step4.title'),
      instruction: hasCompletedRun ? t('stepper.step4.done') : t('stepper.step4.todo'),
      state: hasCompletedRun ? 'active' : 'locked',
    },
  ];

  if (loadingInitial) {
    return (
      <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
        <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
          <div className="absolute -top-40 -left-40 h-96 w-96 rounded-full bg-gradient-to-br from-sky-400/25 to-indigo-500/25 blur-3xl" />
          <div className="absolute top-20 -right-40 h-[32rem] w-[32rem] rounded-full bg-gradient-to-bl from-indigo-500/25 via-purple-500/20 to-transparent blur-3xl" />
        </div>
        <div className="glass-panel relative z-10 flex flex-col items-center gap-3.5 rounded-3xl p-8 shadow-2xl" role="status">
          <div className="h-9 w-9 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
          <span className="text-xs font-semibold text-slate-700 dark:text-slate-200">{t('app.loading')}</span>
        </div>
      </div>
    );
  }

  if (authError) {
    return <AuthScreens error={authError} onSignedIn={() => void loadSession()} />;
  }

  if (loadError) {
    return <AuthScreens error={{ status: 503, detail: loadError }} onSignedIn={() => void loadSession()} />;
  }

  return (
    <div className="relative flex min-h-screen flex-col overflow-x-hidden bg-slate-50 text-slate-900 antialiased dark:bg-slate-950 dark:text-slate-100">
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
        <div className="absolute -top-40 -left-40 h-[28rem] w-[28rem] rounded-full bg-gradient-to-br from-sky-400/25 via-cyan-400/15 to-indigo-500/20 blur-3xl dark:from-sky-500/20 dark:via-cyan-500/10 dark:to-indigo-600/20" />
        <div className="absolute top-10 -right-40 h-[34rem] w-[34rem] rounded-full bg-gradient-to-bl from-indigo-500/25 via-purple-500/20 to-pink-500/10 blur-3xl dark:from-indigo-600/20 dark:via-purple-600/15" />
        <div className="absolute bottom-10 left-1/4 h-96 w-96 rounded-full bg-gradient-to-tr from-teal-400/20 via-sky-400/15 to-indigo-400/10 blur-3xl dark:from-teal-500/15 dark:via-sky-600/15" />
        <div className="absolute -bottom-20 -right-20 h-80 w-80 rounded-full bg-gradient-to-tl from-purple-500/15 to-indigo-500/10 blur-3xl dark:from-purple-600/10" />
      </div>

      <div className="relative z-10 flex min-h-screen flex-col">
        <Header user={currentUser} usage={usage} isMock={MOCK_MODE} />

        <WorkflowStepper steps={stepperSteps} activeStep={activeStep} onSelectStep={setActiveStep} />

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6">
          {activeStep === 1 && (
            <Step1Instruments
              instruments={instruments}
              selectedSymbols={selectedSymbols}
              onToggleSymbol={handleToggleSymbol}
              onProceed={() => setActiveStep(2)}
              isLoading={loadingInstruments}
            />
          )}

          {activeStep === 2 && (
            <Step2DataUpload
              instruments={instruments}
              selectedSymbols={selectedSymbols}
              dataState={dataState}
              documentsEnabled={documentsEnabled}
              onUploadDataset={handleUploadDataset}
              onUploadDocument={handleUploadDocument}
              onRemoveDataset={(sym) => patchData(sym, { dataset: undefined, datasetError: undefined })}
              onRemoveDocument={(sym) => patchData(sym, { document: undefined, documentError: undefined })}
              onProceed={() => setActiveStep(3)}
            />
          )}

          {activeStep === 3 && (
            <Step3RunAnalysis
              isRunning={isRunning}
              canRun={canRun}
              disabledReason={disabledReason}
              errorMessage={isRunning ? null : analysisError}
              onStartAnalysis={handleStartAnalysis}
              onCancelAnalysis={handleCancelAnalysis}
              stages={stages}
              activityLogs={activityLogs}
              elapsedSeconds={elapsedSeconds}
              streamingReportDraft={streamingReportDraft}
              onProceedToReview={() => setActiveStep(4)}
              hasFinalResults={finalResult !== null}
            />
          )}

          {activeStep === 4 && finalResult && (
            <Step4ReviewResults finalResult={finalResult} selectedSymbols={resultSymbols} instruments={instruments} />
          )}

          {activeStep === 4 && !finalResult && (
            <div className="glass-panel rounded-2xl p-8 text-center">
              <p className="text-sm text-slate-600 dark:text-slate-400">{t('app.noResults')}</p>
              <button
                type="button"
                onClick={() => setActiveStep(3)}
                className="mt-4 cursor-pointer rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 px-5 py-2.5 text-xs font-semibold text-white shadow-lg shadow-sky-600/20 transition-all hover:from-sky-500 hover:to-indigo-500"
              >
                {t('app.goToStep3')}
              </button>
            </div>
          )}
        </main>

        <Footer />
      </div>
    </div>
  );
}
