import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  StreamFinalEvent,
  Instrument,
  HistoricalPrediction
} from '../types';
import { NewsRiskMeter } from './NewsRiskMeter';
import { TechnicalIndicatorTable } from './TechnicalIndicatorTable';
import { RiskHistoryChart } from './RiskHistoryChart';
import { getPdfReportUrl, fetchHistory } from '../api';
import {
  Download,
  ShieldAlert,
  AlertTriangle,
  Layers,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';

interface Step4ReviewResultsProps {
  finalResult: StreamFinalEvent;
  selectedSymbols: string[];
  instruments: Instrument[];
}

export const Step4ReviewResults: React.FC<Step4ReviewResultsProps> = ({
  finalResult,
  selectedSymbols,
  instruments
}) => {
  const [activeTab, setActiveTab] = useState<string>('report');
  const [historyCache, setHistoryCache] = useState<Record<string, HistoricalPrediction[]>>({});
  const [loadingHistory, setLoadingHistory] = useState<Record<string, boolean>>({});

  // Calculate total tokens across all agents
  const totalTokens = Object.values(finalResult.usage || {}).reduce(
    (acc, curr) => acc + (curr.input_tokens || 0) + (curr.output_tokens || 0),
    0
  );

  // Load prediction history for instruments when instrument tab is activated
  useEffect(() => {
    if (activeTab !== 'report' && !historyCache[activeTab]) {
      setLoadingHistory((prev) => ({ ...prev, [activeTab]: true }));
      fetchHistory(activeTab)
        .then((data) => {
          setHistoryCache((prev) => ({ ...prev, [activeTab]: data }));
        })
        .catch((err) => {
          console.warn('Failed to load history for', activeTab, err);
        })
        .finally(() => {
          setLoadingHistory((prev) => ({ ...prev, [activeTab]: false }));
        });
    }
  }, [activeTab, historyCache]);

  const pdfUrl = finalResult.analysis_id ? getPdfReportUrl(finalResult.analysis_id) : null;

  // Show only what the API returned: a missing value is "—", never an invented number.
  const checks = finalResult.checks || {};
  const groundingPct =
    typeof checks.grounding_score === 'number' ? `${Math.round(checks.grounding_score * 100)}%` : '—';
  const fullyGrounded = checks.grounding_score === 1;
  const checkedFigures = checks.checked_figures ?? '—';
  const injectionAttempts = finalResult.injection_flags?.length || 0;
  const seconds = finalResult.timings?.total_seconds;
  const duration = finalResult.cached ? 'Cached' : typeof seconds === 'number' ? `${seconds.toFixed(1)}s` : '—';

  const hasWarnings =
    (checks.ungrounded_figures && checks.ungrounded_figures.length > 0) ||
    (checks.unexpected_symbols && checks.unexpected_symbols.length > 0) ||
    injectionAttempts > 0;

  return (
    <section aria-labelledby="step4-heading" className="space-y-6">
      {/* Header with Title and PDF Download button */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-white/50 pb-4 dark:border-white/10">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-tr from-emerald-600 to-teal-500 text-xs font-bold text-white shadow-sm shadow-emerald-500/30">
              4
            </span>
            <h2 id="step4-heading" className="text-base font-bold bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-900 dark:from-white dark:via-sky-100 dark:to-indigo-200 bg-clip-text text-transparent sm:text-lg">
              Review Research Results
            </h2>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            The report, the quality checks, the indicators of each instrument and the news risk history.
          </p>
        </div>

        {/* Download PDF button (plain <a href download>) with gradient */}
        <div className="flex items-center gap-2">
          {finalResult.cached && (
            <span className="rounded-xl border border-white/60 bg-white/60 px-2.5 py-1 text-[11px] font-semibold text-slate-600 backdrop-blur-md dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-300">
              Served from cache
            </span>
          )}
          {pdfUrl && (
          <a
            href={pdfUrl}
            download
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-sky-600/25 hover:from-sky-500 hover:to-indigo-500 transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-600"
          >
            <Download className="h-4 w-4" />
            <span>Download PDF</span>
          </a>
          )}
        </div>
      </div>

      {/* Tabs navigation: "Report" + one tab per selected instrument */}
      <div className="flex items-center gap-2 overflow-x-auto border-b border-white/50 pb-2 dark:border-white/10">
        <button
          type="button"
          onClick={() => setActiveTab('report')}
          className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
            activeTab === 'report'
              ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-md shadow-sky-600/20'
              : 'border border-white/60 bg-white/50 text-slate-600 hover:bg-white/80 dark:border-white/10 dark:bg-slate-800/50 dark:text-slate-300'
          }`}
          aria-selected={activeTab === 'report'}
          role="tab"
        >
          <Layers className="h-4 w-4" />
          <span>Report</span>
        </button>

        {selectedSymbols.map((sym) => {
          const isCurrent = activeTab === sym;
          const tech = finalResult.technical?.[sym] || {};
          const sig = String(tech.signal || '').toLowerCase();

          return (
            <button
              key={sym}
              type="button"
              onClick={() => setActiveTab(sym)}
              className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                isCurrent
                  ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-md shadow-sky-600/20'
                  : 'border border-white/60 bg-white/50 text-slate-600 hover:bg-white/80 dark:border-white/10 dark:bg-slate-800/50 dark:text-slate-300'
              }`}
              aria-selected={isCurrent}
              role="tab"
            >
              <span className="font-mono">{sym}</span>
              {sig === 'bullish' ? (
                <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
              ) : sig === 'bearish' ? (
                <TrendingDown className="h-3.5 w-3.5 text-rose-400" />
              ) : (
                <Minus className="h-3.5 w-3.5 text-slate-400" />
              )}
            </button>
          );
        })}
      </div>

      {/* Tab 1: Executive Report */}
      {activeTab === 'report' && (
        <div className="space-y-6">
          {/* Quality tiles: Figures grounded %, figures checked, injection attempts removed, LLM tokens, duration with glass cards */}
          <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-5">
            {/* Figures grounded */}
            <div className="glass-card rounded-2xl p-4.5">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Figures grounded
              </span>
              <div className="mt-1.5 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black tracking-tight bg-gradient-to-r from-emerald-600 to-teal-500 bg-clip-text text-transparent dark:from-emerald-400 dark:to-teal-300 tabular-nums">
                  {groundingPct}
                </span>
                {fullyGrounded && (
                  <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-bold">
                    All matched
                  </span>
                )}
              </div>
            </div>

            {/* Figures checked */}
            <div className="glass-card rounded-2xl p-4.5">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Figures checked
              </span>
              <div className="mt-1.5 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black tracking-tight bg-gradient-to-r from-sky-600 to-indigo-600 bg-clip-text text-transparent dark:from-sky-400 dark:to-indigo-300 tabular-nums">
                  {checkedFigures}
                </span>
                <span className="text-[11px] text-slate-400 font-medium">figures</span>
              </div>
            </div>

            {/* Injection Attempts Removed */}
            <div className="glass-card rounded-2xl p-4.5">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Injections removed
              </span>
              <div className="mt-1.5 flex items-baseline gap-1.5">
                <span
                  className={`font-mono text-2xl font-black tracking-tight tabular-nums ${
                    injectionAttempts > 0
                      ? 'bg-gradient-to-r from-amber-500 to-rose-500 bg-clip-text text-transparent'
                      : 'text-slate-900 dark:text-white'
                  }`}
                >
                  {injectionAttempts}
                </span>
                
              </div>
            </div>

            {/* LLM Tokens */}
            <div className="glass-card rounded-2xl p-4.5">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                LLM tokens
              </span>
              <div className="mt-1.5 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black tracking-tight bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent dark:from-indigo-400 dark:to-purple-300 tabular-nums">
                  {totalTokens.toLocaleString('tr-TR')}
                </span>
                <span className="text-[11px] text-slate-400 font-medium">tokens</span>
              </div>
            </div>

            {/* Duration */}
            <div className="glass-card rounded-2xl p-4.5 col-span-2 sm:col-span-1">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Duration
              </span>
              <div className="mt-1.5 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black tracking-tight bg-gradient-to-r from-sky-600 to-teal-500 bg-clip-text text-transparent dark:from-sky-400 dark:to-teal-300 tabular-nums">
                  {duration}
                </span>
                
              </div>
            </div>
          </div>

          {/* Warnings Banner for ungrounded figures / unexpected symbols / injection flags */}
          {hasWarnings && (
            <div className="space-y-3">
              {checks.ungrounded_figures && checks.ungrounded_figures.length > 0 && (
                <div className="rounded-2xl border border-amber-300/60 bg-gradient-to-r from-amber-500/10 to-orange-500/5 p-4 text-xs backdrop-blur-md dark:border-amber-500/30">
                  <div className="flex items-start gap-2.5 text-amber-900 dark:text-amber-200">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500 mt-0.5" />
                    <div>
                      <h4 className="font-bold">Ungrounded Numerical Figures Detected</h4>
                      <p className="mt-0.5 text-amber-800 dark:text-amber-300">
                        These numbers in the report do not agree with any value from the data tools:
                      </p>
                      <ul className="mt-1.5 list-disc list-inside font-mono text-[11px]">
                        {checks.ungrounded_figures.map((fig, i) => (
                          <li key={i}>{fig}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {checks.unexpected_symbols && checks.unexpected_symbols.length > 0 && (
                <div className="rounded-2xl border border-amber-300/60 bg-gradient-to-r from-amber-500/10 to-orange-500/5 p-4 text-xs backdrop-blur-md dark:border-amber-500/30">
                  <div className="flex items-start gap-2.5 text-amber-900 dark:text-amber-200">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500 mt-0.5" />
                    <div>
                      <h4 className="font-bold">Unexpected Stock Tickers Referenced</h4>
                      <p className="mt-0.5 text-amber-800 dark:text-amber-300">
                        The report mentioned symbols outside your selected scope: {checks.unexpected_symbols.join(', ')}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {injectionAttempts > 0 && (
                <div className="rounded-2xl border border-rose-300/60 bg-gradient-to-r from-rose-500/10 to-red-500/5 p-4 text-xs backdrop-blur-md dark:border-rose-500/30">
                  <div className="flex items-start gap-2.5 text-rose-900 dark:text-rose-200">
                    <ShieldAlert className="h-4 w-4 shrink-0 text-rose-500 mt-0.5" />
                    <div>
                      <h4 className="font-bold">Prompt Injection Attempts Neutralized</h4>
                      <p className="mt-0.5 text-rose-800 dark:text-rose-300">
                        {injectionAttempts} suspicious instruction pattern(s) in third-party text were found and removed before the agents read it ({finalResult.injection_flags.join(', ')}).
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Rendered Markdown Report inside frosted glass */}
          <div className="glass-panel rounded-2xl p-6 sm:p-8 shadow-sm">
            <div className="prose prose-slate dark:prose-invert max-w-none text-sm leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {finalResult.report}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2+: Individual Instrument Tabs */}
      {activeTab !== 'report' && (
        <div className="space-y-6">
          {(() => {
            const sym = activeTab;
            const inst = instruments.find((i) => i.symbol === sym);
            const technicalData = finalResult.technical?.[sym] || {};
            const riskScore = finalResult.risk?.[sym];
            const historyData = historyCache[sym] || [];
            const isLoadingHist = loadingHistory[sym];

            const signal = String(technicalData.signal || '').toLowerCase();
            let SignalIcon = Minus;
            let signalClass = 'border-white/70 bg-white/60 text-slate-700 dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-300 shadow-xs backdrop-blur-md';

            if (signal === 'bullish') {
              SignalIcon = TrendingUp;
              signalClass = 'border-emerald-300/80 bg-gradient-to-r from-emerald-500/15 via-teal-500/15 to-emerald-500/10 text-emerald-800 dark:border-emerald-500/40 dark:text-emerald-300 shadow-xs backdrop-blur-md';
            } else if (signal === 'bearish') {
              SignalIcon = TrendingDown;
              signalClass = 'border-rose-300/80 bg-gradient-to-r from-rose-500/15 via-red-500/15 to-rose-500/10 text-rose-800 dark:border-rose-500/40 dark:text-rose-300 shadow-xs backdrop-blur-md';
            }

            return (
              <div className="space-y-6">
                {/* Instrument Header Card with Signal Badge */}
                <div className="glass-panel flex flex-wrap items-center justify-between gap-4 rounded-2xl p-5 shadow-sm">
                  <div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-2xl font-black bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-950 dark:from-white dark:via-sky-200 dark:to-indigo-200 bg-clip-text text-transparent">
                        {sym}
                      </span>
                      <span className="text-slate-300 dark:text-slate-600">·</span>
                      <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {inst?.name || sym}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-500">
                      Instrument type: <span className="capitalize font-semibold">{inst?.kind ?? '—'}</span>
                    </p>
                  </div>

                  {/* Signal Badge: Explicit icon + label (never color alone) */}
                  <div className={`flex items-center gap-2 rounded-xl border px-4 py-2 text-xs font-bold shadow-xs backdrop-blur-md ${signalClass}`}>
                    <SignalIcon className="h-4 w-4" aria-hidden="true" />
                    <span>Signal: <strong className="capitalize">{technicalData.signal ? String(technicalData.signal) : 'not available'}</strong></span>
                  </div>
                </div>

                {/* 0-10 News-Risk Meter */}
                {typeof riskScore === 'number' ? (
                  <NewsRiskMeter score={riskScore} symbol={sym} />
                ) : (
                  <div className="glass-panel rounded-2xl p-5 text-xs text-slate-600 dark:text-slate-400">
                    The news analyst did not give a risk score for {sym} in this analysis.
                  </div>
                )}

                {/* Technical Indicators Table */}
                <TechnicalIndicatorTable data={technicalData} symbol={sym} isIndex={inst?.kind === 'index'} />

                {/* Historical Risk Trajectory Chart with Tooltip & Table toggle */}
                {isLoadingHist ? (
                  <div className="h-48 animate-pulse rounded-2xl border border-white/30 bg-white/40 dark:border-white/10 dark:bg-slate-800/40 backdrop-blur-md" />
                ) : (
                  <RiskHistoryChart history={historyData} symbol={sym} />
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* Sources & Attributions Section */}
      {finalResult.sources && finalResult.sources.length > 0 && (
        <div className="glass-panel rounded-2xl p-6 shadow-sm">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-3.5">
            Source Attributions & Data Feeds
          </h4>
          <dl className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 text-xs">
            {finalResult.sources.map((src, i) => (
              <div
                key={i}
                className="rounded-xl border border-white/60 bg-white/50 p-3.5 backdrop-blur-md shadow-xs dark:border-white/10 dark:bg-slate-800/40"
              >
                <dt className="font-bold text-slate-900 dark:text-slate-100">
                  {src.name}
                </dt>
                <dd className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  {src.attribution}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </section>
  );
};
