import React, { useState } from 'react';
import { HistoricalPrediction } from '../types';
import { BarChart3, Table as TableIcon, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface RiskHistoryChartProps {
  history: HistoricalPrediction[];
  symbol: string;
}

export const RiskHistoryChart: React.FC<RiskHistoryChartProps> = ({ history, symbol }) => {
  const { t, i18n } = useTranslation();
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!history || history.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-xs text-slate-500 dark:text-slate-400 shadow-sm">
        {t('history.empty', { symbol })}
      </div>
    );
  }

  // Dimensions for SVG chart
  const svgWidth = 640;
  const svgHeight = 220;
  const paddingLeft = 45;
  const paddingRight = 25;
  const paddingTop = 25;
  const paddingBottom = 35;

  const chartWidth = svgWidth - paddingLeft - paddingRight;
  const chartHeight = svgHeight - paddingTop - paddingBottom;

  // Y-axis is fixed from 0 to 10 for risk score
  const minY = 0;
  const maxY = 10;

  // Plot only analyses that have a risk score; a missing score is never drawn as 0.
  const scored = history.filter((item) => item.risk_score !== null);
  const points = scored.map((item, idx) => {
    const x = paddingLeft + (idx / Math.max(1, scored.length - 1)) * chartWidth;
    const score = item.risk_score as number;
    const y = paddingTop + chartHeight - ((score - minY) / (maxY - minY)) * chartHeight;
    return { ...item, x, y, score };
  });

  const pathD = points.length === 0 ? '' : points.reduce((acc, pt, idx) => {
    return `${acc} ${idx === 0 ? 'M' : 'L'} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`;
  }, '');

  // Fill area under the line
  const areaD = points.length === 0 ? '' : `${pathD} L ${points[points.length - 1].x.toFixed(1)} ${(paddingTop + chartHeight).toFixed(1)} L ${points[0].x.toFixed(1)} ${(paddingTop + chartHeight).toFixed(1)} Z`;

  const formatDate = (iso: string) => {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(i18n.resolvedLanguage, { day: '2-digit', month: 'short' });
    } catch {
      return iso;
    }
  };

  const hoveredPoint = hoveredIndex !== null ? points[hoveredIndex] : null;

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/50 pb-3.5 dark:border-white/10">
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            {t('history.title')}
          </h4>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            {t('history.subtitle', { symbol })}
          </p>
        </div>

        {/* View Mode Toggle with Glass Pill */}
        <div className="flex items-center rounded-xl border border-white/60 bg-white/60 p-1 backdrop-blur-md shadow-xs dark:border-white/10 dark:bg-slate-800/60">
          <button
            type="button"
            onClick={() => setViewMode('chart')}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
              viewMode === 'chart'
                ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
            }`}
            aria-pressed={viewMode === 'chart'}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>{t('history.chart')}</span>
          </button>
          <button
            type="button"
            onClick={() => setViewMode('table')}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
              viewMode === 'table'
                ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
            }`}
            aria-pressed={viewMode === 'table'}
          >
            <TableIcon className="h-3.5 w-3.5" />
            <span>{t('history.table')}</span>
          </button>
        </div>
      </div>

      {viewMode === 'chart' ? (
        <div className="relative mt-5">
          <div className="w-full overflow-x-auto">
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full max-w-full h-auto select-none"
              role="img"
              aria-label={t('history.chartLabel', { symbol })}
            >
              <defs>
                <linearGradient id={`areaGrad-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.4" />
                  <stop offset="60%" stopColor="#6366f1" stopOpacity="0.15" />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
                </linearGradient>
                <linearGradient id={`lineGrad-${symbol}`} x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#38bdf8" />
                  <stop offset="50%" stopColor="#0284c7" />
                  <stop offset="100%" stopColor="#6366f1" />
                </linearGradient>
              </defs>

              {/* Band guide background bars (Low <3, Mod 3-7, High >=7) */}
              {/* High risk band 7-10 */}
              <rect
                x={paddingLeft}
                y={paddingTop}
                width={chartWidth}
                height={chartHeight * 0.3}
                fill="currentColor"
                className="text-rose-500/5 dark:text-rose-500/10"
              />
              {/* Mod risk band 3-7 */}
              <rect
                x={paddingLeft}
                y={paddingTop + chartHeight * 0.3}
                width={chartWidth}
                height={chartHeight * 0.4}
                fill="currentColor"
                className="text-amber-500/5 dark:text-amber-500/10"
              />
              {/* Low risk band 0-3 */}
              <rect
                x={paddingLeft}
                y={paddingTop + chartHeight * 0.7}
                width={chartWidth}
                height={chartHeight * 0.3}
                fill="currentColor"
                className="text-emerald-500/5 dark:text-emerald-500/10"
              />

              {/* Y Grid lines at 10, 7, 3, 0 */}
              {[10, 7, 3, 0].map((score) => {
                const y = paddingTop + chartHeight - ((score - minY) / (maxY - minY)) * chartHeight;
                return (
                  <g key={score}>
                    <line
                      x1={paddingLeft}
                      y1={y}
                      x2={svgWidth - paddingRight}
                      y2={y}
                      stroke="currentColor"
                      strokeDasharray={score === 0 ? 'none' : '3 3'}
                      className="text-slate-200/60 dark:text-slate-800"
                    />
                    <text
                      x={paddingLeft - 8}
                      y={y + 3}
                      textAnchor="end"
                      className="text-[10px] font-mono fill-slate-400 dark:fill-slate-500"
                    >
                      {score}
                    </text>
                  </g>
                );
              })}

              {/* Area fill with rich multi-stop gradient */}
              <path d={areaD} fill={`url(#areaGrad-${symbol})`} />

              {/* Line path with gradient stroke */}
              <path
                d={pathD}
                fill="none"
                stroke={`url(#lineGrad-${symbol})`}
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Data points */}
              {points.map((pt, idx) => {
                const isHovered = hoveredIndex === idx;
                return (
                  <g
                    key={idx}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredIndex(idx)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    tabIndex={0}
                    role="button"
                    aria-label={t('history.pointLabel', { date: formatDate(pt.timestamp), score: pt.score, signal: t(`signal.${String(pt.signal).toLowerCase()}`, { defaultValue: String(pt.signal) }) })}
                    onFocus={() => setHoveredIndex(idx)}
                    onBlur={() => setHoveredIndex(null)}
                  >
                    {/* Expand touch target */}
                    <circle cx={pt.x} cy={pt.y} r={14} fill="transparent" />
                    {/* Ring highlight on hover */}
                    {isHovered && (
                      <circle cx={pt.x} cy={pt.y} r={8} fill="#38bdf8" opacity={0.35} />
                    )}
                    {/* Point dot */}
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={isHovered ? 5 : 4}
                      fill="#ffffff"
                      stroke="#0284c7"
                      strokeWidth={isHovered ? 3 : 2}
                    />
                    {/* X axis date label */}
                    <text
                      x={pt.x}
                      y={svgHeight - 10}
                      textAnchor="middle"
                      className="text-[10px] font-mono fill-slate-400 dark:fill-slate-500"
                    >
                      {formatDate(pt.timestamp)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Interactive Tooltip Overlay with Frosted Dark Glass */}
          {hoveredPoint && (
            <div
              className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-2xl border border-white/20 bg-slate-950/90 px-4 py-2.5 text-xs text-white shadow-2xl backdrop-blur-xl"
              style={{
                left: `${(hoveredPoint.x / svgWidth) * 100}%`,
                top: `${(hoveredPoint.y / svgHeight) * 100}%`,
                marginTop: '-12px'
              }}
            >
              <div className="font-bold text-slate-200">{formatDate(hoveredPoint.timestamp)}</div>
              <div className="mt-1 flex items-center gap-2 text-slate-300">
                <span>{t('history.riskScore')}:</span>
                <span className="font-mono font-bold bg-gradient-to-r from-sky-400 to-indigo-300 bg-clip-text text-transparent tabular-nums">
                  {hoveredPoint.score.toFixed(1)} / 10
                </span>
              </div>
              {hoveredPoint.price_at_prediction && (
                <div className="flex items-center gap-2 text-slate-300">
                  <span>{t('history.price')}:</span>
                  <span className="font-mono font-semibold text-white tabular-nums">
                    {hoveredPoint.price_at_prediction.toLocaleString(i18n.resolvedLanguage, { maximumFractionDigits: 2 })}
                  </span>
                </div>
              )}
              <div className="flex items-center gap-2 text-slate-300">
                <span>{t('history.signal')}:</span>
                <span className="capitalize font-bold text-sky-400">
                  {t(`signal.${String(hoveredPoint.signal).toLowerCase()}`, { defaultValue: String(hoveredPoint.signal) })}
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Table View */
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs" aria-label={`Prediction history table for ${symbol}`}>
            <thead className="border-b border-white/40 bg-white/30 text-[11px] font-bold uppercase tracking-wider text-slate-500 backdrop-blur-xs dark:border-white/10 dark:bg-slate-800/40 dark:text-slate-400">
              <tr>
                <th scope="col" className="px-5 py-3">{t('history.date')}</th>
                <th scope="col" className="px-5 py-3">{t('history.signal')}</th>
                <th scope="col" className="px-5 py-3 text-right">{t('history.riskScore')}</th>
                <th scope="col" className="px-5 py-3 text-right">{t('history.price')}</th>
                <th scope="col" className="px-5 py-3">{t('history.visibleTo')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100/60 dark:divide-white/5 font-mono text-xs">
              {history.map((row, idx) => {
                const signal = (row.signal ?? '').toLowerCase();
                let Icon = Minus;
                let colorClass = 'text-slate-600 dark:text-slate-400';
                if (signal === 'bullish') {
                  Icon = TrendingUp;
                  colorClass = 'text-emerald-600 dark:text-emerald-400 font-bold';
                } else if (signal === 'bearish') {
                  Icon = TrendingDown;
                  colorClass = 'text-rose-600 dark:text-rose-400 font-bold';
                }

                return (
                  <tr key={idx} className="hover:bg-sky-500/5 dark:hover:bg-sky-400/5 transition-colors">
                    <td className="px-5 py-3 font-sans text-slate-800 dark:text-slate-200">
                      {formatDate(row.timestamp)}
                    </td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex items-center gap-1.5 ${colorClass}`}>
                        <Icon className="h-3.5 w-3.5" />
                        <span>{row.signal ? t(`signal.${String(row.signal).toLowerCase()}`, { defaultValue: String(row.signal) }) : '—'}</span>
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right tabular-nums text-slate-900 dark:text-slate-100 font-bold">
                      {row.risk_score !== null ? `${row.risk_score.toFixed(1)} / 10` : '—'}
                    </td>
                    <td className="px-5 py-3 text-right tabular-nums text-slate-900 dark:text-slate-100">
                      {row.price_at_prediction !== null ? row.price_at_prediction.toLocaleString(i18n.resolvedLanguage, { maximumFractionDigits: 2 }) : '—'}
                    </td>
                    <td className="px-5 py-3 font-sans text-slate-600 dark:text-slate-400">
                      {row.private ? t('history.onlyYou') : t('history.allUsers')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
