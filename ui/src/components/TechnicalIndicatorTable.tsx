import React from 'react';
import { TrendingUp, TrendingDown, Minus, Check, X, Info } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { TechnicalIndicatorData } from '../types';

interface TechnicalIndicatorTableProps {
  data: TechnicalIndicatorData;
  symbol: string;
  /** Index levels are points, not Turkish lira. */
  isIndex: boolean;
}

export const TechnicalIndicatorTable: React.FC<TechnicalIndicatorTableProps> = ({ data, symbol, isIndex }) => {
  const { t, i18n } = useTranslation();
  const formatVal = (val: unknown, digits = 2) => {
    if (val === undefined || val === null || val === '') return '—';
    if (typeof val === 'boolean') return val ? t('common.yes') : t('common.no');
    if (typeof val === 'number') return val.toLocaleString(i18n.resolvedLanguage, { maximumFractionDigits: digits });
    return String(val);
  };
  const level = (val: unknown) =>
    typeof val === 'number' ? (isIndex ? t('indicators.points', { value: formatVal(val) }) : `₺${formatVal(val)}`) : formatVal(val);

  const signal = String(data.signal || '').toLowerCase();
  let SignalIcon = Minus;
  let signalBadgeClass = 'border-white/70 bg-white/60 text-slate-700 dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-300 shadow-xs backdrop-blur-md';
  let signalText = signal ? t('signal.neutral') : t('step4.notAvailable');

  if (signal === 'bullish') {
    SignalIcon = TrendingUp;
    signalBadgeClass = 'border-emerald-300/80 bg-gradient-to-r from-emerald-500/15 via-teal-500/15 to-emerald-500/10 text-emerald-800 dark:border-emerald-500/40 dark:text-emerald-300 shadow-xs backdrop-blur-md';
    signalText = t('signal.bullish');
  } else if (signal === 'bearish') {
    SignalIcon = TrendingDown;
    signalBadgeClass = 'border-rose-300/80 bg-gradient-to-r from-rose-500/15 via-red-500/15 to-rose-500/10 text-rose-800 dark:border-rose-500/40 dark:text-rose-300 shadow-xs backdrop-blur-md';
    signalText = t('signal.bearish');
  }

  const above = data.price_above_ema34;

  const rows: { key: string; label: string; value?: string; customNode?: React.ReactNode; hint: string }[] = [
    {
      key: 'price',
      label: t('indicators.rows.price.label'),
      value: level(data.price),
      hint: isIndex ? t('indicators.priceHintIndex') : t('indicators.priceHintEquity'),
    },
    {
      key: 'signal',
      label: t('indicators.rows.signal.label'),
      customNode: (
        <span className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-semibold ${signalBadgeClass}`}>
          <SignalIcon className="h-3.5 w-3.5" aria-hidden="true" />
          <span>{signalText}</span>
        </span>
      ),
      hint: t('indicators.rows.signal.hint'),
    },
    { key: 'rsi', label: t('indicators.rows.rsi.label'), value: formatVal(data.rsi), hint: t('indicators.rows.rsi.hint') },
    {
      key: 'predicted_next_rsi',
      label: t('indicators.rows.predicted_next_rsi.label'),
      value: formatVal(data.predicted_next_rsi),
      hint: t('indicators.rows.predicted_next_rsi.hint'),
    },
    { key: 'ema34', label: t('indicators.rows.ema34.label'), value: level(data.ema34), hint: t('indicators.rows.ema34.hint') },
    { key: 'ema89', label: t('indicators.rows.ema89.label'), value: level(data.ema89), hint: t('indicators.rows.ema89.hint') },
    {
      key: 'price_above_ema34',
      label: t('indicators.rows.price_above_ema34.label'),
      customNode:
        typeof above === 'boolean' ? (
          <span className={`inline-flex items-center gap-1.5 font-medium ${above ? 'text-emerald-700 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'}`}>
            {above ? <Check className="h-4 w-4" aria-hidden="true" /> : <X className="h-4 w-4" aria-hidden="true" />}
            <span>{above ? t('common.yes') : t('common.no')}</span>
          </span>
        ) : (
          <span>—</span>
        ),
      hint: t('indicators.rows.price_above_ema34.hint'),
    },
    { key: 'macd', label: t('indicators.rows.macd.label'), value: formatVal(data.macd, 4), hint: t('indicators.rows.macd.hint') },
    {
      key: 'bb_pct',
      label: t('indicators.rows.bb_pct.label'),
      value: formatVal(data.bb_pct),
      hint: t('indicators.rows.bb_pct.hint'),
    },
    {
      key: 'channel_position',
      label: t('indicators.rows.channel_position.label'),
      value: formatVal(data.channel_position),
      hint: t('indicators.rows.channel_position.hint'),
    },
    {
      key: 'momentum_divergence',
      label: t('indicators.rows.momentum_divergence.label'),
      value: formatVal(data.momentum_divergence),
      hint: t('indicators.rows.momentum_divergence.hint'),
    },
    { key: 'as_of', label: t('indicators.rows.as_of.label'), value: formatVal(data.as_of), hint: t('indicators.rows.as_of.hint') },
    { key: 'source', label: t('indicators.rows.source.label'), value: formatVal(data.source), hint: t('indicators.rows.source.hint') },
  ];

  return (
    <div className="glass-panel overflow-hidden rounded-2xl shadow-sm">
      <div className="border-b border-white/50 bg-white/40 px-6 py-4 backdrop-blur-md dark:border-white/10 dark:bg-slate-900/40 flex items-center justify-between">
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            {t('indicators.title')}
          </h4>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            {t('indicators.subtitle', { symbol })}
          </p>
        </div>
        <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400">
          <Info className="h-3.5 w-3.5 text-sky-500" aria-hidden="true" />
          <span>{t('indicators.source')}</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm" aria-label={t('indicators.tableLabel', { symbol })}>
          <thead className="border-b border-white/40 bg-white/30 text-[11px] font-bold uppercase tracking-wider text-slate-500 backdrop-blur-xs dark:border-white/10 dark:bg-slate-800/40 dark:text-slate-400">
            <tr>
              <th scope="col" className="px-6 py-3">{t('indicators.colIndicator')}</th>
              <th scope="col" className="px-6 py-3 text-right">{t('indicators.colValue')}</th>
              <th scope="col" className="hidden md:table-cell px-6 py-3 text-slate-400">{t('indicators.colContext')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100/60 dark:divide-white/5 font-mono text-xs">
            {rows.map((row) => (
              <tr key={row.key} className="hover:bg-sky-500/5 dark:hover:bg-sky-400/5 transition-colors">
                <td className="px-6 py-3.5 font-sans font-semibold text-slate-900 dark:text-slate-100">
                  {row.label}
                </td>
                <td className="px-6 py-3.5 text-right font-mono tabular-nums text-slate-800 dark:text-slate-200">
                  {row.customNode || row.value}
                </td>
                <td className="hidden md:table-cell px-6 py-3.5 font-sans text-xs text-slate-500 dark:text-slate-400">
                  {row.hint}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
