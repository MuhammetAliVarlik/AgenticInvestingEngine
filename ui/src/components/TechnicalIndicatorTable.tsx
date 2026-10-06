import React from 'react';
import { TrendingUp, TrendingDown, Minus, Check, X, Info } from 'lucide-react';
import { TechnicalIndicatorData } from '../types';

interface TechnicalIndicatorTableProps {
  data: TechnicalIndicatorData;
  symbol: string;
  /** Index levels are points, not Turkish lira. */
  isIndex: boolean;
}

export const TechnicalIndicatorTable: React.FC<TechnicalIndicatorTableProps> = ({ data, symbol, isIndex }) => {
  const formatVal = (val: unknown, digits = 2) => {
    if (val === undefined || val === null || val === '') return '—';
    if (typeof val === 'boolean') return val ? 'Yes' : 'No';
    if (typeof val === 'number') return val.toLocaleString('tr-TR', { maximumFractionDigits: digits });
    return String(val);
  };
  const level = (val: unknown) =>
    typeof val === 'number' ? (isIndex ? `${formatVal(val)} pts` : `₺${formatVal(val)}`) : formatVal(val);

  const signal = String(data.signal || '').toLowerCase();
  let SignalIcon = Minus;
  let signalBadgeClass = 'border-white/70 bg-white/60 text-slate-700 dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-300 shadow-xs backdrop-blur-md';
  let signalText = signal ? 'Neutral' : 'Not available';

  if (signal === 'bullish') {
    SignalIcon = TrendingUp;
    signalBadgeClass = 'border-emerald-300/80 bg-gradient-to-r from-emerald-500/15 via-teal-500/15 to-emerald-500/10 text-emerald-800 dark:border-emerald-500/40 dark:text-emerald-300 shadow-xs backdrop-blur-md';
    signalText = 'Bullish';
  } else if (signal === 'bearish') {
    SignalIcon = TrendingDown;
    signalBadgeClass = 'border-rose-300/80 bg-gradient-to-r from-rose-500/15 via-red-500/15 to-rose-500/10 text-rose-800 dark:border-rose-500/40 dark:text-rose-300 shadow-xs backdrop-blur-md';
    signalText = 'Bearish';
  }

  const above = data.price_above_ema34;

  const rows: { key: string; label: string; value?: string; customNode?: React.ReactNode; hint: string }[] = [
    {
      key: 'price',
      label: 'Last close',
      value: level(data.price),
      hint: isIndex ? 'Daily close of the index (TCMB EVDS)' : 'Last close in the uploaded price file',
    },
    {
      key: 'signal',
      label: 'Signal',
      customNode: (
        <span className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-semibold ${signalBadgeClass}`}>
          <SignalIcon className="h-3.5 w-3.5" aria-hidden="true" />
          <span>{signalText}</span>
        </span>
      ),
      hint: 'From the forecast RSI: below 30 bullish, above 70 bearish, else neutral',
    },
    { key: 'rsi', label: 'RSI (14)', value: formatVal(data.rsi), hint: '14-period relative strength index' },
    {
      key: 'predicted_next_rsi',
      label: 'Forecast next RSI',
      value: formatVal(data.predicted_next_rsi),
      hint: 'RandomForest forecast of the RSI for the next trading day',
    },
    { key: 'ema34', label: 'EMA 34', value: level(data.ema34), hint: '34-day exponential moving average' },
    { key: 'ema89', label: 'EMA 89', value: level(data.ema89), hint: '89-day exponential moving average' },
    {
      key: 'price_above_ema34',
      label: 'Close above EMA 34',
      customNode:
        typeof above === 'boolean' ? (
          <span className={`inline-flex items-center gap-1.5 font-medium ${above ? 'text-emerald-700 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'}`}>
            {above ? <Check className="h-4 w-4" aria-hidden="true" /> : <X className="h-4 w-4" aria-hidden="true" />}
            <span>{above ? 'Yes' : 'No'}</span>
          </span>
        ) : (
          <span>—</span>
        ),
      hint: 'Short-term trend filter',
    },
    { key: 'macd', label: 'MACD (12, 26)', value: formatVal(data.macd, 4), hint: 'MACD line: EMA 12 minus EMA 26' },
    {
      key: 'bb_pct',
      label: 'Bollinger %B',
      value: formatVal(data.bb_pct),
      hint: 'Close inside the 20-day, 2σ band: 0 = lower band, 1 = upper band',
    },
    {
      key: 'channel_position',
      label: 'Channel position',
      value: formatVal(data.channel_position),
      hint: 'Position of the close in the 20-day range: bottom, middle or top',
    },
    {
      key: 'momentum_divergence',
      label: 'Momentum divergence',
      value: formatVal(data.momentum_divergence),
      hint: 'RSI and price moved in opposite directions in the last 5 days',
    },
    { key: 'as_of', label: 'As of', value: formatVal(data.as_of), hint: 'Date of the last price' },
    { key: 'source', label: 'Price source', value: formatVal(data.source), hint: 'Data source of the prices' },
  ];

  return (
    <div className="glass-panel overflow-hidden rounded-2xl shadow-sm">
      <div className="border-b border-white/50 bg-white/40 px-6 py-4 backdrop-blur-md dark:border-white/10 dark:bg-slate-900/40 flex items-center justify-between">
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Technical indicators
          </h4>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Values that the technical analyst used for {symbol}
          </p>
        </div>
        <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400">
          <Info className="h-3.5 w-3.5 text-sky-500" aria-hidden="true" />
          <span>Source: technical_snapshot tool</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm" aria-label={`Technical indicators for ${symbol}`}>
          <thead className="border-b border-white/40 bg-white/30 text-[11px] font-bold uppercase tracking-wider text-slate-500 backdrop-blur-xs dark:border-white/10 dark:bg-slate-800/40 dark:text-slate-400">
            <tr>
              <th scope="col" className="px-6 py-3">Indicator Key</th>
              <th scope="col" className="px-6 py-3 text-right">Value</th>
              <th scope="col" className="hidden md:table-cell px-6 py-3 text-slate-400">Context</th>
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
