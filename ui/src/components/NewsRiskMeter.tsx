import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';

interface NewsRiskMeterProps {
  score: number; // 0 to 10
  symbol?: string;
}

export const NewsRiskMeter: React.FC<NewsRiskMeterProps> = ({ score, symbol }) => {
  const normalizedScore = Math.max(0, Math.min(10, score));

  // Bands: Low < 3, Moderate 3-7, High >= 7
  let band: 'Low' | 'Moderate' | 'High';
  let bandDescription: string;
  let bandColorClass: string;
  let bandBadgeClass: string;
  let Icon = ShieldCheck;

  if (normalizedScore < 3.0) {
    band = 'Low';
    bandDescription = 'Headlines and disclosures show minimal regulatory or sentiment headwinds.';
    bandColorClass = 'text-emerald-700 dark:text-emerald-400';
    bandBadgeClass = 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-800/60 dark:bg-emerald-950/40 dark:text-emerald-300';
    Icon = ShieldCheck;
  } else if (normalizedScore < 7.0) {
    band = 'Moderate';
    bandDescription = 'Balanced news backdrop with sector rotation or ordinary market chatter.';
    bandColorClass = 'text-amber-700 dark:text-amber-400';
    bandBadgeClass = 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-300';
    Icon = AlertTriangle;
  } else {
    band = 'High';
    bandDescription = 'Elevated controversy, ongoing litigation, or material regulatory scrutiny.';
    bandColorClass = 'text-rose-700 dark:text-rose-400';
    bandBadgeClass = 'border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-800/60 dark:bg-rose-950/40 dark:text-rose-300';
    Icon = ShieldAlert;
  }

  const fillPercentage = (normalizedScore / 10) * 100;

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/50 pb-3.5 dark:border-white/10">
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            News & Sentiment Risk Meter
          </h4>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            {symbol ? `Evaluating recent headlines & material disclosures for ${symbol}` : 'Trailing 14-day sentiment'}
          </p>
        </div>

        {/* Explicit label + icon + score with glass badge */}
        <div className={`flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-semibold backdrop-blur-md shadow-xs ${bandBadgeClass}`}>
          <Icon className="h-4 w-4" aria-hidden="true" />
          <span>Band: <strong>{band} Risk</strong></span>
          <span className="opacity-40">|</span>
          <span className="font-mono font-bold tabular-nums">{normalizedScore.toFixed(1)} / 10</span>
        </div>
      </div>

      <div className="mt-5">
        {/* Visual progress track with 3 band tick marks */}
        <div className="relative h-3.5 w-full overflow-hidden rounded-full bg-slate-200/70 p-0.5 shadow-inner dark:bg-slate-800/80 ring-1 ring-black/5 dark:ring-white/5">
          {/* Band zones guides */}
          <div className="absolute inset-0 grid grid-cols-10 opacity-30 pointer-events-none">
            <div className="col-span-3 border-r border-slate-400 dark:border-slate-500"></div>
            <div className="col-span-4 border-r border-slate-400 dark:border-slate-500"></div>
            <div className="col-span-3"></div>
          </div>
          {/* Active bar with gradient fill */}
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              normalizedScore < 3
                ? 'bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-400 shadow-sm shadow-emerald-500/30'
                : normalizedScore < 7
                ? 'bg-gradient-to-r from-amber-500 via-orange-400 to-amber-400 shadow-sm shadow-amber-500/30'
                : 'bg-gradient-to-r from-rose-500 via-red-500 to-pink-500 shadow-sm shadow-rose-500/30'
            }`}
            style={{ width: `${fillPercentage}%` }}
            role="progressbar"
            aria-valuenow={normalizedScore}
            aria-valuemin={0}
            aria-valuemax={10}
            aria-label={`News risk score ${normalizedScore.toFixed(1)} out of 10, ${band} risk`}
          />
        </div>

        {/* Legend showing explicit bands */}
        <div className="mt-2.5 grid grid-cols-3 text-center text-[11px] text-slate-500 dark:text-slate-400">
          <div className="text-left font-medium">
            <span className="text-emerald-600 dark:text-emerald-400 font-bold">Low</span> (&lt; 3.0)
          </div>
          <div className="font-medium">
            <span className="text-amber-600 dark:text-amber-400 font-bold">Moderate</span> (3.0 – 6.9)
          </div>
          <div className="text-right font-medium">
            <span className="text-rose-600 dark:text-rose-400 font-bold">High</span> (≥ 7.0)
          </div>
        </div>

        <p className="mt-3.5 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
          <strong className={bandColorClass}>{band} Risk:</strong> {bandDescription}
        </p>
      </div>
    </div>
  );
};
