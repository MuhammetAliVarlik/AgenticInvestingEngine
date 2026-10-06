import React, { useState, useMemo } from 'react';
import { Instrument } from '../types';
import { Search, Check, ArrowRight, X, FileCheck, FileSpreadsheet } from 'lucide-react';

interface Step1InstrumentsProps {
  instruments: Instrument[];
  selectedSymbols: string[];
  onToggleSymbol: (symbol: string) => void;
  onProceed: () => void;
  isLoading?: boolean;
}

export const Step1Instruments: React.FC<Step1InstrumentsProps> = ({
  instruments,
  selectedSymbols,
  onToggleSymbol,
  onProceed,
  isLoading
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'equity' | 'index'>('all');

  const filteredInstruments = useMemo(() => {
    return instruments.filter((item) => {
      if (kindFilter !== 'all' && item.kind !== kindFilter) return false;
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase().trim();
      return (
        item.symbol.toLowerCase().includes(q) ||
        item.name.toLowerCase().includes(q)
      );
    });
  }, [instruments, kindFilter, searchQuery]);

  const maxReached = selectedSymbols.length >= 3;

  return (
    <section aria-labelledby="step1-heading" className="space-y-6">
      {/* Section Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-white/50 pb-4 dark:border-white/10">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-tr from-sky-600 to-indigo-600 text-xs font-bold text-white shadow-sm shadow-sky-500/30">
              1
            </span>
            <h2 id="step1-heading" className="text-base font-bold bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-900 dark:from-white dark:via-sky-100 dark:to-indigo-200 bg-clip-text text-transparent sm:text-lg">
              Choose Instruments
            </h2>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Select up to 3 Borsa İstanbul equities or indices for coordinated multi-agent cross-analysis.
          </p>
        </div>

        {/* Counter badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 rounded-xl border border-white/60 bg-white/60 px-3 py-1.5 text-xs font-medium backdrop-blur-md shadow-xs dark:border-white/10 dark:bg-slate-800/60">
            <span className="text-slate-500 dark:text-slate-400">Selection:</span>
            <span className="font-mono font-bold bg-gradient-to-r from-sky-600 to-indigo-600 bg-clip-text text-transparent dark:from-sky-400 dark:to-indigo-400 tabular-nums">
              {selectedSymbols.length} / 3
            </span>
          </div>

          <button
            type="button"
            onClick={onProceed}
            disabled={selectedSymbols.length === 0}
            className={`flex items-center gap-2 rounded-xl px-5 py-2 text-xs font-semibold shadow-md transition-all ${
              selectedSymbols.length > 0
                ? 'bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 text-white shadow-sky-600/25 hover:from-sky-500 hover:to-indigo-500 hover:shadow-sky-600/35 cursor-pointer'
                : 'bg-slate-200/60 text-slate-400 dark:bg-slate-800/60 dark:text-slate-500 cursor-not-allowed'
            }`}
          >
            <span>Continue to Data Upload</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Selected Instruments Strip with Frosted Glass & Gradient border */}
      {selectedSymbols.length > 0 && (
        <div className="rounded-2xl border border-sky-300/60 bg-gradient-to-r from-sky-500/10 via-indigo-500/10 to-teal-500/5 p-4 shadow-sm backdrop-blur-xl dark:border-sky-500/30 dark:from-sky-500/15 dark:via-indigo-500/15">
          <div className="text-xs font-semibold text-sky-950 dark:text-sky-200 mb-2.5 flex items-center justify-between">
            <span>Active Targets for Investigation ({selectedSymbols.length}/3):</span>
            <span className="text-[11px] font-medium text-sky-700 dark:text-sky-300">
              {3 - selectedSymbols.length} slot{3 - selectedSymbols.length !== 1 ? 's' : ''} remaining
            </span>
          </div>
          <div className="flex flex-wrap gap-2.5">
            {selectedSymbols.map((sym) => {
              const item = instruments.find((i) => i.symbol === sym);
              return (
                <div
                  key={sym}
                  className="flex items-center gap-2 rounded-xl border border-white/80 bg-white/80 px-3 py-1.5 text-xs text-slate-900 shadow-xs backdrop-blur-md dark:border-white/10 dark:bg-slate-900/80 dark:text-slate-100"
                >
                  <span className="font-mono font-bold bg-gradient-to-r from-sky-600 to-indigo-600 bg-clip-text text-transparent dark:from-sky-400 dark:to-indigo-300">{sym}</span>
                  <span className="max-w-[150px] truncate text-[11px] text-slate-500 dark:text-slate-400">
                    {item?.name || sym}
                  </span>
                  <button
                    type="button"
                    onClick={() => onToggleSymbol(sym)}
                    className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200 transition-colors"
                    aria-label={`Remove ${sym}`}
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Search and Filter Bar */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by symbol or company name (e.g. THYAO, Aselsan, XU100)..."
            className="w-full rounded-xl border border-white/70 bg-white/75 py-2.5 pl-10 pr-3.5 text-xs text-slate-900 placeholder:text-slate-400 shadow-xs backdrop-blur-md focus:border-sky-500 focus:outline-hidden focus:ring-2 focus:ring-sky-500/20 dark:border-white/10 dark:bg-slate-900/75 dark:text-slate-100 dark:placeholder:text-slate-500"
          />
        </div>

        {/* Filter tabs */}
        <div className="flex items-center rounded-xl border border-white/60 bg-white/60 p-1 backdrop-blur-md shadow-xs dark:border-white/10 dark:bg-slate-800/60">
          <button
            type="button"
            onClick={() => setKindFilter('all')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              kindFilter === 'all'
                ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
            }`}
          >
            All ({instruments.length})
          </button>
          <button
            type="button"
            onClick={() => setKindFilter('equity')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              kindFilter === 'equity'
                ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
            }`}
          >
            Equities ({instruments.filter((i) => i.kind === 'equity').length})
          </button>
          <button
            type="button"
            onClick={() => setKindFilter('index')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              kindFilter === 'index'
                ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
            }`}
          >
            Indices ({instruments.filter((i) => i.kind === 'index').length})
          </button>
        </div>
      </div>

      {/* Grid of Instruments with Glass Cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-24 animate-pulse rounded-2xl border border-white/30 bg-white/40 dark:border-white/10 dark:bg-slate-800/40 backdrop-blur-md"
            />
          ))}
        </div>
      ) : filteredInstruments.length === 0 ? (
        <div className="glass-panel rounded-2xl p-8 text-center text-xs text-slate-500 dark:text-slate-400">
          No instruments found matching &ldquo;{searchQuery}&rdquo;.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filteredInstruments.map((item) => {
            const isSelected = selectedSymbols.includes(item.symbol);
            const isDisabled = !isSelected && maxReached;

            return (
              <button
                key={item.symbol}
                type="button"
                onClick={() => onToggleSymbol(item.symbol)}
                disabled={isDisabled}
                aria-pressed={isSelected}
                className={`relative flex flex-col justify-between rounded-2xl p-4 text-left transition-all ${
                  isSelected
                    ? 'border-sky-400/80 bg-gradient-to-br from-sky-500/20 via-indigo-500/15 to-teal-500/10 shadow-lg shadow-sky-500/15 backdrop-blur-xl ring-2 ring-sky-400/50 dark:border-sky-400/50'
                    : isDisabled
                    ? 'opacity-40 cursor-not-allowed border-white/30 bg-white/20 dark:border-white/5 dark:bg-slate-900/30 backdrop-blur-xs'
                    : 'glass-card glass-card-interactive cursor-pointer'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-bold tracking-tight text-slate-900 dark:text-slate-50">
                      {item.symbol}
                    </span>

                    {/* Badge */}
                    {item.public_prices ? (
                      <span className="inline-flex items-center gap-1 rounded-xl border border-emerald-300/60 bg-gradient-to-r from-emerald-500/15 to-teal-500/10 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-800 backdrop-blur-xs dark:border-emerald-500/30 dark:text-emerald-300">
                        <FileCheck className="h-3 w-3" />
                        <span>No file needed</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-xl border border-amber-300/60 bg-gradient-to-r from-amber-500/10 to-orange-500/10 px-2.5 py-0.5 text-[10px] font-medium text-amber-800 backdrop-blur-xs dark:border-amber-500/30 dark:text-amber-300">
                        <FileSpreadsheet className="h-3 w-3 text-amber-600 dark:text-amber-400" />
                        <span>Price file needed</span>
                      </span>
                    )}
                  </div>

                  <p className="mt-1.5 line-clamp-2 text-xs text-slate-600 dark:text-slate-300">
                    {item.name}
                  </p>
                </div>

                <div className="mt-3.5 flex items-center justify-between border-t border-slate-200/40 pt-2.5 text-[11px] text-slate-400 dark:border-white/10">
                  <span className="capitalize">{item.kind}</span>
                  <div className="flex items-center gap-1">
                    {isSelected ? (
                      <span className="flex items-center gap-1 font-bold text-sky-700 dark:text-sky-300">
                        <Check className="h-3.5 w-3.5 stroke-[3]" />
                        <span>Selected</span>
                      </span>
                    ) : (
                      <span className="text-slate-400">Click to add</span>
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
};
