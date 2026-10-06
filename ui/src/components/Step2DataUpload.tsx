import React, { useRef } from 'react';
import { Instrument, UploadedDataState } from '../types';
import {
  Upload,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  FileSpreadsheet,
  FileText,
  RotateCcw,
  Trash2,
  ArrowRight,
  Loader2
} from 'lucide-react';

interface Step2DataUploadProps {
  instruments: Instrument[];
  selectedSymbols: string[];
  dataState: Record<string, UploadedDataState>;
  onUploadDataset: (symbol: string, file: File) => Promise<void>;
  onUploadDocument: (symbol: string, file: File) => Promise<void>;
  onRemoveDataset: (symbol: string) => void;
  onRemoveDocument: (symbol: string) => void;
  /** False when the deployment does not accept documents (anonymous access codes). */
  documentsEnabled: boolean;
  onProceed: () => void;
}

export const Step2DataUpload: React.FC<Step2DataUploadProps> = ({
  instruments,
  selectedSymbols,
  dataState,
  onUploadDataset,
  onUploadDocument,
  onRemoveDataset,
  onRemoveDocument,
  documentsEnabled,
  onProceed
}) => {
  // Check readiness: every selected equity must have a validated dataset
  const missingEquities = selectedSymbols.filter((sym) => {
    const inst = instruments.find((i) => i.symbol === sym);
    if (!inst || inst.public_prices) return false;
    const st = dataState[sym];
    return !st || !st.dataset;
  });

  const isAnyUploading = selectedSymbols.some((sym) => {
    const st = dataState[sym];
    return st?.datasetUploading || st?.documentUploading;
  });

  const canProceed = missingEquities.length === 0 && !isAnyUploading;

  return (
    <section aria-labelledby="step2-heading" className="space-y-6">
      {/* Section Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-white/50 pb-4 dark:border-white/10">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-tr from-sky-600 to-indigo-600 text-xs font-bold text-white shadow-sm shadow-sky-500/30">
              2
            </span>
            <h2 id="step2-heading" className="text-base font-bold bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-900 dark:from-white dark:via-sky-100 dark:to-indigo-200 bg-clip-text text-transparent sm:text-lg">
              Provide Data & Disclosures
            </h2>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Equities need a daily price file (.xlsx or .csv).{' '}
            {documentsEnabled
              ? 'A disclosure filing (.pdf, .png or .jpg) is optional.'
              : 'Document upload is off in this demo.'}
          </p>
        </div>

        <button
          type="button"
          onClick={onProceed}
          disabled={!canProceed}
          className={`flex items-center gap-2 rounded-xl px-5 py-2 text-xs font-semibold shadow-md transition-all ${
            canProceed
              ? 'bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 text-white shadow-sky-600/25 hover:from-sky-500 hover:to-indigo-500 hover:shadow-sky-600/35 cursor-pointer'
              : 'bg-slate-200/60 text-slate-400 dark:bg-slate-800/60 dark:text-slate-500 cursor-not-allowed'
          }`}
        >
          <span>Continue to Analysis</span>
          <ArrowRight className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Explanatory callout for why equities need a file with glass gradient */}
      <div className="rounded-2xl border border-sky-300/60 bg-gradient-to-r from-sky-500/10 via-indigo-500/10 to-teal-500/5 p-4.5 text-xs shadow-sm backdrop-blur-xl dark:border-sky-500/30 dark:from-sky-500/15 dark:via-indigo-500/15">
        <div className="flex items-start gap-3.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 text-white shadow-sm">
            <FileSpreadsheet className="h-4 w-4" />
          </div>
          <div className="space-y-1">
            <h3 className="font-semibold text-sky-950 dark:text-sky-200 text-sm">
              Why do individual equities require historical price files?
            </h3>
            <p className="leading-relaxed text-sky-900/90 dark:text-sky-300/90">
              Borsa İstanbul equity prices are licensed data. Thus the engine analyses a daily price
              file that you can get yourself, for example from your brokerage or from İş Yatırım:
            </p>
            <div className="pt-1.5 flex flex-wrap items-center gap-2">
              <a
                href="https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Tarihsel-Fiyat-Bilgileri.aspx"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 font-semibold text-sky-700 underline hover:text-sky-900 dark:text-sky-300 dark:hover:text-sky-100"
              >
                <span>İş Yatırım historical prices (Excel download)</span>
                <ExternalLink className="h-3 w-3" />
              </a>
              <span className="text-slate-400 dark:text-slate-500">
                · Upload the Excel file without changes
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Instrument File Upload Zones */}
      <div className="space-y-4">
        {selectedSymbols.map((symbol) => {
          const inst = instruments.find((i) => i.symbol === symbol);
          const isIndex = inst?.kind === 'index' || inst?.public_prices;
          const state = dataState[symbol] || {};

          return (
            <div
              key={symbol}
              className="glass-panel rounded-2xl p-5 shadow-sm transition-all"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/50 pb-3.5 dark:border-white/10">
                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-base font-bold bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-950 dark:from-white dark:via-sky-200 dark:to-indigo-200 bg-clip-text text-transparent">
                    {symbol}
                  </span>
                  <span className="text-slate-300 dark:text-slate-600">·</span>
                  <span className="text-xs text-slate-600 dark:text-slate-300 font-medium">
                    {inst?.name || symbol}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {isIndex ? (
                    <span className="rounded-lg border border-emerald-300/60 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-semibold text-emerald-800 dark:border-emerald-500/30 dark:text-emerald-300">
                      No file needed
                    </span>
                  ) : (
                    <span className="rounded-lg border border-sky-300/60 bg-sky-500/10 px-2.5 py-1 text-[11px] font-semibold text-sky-800 dark:border-sky-500/30 dark:text-sky-300">
                      Price file needed
                    </span>
                  )}
                </div>
              </div>

              {/* Upload Dropzones Grid */}
              <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                {/* 1. Price History File Dropzone (Required for equities) */}
                {!isIndex ? (
                  <UploadDropzone
                    label="Historical Price File (.xlsx / .csv)"
                    required
                    symbol={symbol}
                    accept=".xlsx,.csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv"
                    isUploading={state.datasetUploading}
                    validationResult={
                      state.dataset
                        ? `${state.dataset.rows} trading days, ${state.dataset.start} to ${state.dataset.end}`
                        : undefined
                    }
                    errorMessage={state.datasetError}
                    onFileSelected={(file) => onUploadDataset(symbol, file)}
                    onReplace={(file) => onUploadDataset(symbol, file)}
                    onRemove={() => onRemoveDataset(symbol)}
                    icon={<FileSpreadsheet className="h-5 w-5 text-sky-600 dark:text-sky-400" />}
                    helpText="Accepts İş Yatırım historical export (.xlsx) or standard CSV format."
                  />
                ) : (
                  <div className="flex flex-col justify-center rounded-2xl border border-dashed border-emerald-300/60 bg-emerald-500/10 p-5 text-xs dark:border-emerald-500/30 dark:bg-emerald-950/20 backdrop-blur-sm">
                    <div className="flex items-center gap-2 font-semibold text-emerald-800 dark:text-emerald-300">
                      <CheckCircle2 className="h-4 w-4" />
                      <span>Prices come from TCMB EVDS</span>
                    </div>
                    <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                      The engine gets the daily closing levels of the index from the Central Bank of
                      the Republic of Türkiye (EVDS). You do not need a file.
                    </p>
                  </div>
                )}

                {/* 2. Disclosure Document Dropzone (Optional for all) */}
                {documentsEnabled ? (
                <UploadDropzone
                  label="Material Disclosure / Filing (Optional)"
                  required={false}
                  symbol={symbol}
                  accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/*"
                  isUploading={state.documentUploading}
                  validationResult={
                    state.document
                      ? `${state.document.pages} pages analyzed (${state.document.ocr_pages} OCR-scanned)`
                      : undefined
                  }
                  errorMessage={state.documentError}
                  onFileSelected={(file) => onUploadDocument(symbol, file)}
                  onReplace={(file) => onUploadDocument(symbol, file)}
                  onRemove={() => onRemoveDocument(symbol)}
                  icon={<FileText className="h-5 w-5 text-slate-500 dark:text-slate-400" />}
                  helpText="A KAP filing or a scan of it (.pdf, .png or .jpg). Scanned pages are read with OCR."
                />
                ) : (
                  <div className="flex flex-col justify-center rounded-2xl border border-dashed border-slate-300/70 bg-slate-500/5 p-5 text-xs dark:border-white/10">
                    <div className="flex items-center gap-2 font-semibold text-slate-700 dark:text-slate-300">
                      <FileText className="h-4 w-4" />
                      <span>Document upload is off</span>
                    </div>
                    <p className="mt-1 leading-relaxed text-slate-600 dark:text-slate-400">
                      Documents can contain personal data. Thus this demo does not accept them.
                    </p>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Progress and Continue Bar */}
      <div className="glass-panel flex items-center justify-between rounded-2xl p-4.5 shadow-sm">
        <div>
          <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
            {missingEquities.length === 0
              ? 'All necessary price files are validated.'
              : `Price file missing for: ${missingEquities.join(', ')}`}
          </span>
        </div>

        <button
          type="button"
          onClick={onProceed}
          disabled={!canProceed}
          className={`flex items-center gap-2 rounded-xl px-5 py-2.5 text-xs font-semibold shadow-md transition-all ${
            canProceed
              ? 'bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 text-white shadow-sky-600/25 hover:from-sky-500 hover:to-indigo-500 hover:shadow-sky-600/35 cursor-pointer'
              : 'bg-slate-200/60 text-slate-400 dark:bg-slate-800/60 dark:text-slate-500 cursor-not-allowed'
          }`}
        >
          <span>Continue to Run Analysis</span>
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </section>
  );
};

interface UploadDropzoneProps {
  label: string;
  required: boolean;
  symbol: string;
  accept: string;
  isUploading?: boolean;
  validationResult?: string;
  errorMessage?: string;
  onFileSelected: (file: File) => void;
  onReplace: (file: File) => void;
  onRemove: () => void;
  icon: React.ReactNode;
  helpText: string;
}

const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  label,
  required,
  symbol,
  accept,
  isUploading,
  validationResult,
  errorMessage,
  onFileSelected,
  onRemove,
  icon,
  helpText
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
      e.target.value = ''; // Reset input to allow re-uploading same filename
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={`relative flex flex-col justify-between rounded-2xl border p-4.5 transition-all backdrop-blur-md shadow-xs ${
        validationResult
          ? 'border-emerald-400/60 bg-gradient-to-br from-emerald-500/10 via-teal-500/5 to-transparent dark:border-emerald-500/30 dark:from-emerald-500/15'
          : errorMessage
          ? 'border-rose-400/60 bg-gradient-to-br from-rose-500/10 via-amber-500/5 to-transparent dark:border-rose-500/30 dark:from-rose-500/15'
          : 'border-dashed border-white/80 bg-white/50 hover:bg-white/80 hover:border-sky-400/50 dark:border-white/10 dark:bg-slate-800/40 dark:hover:bg-slate-800/70'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
        aria-label={`Upload file for ${symbol} - ${label}`}
      />

      <div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {icon}
            <span className="text-xs font-bold text-slate-900 dark:text-slate-100">
              {label}
            </span>
          </div>
          {required && (
            <span className="text-[10px] font-bold uppercase tracking-wider text-rose-600 dark:text-rose-400">
              Required
            </span>
          )}
        </div>

        {/* Status display */}
        <div className="mt-3">
          {isUploading ? (
            <div className="flex items-center gap-2 text-xs font-medium text-sky-600 dark:text-sky-400">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Uploading and validating…</span>
            </div>
          ) : validationResult ? (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 dark:text-emerald-400">
                <CheckCircle2 className="h-4 w-4" />
                <span>Validated</span>
              </div>
              <p className="font-mono text-xs text-slate-700 dark:text-slate-200">
                &ldquo;{validationResult}&rdquo;
              </p>
            </div>
          ) : errorMessage ? (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-bold text-rose-600 dark:text-rose-400">
                <AlertCircle className="h-4 w-4" />
                <span>Upload Failed</span>
              </div>
              <p className="text-xs text-rose-600 dark:text-rose-400">{errorMessage}</p>
            </div>
          ) : (
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Drag and drop file here, or click to browse.
              </p>
              <p className="mt-1 text-[11px] text-slate-400">{helpText}</p>
            </div>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div className="mt-4 flex items-center justify-end gap-2 border-t border-slate-200/50 pt-3 dark:border-white/10">
        {validationResult || errorMessage ? (
          <>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 rounded-xl border border-white/70 bg-white/70 px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-xs hover:bg-white dark:border-white/10 dark:bg-slate-800/70 dark:text-slate-200 dark:hover:bg-slate-800 transition-all cursor-pointer"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Replace</span>
            </button>
            <button
              type="button"
              onClick={onRemove}
              className="flex items-center gap-1.5 rounded-xl border border-rose-200/60 bg-rose-50/60 px-3 py-1.5 text-xs font-semibold text-rose-700 shadow-xs hover:bg-rose-100 dark:border-rose-900/40 dark:bg-rose-950/40 dark:text-rose-300 dark:hover:bg-rose-900/60 transition-all cursor-pointer"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Remove</span>
            </button>
          </>
        ) : (
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-sky-600/20 hover:from-sky-500 hover:to-indigo-500 transition-all cursor-pointer"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Select File</span>
          </button>
        )}
      </div>
    </div>
  );
};
