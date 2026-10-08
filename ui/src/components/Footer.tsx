import { useState } from 'react';
import { Info, ShieldCheck, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Short privacy notice (ASD-STE100), shown from the footer and the sign-in screens. */
export function PrivacyNotice() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex cursor-pointer items-center gap-1.5 font-semibold text-sky-700 underline-offset-2 hover:underline dark:text-sky-300"
      >
        <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
        {t('privacy.link')}
      </button>
      {open && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="privacy-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="glass-panel max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl p-6 text-left text-sm leading-relaxed text-slate-700 dark:text-slate-300"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <h2 id="privacy-title" className="text-base font-bold text-slate-900 dark:text-white">
                {t('privacy.title')}
              </h2>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label={t('common.close')}
                className="cursor-pointer rounded-lg p-1 text-slate-500 hover:bg-slate-200/60 dark:hover:bg-slate-800"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
            <ul className="mt-3 list-disc space-y-2 pl-5">
              {(t('privacy.items', { returnObjects: true }) as string[]).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}

export function Footer() {
  const { t } = useTranslation();
  return (
    <footer className="mt-16 border-t border-white/50 bg-white/70 py-8 backdrop-blur-xl dark:border-white/10 dark:bg-slate-900/70">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="flex flex-col items-center justify-between gap-4 text-center sm:flex-row sm:text-left">
          <div>
            <div className="flex items-center justify-center gap-2 sm:justify-start">
              <span lang="en" className="text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-white">
                Investing Engine
              </span>
              <span className="text-slate-300 dark:text-slate-600">·</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">{t('footer.tagline')}</span>
            </div>
            <p className="mt-1 max-w-xl text-xs text-slate-500 dark:text-slate-400">
              {t('footer.about')}
            </p>
          </div>

          <div className="flex flex-col items-center gap-2 sm:items-end">
            <div className="flex items-center gap-2 rounded-xl border border-white/60 bg-white/60 px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-xs backdrop-blur-md dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-300">
              <Info className="h-3.5 w-3.5 text-sky-500" aria-hidden="true" />
              <span>{t('common.disclaimer')}</span>
            </div>
            <span className="text-xs">
              <PrivacyNotice />
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
