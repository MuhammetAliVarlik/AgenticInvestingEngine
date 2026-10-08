import { useTranslation } from 'react-i18next';
import { LANGUAGES } from '../i18n';

/** EN / TR toggle for the interface language; the choice is kept in the browser. */
export function LanguageSwitcher() {
  const { t, i18n } = useTranslation();
  const current = i18n.resolvedLanguage ?? 'en';
  return (
    <div
      role="group"
      aria-label={t('language.label')}
      className="flex items-center rounded-lg border border-white/60 bg-white/60 p-0.5 text-[11px] font-semibold shadow-xs backdrop-blur-md dark:border-white/10 dark:bg-slate-800/60"
    >
      {LANGUAGES.map((lang) => (
        <button
          key={lang.code}
          type="button"
          lang={lang.code}
          title={lang.name}
          aria-pressed={current === lang.code}
          onClick={() => void i18n.changeLanguage(lang.code)}
          className={`cursor-pointer rounded-md px-2 py-1 transition-all ${
            current === lang.code
              ? 'bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow-xs'
              : 'text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white'
          }`}
        >
          {lang.label}
        </button>
      ))}
    </div>
  );
}
