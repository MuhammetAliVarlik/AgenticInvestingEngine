import { Activity, KeyRound, LogOut, Sparkles, User as UserIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { UsageStats, UserSession } from '../types';
import { LanguageSwitcher } from './LanguageSwitcher';

interface HeaderProps {
  user: UserSession | null;
  usage: UsageStats | null;
  isMock: boolean;
}

export function Header({ user, usage, isMock }: HeaderProps) {
  // An access code has its own quota; otherwise show the daily budget of the account.
  const { t, i18n } = useTranslation();
  const code = user?.code;
  const used = code ? code.used : usage?.analyses;
  const limit = code ? code.quota : usage?.analyses_limit;
  const label = code ? t('header.analysesOnCode') : t('header.analysesToday');
  const percentUsed =
    used !== undefined && limit ? Math.min(100, Math.round((used / Math.max(1, limit)) * 100)) : 0;

  return (
    <header className="sticky top-0 z-30 border-b border-white/50 bg-white/75 shadow-xs backdrop-blur-xl dark:border-white/10 dark:bg-slate-900/75">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-600 via-indigo-600 to-cyan-400 text-white shadow-md shadow-sky-500/25 ring-1 ring-white/30">
            <Activity className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-base font-bold tracking-tight text-slate-900 dark:text-white sm:text-lg">
              Investing Engine
            </span>
            <span className="hidden text-xs text-slate-300 dark:text-slate-600 sm:inline">·</span>
            <span className="hidden text-xs font-medium text-slate-500 dark:text-slate-400 sm:inline">
              {t('header.tagline')}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 sm:gap-5">
          {used !== undefined && limit !== undefined && (
            <div className="flex items-center gap-2 text-xs" title={`${used} of ${limit} ${label}`}>
              <div className="flex items-center gap-1.5 font-mono font-medium tabular-nums text-slate-800 dark:text-slate-200">
                <span className="font-semibold text-slate-900 dark:text-white">{used}</span>
                <span className="text-slate-400">/</span>
                <span>{limit}</span>
                <span className="hidden font-sans md:inline">{label}</span>
              </div>
              <div className="hidden h-1.5 w-14 overflow-hidden rounded-full bg-slate-200/80 ring-1 ring-black/5 dark:bg-slate-800 dark:ring-white/5 sm:block">
                <div
                  className={`h-full transition-all duration-300 ${
                    percentUsed >= 100
                      ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                      : 'bg-gradient-to-r from-sky-500 via-indigo-500 to-teal-400'
                  }`}
                  style={{ width: `${percentUsed}%` }}
                />
              </div>
            </div>
          )}

          {isMock && (
            <span
              className="hidden items-center gap-1.5 rounded-lg border border-amber-300/70 bg-amber-500/10 px-2.5 py-1 text-[11px] font-semibold text-amber-800 dark:border-amber-500/30 dark:text-amber-300 lg:flex"
              title={t('header.sampleDataTitle')}
            >
              <Sparkles className="h-3 w-3" aria-hidden="true" />
              {t('header.sampleData')}
            </span>
          )}

          <LanguageSwitcher />

          {user && (
            <div className="flex items-center gap-2.5">
              <div className="flex items-center gap-2 rounded-xl border border-white/60 bg-white/60 px-3 py-1.5 text-xs text-slate-700 shadow-xs backdrop-blur-md dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-200">
                {code ? (
                  <KeyRound className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
                ) : (
                  <UserIcon className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
                )}
                <span
                  className="max-w-[130px] truncate sm:max-w-[200px]"
                  title={
                    code
                      ? t('header.expires', { date: new Date(code.expires_at).toLocaleString(i18n.resolvedLanguage) })
                      : user.user
                  }
                >
                  {code ? t('header.accessCode', { id: code.id }) : user.provider === 'none' ? t('header.localMode') : user.user}
                </span>
              </div>
              {user.logout_url && (
                <a
                  href={user.logout_url}
                  className="flex items-center gap-1 rounded-xl border border-white/60 bg-white/40 p-1.5 text-xs text-slate-600 shadow-xs backdrop-blur-md transition-all hover:bg-white/80 hover:text-slate-900 dark:border-white/10 dark:bg-slate-800/40 dark:text-slate-400 dark:hover:bg-slate-800/80 dark:hover:text-white"
                  aria-label={t('header.signOut')}
                >
                  <LogOut className="h-3.5 w-3.5" aria-hidden="true" />
                  <span className="hidden md:inline">{t('header.signOut')}</span>
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
