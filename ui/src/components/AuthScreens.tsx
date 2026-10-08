import { useState, type FormEvent, type ReactNode } from 'react';
import { ArrowRight, KeyRound, Lock, LogOut, RefreshCw, ServerCrash, ShieldX } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { AuthError } from '../types';
import { LanguageSwitcher } from './LanguageSwitcher';
import { redeemAccessCode } from '../api';
import { PrivacyNotice } from './Footer';

interface AuthScreensProps {
  error: AuthError;
  onSignedIn: () => void;
}

function Card({ icon, tone, title, children }: { icon: ReactNode; tone: 'blue' | 'red'; title: string; children: ReactNode }) {
  const iconTone =
    tone === 'blue'
      ? 'from-sky-600 to-indigo-600 shadow-sky-500/25'
      : 'from-rose-600 to-amber-600 shadow-rose-500/25';
  return (
    <main className="relative flex min-h-screen items-center justify-center bg-slate-50 px-4 py-12 dark:bg-slate-950">
      <div className="glass-panel w-full max-w-md rounded-3xl p-8 text-center shadow-xl">
        <div className="mb-4 flex justify-end">
          <LanguageSwitcher />
        </div>
        <div className={`mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-tr text-white shadow-lg ${iconTone}`}>
          {icon}
        </div>
        <h1 className="mt-6 text-xl font-bold tracking-tight text-slate-900 dark:text-white">{title}</h1>
        {children}
        <div className="mt-6 border-t border-slate-200/70 pt-4 text-xs text-slate-500 dark:border-white/10 dark:text-slate-400">
          <PrivacyNotice />
        </div>
      </div>
    </main>
  );
}

const secondaryButton =
  'inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white/70 px-4 py-2.5 text-xs font-semibold text-slate-700 transition-all hover:bg-white dark:border-white/10 dark:bg-slate-800/60 dark:text-slate-200 dark:hover:bg-slate-800 cursor-pointer';
const primaryButton =
  'inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-600 via-indigo-600 to-sky-700 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-sky-600/25 transition-all hover:from-sky-500 hover:to-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-600 disabled:cursor-not-allowed disabled:opacity-60';

function AccessCodeForm({ onSignedIn }: { onSignedIn: () => void }) {
  const { t } = useTranslation();
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await redeemAccessCode(code.trim());
      onSignedIn();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t('auth.codeRejected'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="mt-6 flex flex-col gap-3 text-left">
      <label htmlFor="access-code" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
        {t('auth.codeLabel')}
      </label>
      <input
        id="access-code"
        name="access-code"
        autoComplete="off"
        spellCheck={false}
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder="IE-…"
        className="w-full rounded-xl border border-slate-300 bg-white/80 px-3.5 py-2.5 font-mono text-sm text-slate-900 outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30 dark:border-white/15 dark:bg-slate-900/70 dark:text-white"
      />
      {error && (
        <p role="alert" className="text-xs font-medium text-rose-600 dark:text-rose-400">
          {error}
        </p>
      )}
      <button type="submit" disabled={busy || !code.trim()} className={primaryButton}>
        <span>{busy ? t('auth.checking') : t('auth.continue')}</span>
        <ArrowRight className="h-4 w-4" aria-hidden="true" />
      </button>
      <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        {t('auth.codeHelp')}
      </p>
    </form>
  );
}

export function AuthScreens({ error, onSignedIn }: AuthScreensProps) {
  const { t } = useTranslation();
  if (error.status === 401 && error.method === 'code') {
    return (
      <Card icon={<KeyRound className="h-6 w-6" aria-hidden="true" />} tone="blue" title={t('auth.codeTitle')}>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
          {t('auth.codeIntro')}
        </p>
        <AccessCodeForm onSignedIn={onSignedIn} />
      </Card>
    );
  }

  if (error.status === 401) {
    return (
      <Card icon={<Lock className="h-6 w-6" aria-hidden="true" />} tone="blue" title={t('auth.signInTitle')}>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
          {t('auth.signInIntro')}
        </p>
        <div className="mt-7 flex flex-col gap-3">
          {error.login_url && (
            <a href={error.login_url} className={primaryButton}>
              <span>{t('auth.signIn')}</span>
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </a>
          )}
        </div>
      </Card>
    );
  }

  if (error.status === 403) {
    return (
      <Card icon={<ShieldX className="h-6 w-6" aria-hidden="true" />} tone="red" title={t('auth.refusedTitle')}>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{error.detail}</p>
        <div className="mt-7 flex flex-col gap-3">
          {error.logout_url && (
            <a href={error.logout_url} className={secondaryButton}>
              <LogOut className="h-4 w-4" aria-hidden="true" />
              <span>{t('header.signOut')}</span>
            </a>
          )}
        </div>
      </Card>
    );
  }

  return (
    <Card icon={<ServerCrash className="h-6 w-6" aria-hidden="true" />} tone="red" title={t('auth.unavailableTitle')}>
      <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{error.detail}</p>
      <div className="mt-7 flex flex-col gap-3">
        <button type="button" onClick={onSignedIn} className={secondaryButton}>
          <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
          <span>{t('auth.tryAgain')}</span>
        </button>
      </div>
    </Card>
  );
}
