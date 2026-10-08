import React from 'react';
import { Check, Lock } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export type StepState = 'locked' | 'active' | 'complete';

export interface StepItem {
  id: number;
  title: string;
  instruction: string;
  state: StepState;
}

interface WorkflowStepperProps {
  steps: StepItem[];
  activeStep: number;
  onSelectStep: (stepNumber: number) => void;
}

export const WorkflowStepper: React.FC<WorkflowStepperProps> = ({
  steps,
  activeStep,
  onSelectStep
}) => {
  const { t } = useTranslation();
  return (
    <nav
      aria-label={t('stepper.label')}
      className="sticky top-[57px] z-20 border-b border-white/50 bg-white/70 px-4 py-3 backdrop-blur-xl shadow-xs dark:border-white/10 dark:bg-slate-900/70"
    >
      <div className="mx-auto max-w-7xl">
        <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4 md:gap-4">
          {steps.map((step) => {
            const isCurrent = step.id === activeStep;
            const isClickable = step.state !== 'locked' || isCurrent;

            return (
              <li key={step.id}>
                <button
                  type="button"
                  onClick={() => {
                    if (isClickable) onSelectStep(step.id);
                  }}
                  disabled={!isClickable}
                  aria-current={isCurrent ? 'step' : undefined}
                  className={`group flex w-full flex-col items-start rounded-2xl border p-3 text-left transition-all ${
                    isCurrent
                      ? 'border-sky-400/70 bg-gradient-to-br from-sky-500/15 via-indigo-500/10 to-teal-500/10 shadow-md shadow-sky-500/10 backdrop-blur-md dark:border-sky-400/40 dark:from-sky-500/20 dark:via-indigo-500/15 ring-1 ring-sky-400/40'
                      : step.state === 'complete'
                      ? 'border-white/80 bg-white/60 hover:bg-white/90 hover:border-emerald-300 dark:border-white/10 dark:bg-slate-800/50 dark:hover:bg-slate-800/80 shadow-xs backdrop-blur-md cursor-pointer'
                      : 'border-white/40 bg-white/30 opacity-60 dark:border-white/5 dark:bg-slate-900/30 backdrop-blur-sm cursor-not-allowed'
                  }`}
                >
                  <div className="flex w-full items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-bold shadow-xs ${
                          step.state === 'complete'
                            ? 'bg-gradient-to-tr from-emerald-600 to-teal-500 text-white'
                            : isCurrent
                            ? 'bg-gradient-to-tr from-sky-600 to-indigo-600 text-white shadow-sky-500/25 ring-2 ring-white/50'
                            : 'bg-slate-200 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                        }`}
                      >
                        {step.state === 'complete' ? (
                          <Check className="h-3 w-3 stroke-[3]" aria-hidden="true" />
                        ) : step.state === 'locked' ? (
                          <Lock className="h-2.5 w-2.5" aria-hidden="true" />
                        ) : (
                          step.id
                        )}
                      </span>
                      <span
                        className={`text-xs font-semibold tracking-tight ${
                          isCurrent
                            ? 'text-sky-950 dark:text-sky-200 font-bold'
                            : step.state === 'complete'
                            ? 'text-slate-900 dark:text-slate-100'
                            : 'text-slate-500 dark:text-slate-400'
                        }`}
                      >
                        {step.title}
                      </span>
                    </div>

                    <span
                      className={`text-[10px] font-medium uppercase tracking-wider ${
                        step.state === 'complete'
                          ? 'text-emerald-700 dark:text-emerald-400'
                          : isCurrent
                          ? 'text-sky-700 dark:text-sky-300 font-semibold'
                          : 'text-slate-400 dark:text-slate-500'
                      }`}
                    >
                      {t(`stepper.state.${step.state}`)}
                    </span>
                  </div>

                  <p
                    className={`mt-1 line-clamp-1 text-[11px] ${
                      isCurrent
                        ? 'text-sky-900/80 dark:text-sky-300/80'
                        : 'text-slate-500 dark:text-slate-400'
                    }`}
                  >
                    {step.instruction}
                  </p>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </nav>
  );
};
