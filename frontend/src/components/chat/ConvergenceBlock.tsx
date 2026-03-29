import { Target, TrendingUp, AlertTriangle, CheckCircle2, Search, Calculator } from 'lucide-react';
import type {
  GoalExtractedEvent,
  AnalyticalEstimateEvent,
  ConvergenceStepEvent,
  GoalSatisfiedEvent,
  ConvergenceAdaptingEvent,
  ConvergenceStalledEvent,
  ConvergenceExhaustedEvent,
} from '../../types';

/* ------------------------------------------------------------------ */
/*  Goal Extracted                                                     */
/* ------------------------------------------------------------------ */
export function GoalExtractedBlock({ data }: { data: GoalExtractedEvent }) {
  const g = data.goal;
  const goalLabel =
    g.goal_type === 'find_threshold'
      ? `Find ${g.search_variable ?? 'parameter'} where ${g.metric} ${g.operator ?? '>'} ${g.threshold ?? '?'} ${g.unit ?? ''}`
      : g.goal_type === 'maximize'
        ? `Maximize ${g.metric}`
        : g.goal_type === 'minimize'
          ? `Minimize ${g.metric}`
          : g.goal_type === 'compare_select'
            ? `Compare & select best option`
            : `Characterize ${g.metric}`;

  return (
    <div className="flex items-start gap-2 py-1.5 px-3 rounded-lg bg-cyan-500/5 border border-cyan-500/10 animate-fade-in">
      <Target className="w-3.5 h-3.5 text-cyan-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-[11px] font-mono text-cyan-400/80 font-medium">Goal Identified</p>
        <p className="text-[11px] font-mono text-cyan-300/60 mt-0.5">{goalLabel}</p>
        {g.search_bounds && (
          <p className="text-[10px] font-mono text-cyan-300/40 mt-0.5">
            Search range: [{g.search_bounds[0]?.toLocaleString()}, {g.search_bounds[1]?.toLocaleString()}]
            {g.module_name ? ` | Module: ${g.module_name}` : ''}
          </p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Analytical Estimate                                                */
/* ------------------------------------------------------------------ */
export function AnalyticalEstimateBlock({ data }: { data: AnalyticalEstimateEvent }) {
  const est = data.estimate;
  if (!est?.estimate) return null;

  return (
    <div className="flex items-start gap-2 py-1.5 px-3 rounded-lg bg-purple-500/5 border border-purple-500/10 animate-fade-in">
      <Calculator className="w-3.5 h-3.5 text-purple-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-[11px] font-mono text-purple-400/80 font-medium">Analytical Pre-Flight</p>
        <p className="text-[11px] font-mono text-purple-300/60 mt-0.5">
          Estimate: <span className="text-purple-300/90 font-medium">{est.estimate.toLocaleString()} {est.unit ?? ''}</span>
          <span className="text-purple-300/40 ml-2">({est.confidence})</span>
        </p>
        {est.governing_equation && (
          <p className="text-[10px] font-mono text-purple-300/40 mt-0.5 italic">{est.governing_equation}</p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Convergence Step (iteration progress)                              */
/* ------------------------------------------------------------------ */
export function ConvergenceStepBlock({ data }: { data: ConvergenceStepEvent }) {
  const ev = data.evaluation;
  const pct = Math.round((data.iteration / data.max_iterations) * 100);

  return (
    <div className="flex items-start gap-2 py-1 px-3 animate-fade-in">
      <Search className="w-3 h-3 text-[#6ee7b7]/50 mt-0.5 shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-[10px] font-mono text-[#6ee7b7]/50">
            Iteration {data.iteration}/{data.max_iterations}
          </p>
          {/* Mini progress bar */}
          <div className="flex-1 max-w-24 h-1 rounded-full bg-white/5 overflow-hidden">
            <div
              className="h-full rounded-full bg-[#6ee7b7]/30 transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-[10px] font-mono text-white/30">{data.total_solver_runs} runs</p>
        </div>
        <p className="text-[10px] font-mono text-white/40 mt-0.5">{ev.details}</p>
        {ev.closest_value != null && !ev.satisfied && (
          <p className="text-[10px] font-mono text-amber-400/50 mt-0.5">
            Closest: {ev.closest_value.toLocaleString(undefined, { maximumSignificantDigits: 4 })}
            {ev.gap_ratio != null ? ` (${ev.gap_ratio.toFixed(2)}x from target)` : ''}
          </p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Goal Satisfied (the answer!)                                       */
/* ------------------------------------------------------------------ */
export function GoalSatisfiedBlock({ data }: { data: GoalSatisfiedEvent }) {
  return (
    <div className="flex items-start gap-2 py-2 px-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 animate-fade-in">
      <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-[11px] font-mono text-emerald-400 font-bold">Answer Found</p>
        <p className="text-sm font-mono text-emerald-300 mt-1 font-bold">
          {data.answer?.toLocaleString(undefined, { maximumSignificantDigits: 6 })} {data.answer_unit ?? ''}
        </p>
        <div className="flex items-center gap-3 mt-1">
          {data.accuracy_pct != null && (
            <span className="text-[10px] font-mono text-emerald-400/60">
              Accuracy: ±{data.accuracy_pct.toFixed(2)}%
            </span>
          )}
          <span className="text-[10px] font-mono text-emerald-400/40">
            {data.iterations_used} iteration{data.iterations_used !== 1 ? 's' : ''} | {data.total_solver_runs} solver runs
          </span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Convergence Adapting                                               */
/* ------------------------------------------------------------------ */
export function ConvergenceAdaptingBlock({ data }: { data: ConvergenceAdaptingEvent }) {
  const actionLabel =
    data.action === 'expand_range' ? 'Expanding search range'
    : data.action === 'bisect' ? 'Bisecting for precision'
    : data.action === 'refine' ? 'Refining around answer'
    : data.action;

  return (
    <div className="flex items-center gap-2 py-1 px-3 animate-fade-in">
      <TrendingUp className="w-3 h-3 text-amber-400/50 shrink-0" />
      <p className="text-[10px] font-mono text-amber-400/50">
        {actionLabel}
        {data.next_bounds ? ` → [${data.next_bounds[0]?.toLocaleString()}, ${data.next_bounds[1]?.toLocaleString()}]` : ''}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Convergence Stalled                                                */
/* ------------------------------------------------------------------ */
export function ConvergenceStalledBlock({ data }: { data: ConvergenceStalledEvent }) {
  return (
    <div className="flex items-start gap-2 py-1.5 px-3 rounded-lg bg-amber-500/5 border border-amber-500/10 animate-fade-in">
      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-[11px] font-mono text-amber-400/80 font-medium">Convergence Stalled</p>
        <p className="text-[10px] font-mono text-amber-300/50 mt-0.5">{data.reason}</p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Convergence Exhausted                                              */
/* ------------------------------------------------------------------ */
export function ConvergenceExhaustedBlock({ data }: { data: ConvergenceExhaustedEvent }) {
  return (
    <div className="flex items-start gap-2 py-1.5 px-3 rounded-lg bg-amber-500/5 border border-amber-500/10 animate-fade-in">
      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-[11px] font-mono text-amber-400/80 font-medium">Max Iterations Reached</p>
        <p className="text-[10px] font-mono text-amber-300/50 mt-0.5">
          {data.iterations_used} iterations, {data.total_solver_runs} solver runs — could not fully converge.
        </p>
      </div>
    </div>
  );
}
