import { useState } from 'react';
import { Play, Check, X, ChevronDown, Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import { Badge } from '../ui/Badge';
import { MetricCard } from '../ui/MetricCard';
import { FieldHeatmap } from '../visualizations/FieldHeatmap';
import { FieldLinePlot } from '../visualizations/FieldLinePlot';
import { IVCurvePlot } from '../visualizations/IVCurvePlot';
import { SweepChart } from '../visualizations/SweepChart';
import { OptimizationChart } from '../visualizations/OptimizationChart';
import { formatNumber } from '../../utils/format';
import type { ToolStartEvent, ToolResultEvent, ToolProgressEvent, ToolResultPayload } from '../../types';

interface Props {
  start: ToolStartEvent;
  result?: ToolResultEvent;
  progress?: ToolProgressEvent;
}

function ProgressBar({ step, total, detail }: { step: number; total: number; detail: Record<string, unknown> }) {
  const pct = total > 0 ? Math.min((step / total) * 100, 100) : 0;
  const detailStr = detail.detail as string | undefined;
  const label = detail.label as string | undefined;
  const paramVal = detail.param_value as number | undefined;
  const elapsed = detail.elapsed_s as number | undefined;

  // Build a human-readable status line
  let statusText: string;
  if (detailStr) {
    // run_module sends a full detail string like "Step 120/500 — 10s elapsed"
    statusText = detailStr;
  } else {
    statusText = `Step ${step}/${total}`;
    if (label) statusText += ` — ${label}`;
    else if (paramVal !== undefined) statusText += ` — val=${formatNumber(paramVal)}`;
    if (elapsed !== undefined) statusText += ` (${formatNumber(elapsed)}s)`;
  }

  return (
    <div className="mt-1.5 mb-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] font-mono text-[#6ee7b7]/50">{statusText}</span>
        <span className="text-[10px] font-mono text-white/30">{Math.round(pct)}%</span>
      </div>
      <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-[#6ee7b7]/60 to-cyan-400/60 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function ToolBlock({ start, result, progress }: Props) {
  const [open, setOpen] = useState(true);
  const isComplete = !!result;
  const success = result?.success ?? false;
  const isRunning = !isComplete;

  return (
    <div className="rounded-xl bg-[#0d0f14]/50 border border-white/[0.04] overflow-hidden animate-fade-in">
      {/* Header */}
      <button
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-white/[0.02] transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-[#6ee7b7]/10 flex items-center justify-center">
            {isRunning ? (
              progress ? (
                <Loader2 className="w-2.5 h-2.5 text-[#6ee7b7]/60 animate-spin" />
              ) : (
                <Play className="w-2.5 h-2.5 text-[#6ee7b7]/60" />
              )
            ) : success ? (
              <Check className="w-2.5 h-2.5 text-[#6ee7b7]/60" />
            ) : (
              <X className="w-2.5 h-2.5 text-red-400/60" />
            )}
          </div>
          <span className="text-[11px] font-mono text-white/50">{start.tool}</span>
          <span
            className={`text-[10px] font-mono ${
              isRunning
                ? progress
                  ? 'text-[#6ee7b7]/50'
                  : 'text-yellow-400/50'
                : success
                  ? 'text-[#6ee7b7]/50'
                  : 'text-red-400/50'
            }`}
          >
            {isRunning
              ? progress
                ? `step ${progress.step}/${progress.total}`
                : 'running...'
              : `${result.elapsed_s}s`}
          </span>
        </div>
        <ChevronDown
          className={`w-3 h-3 text-white/20 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Body */}
      <div
        className={`overflow-hidden transition-all duration-300 ${
          open ? 'max-h-[4000px] pb-3' : 'max-h-0'
        }`}
      >
        <div className="px-3">
          {/* Progress bar (while running) */}
          {isRunning && progress && (
            <ProgressBar step={progress.step} total={progress.total} detail={progress.detail} />
          )}

          {/* Args */}
          <p className="text-[10px] font-mono text-white/20 mb-1">Args:</p>
          <pre className="text-[10px] font-mono text-white/30 bg-[#07080a]/50 rounded-lg p-2 overflow-auto max-h-24">
            {JSON.stringify(start.args, null, 2)}
          </pre>

          {/* Result */}
          {result && (
            <div className="mt-2 pt-2 border-t border-white/[0.04]">
              <ToolResultContent tool={start.tool} result={result.result} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ToolResultContent({ tool, result }: { tool: string; result: ToolResultPayload }) {
  if (tool === 'run_module') {
    return <RunModuleResult result={result} />;
  }
  if (tool === 'sweep_parameter') {
    return <SweepResult result={result} />;
  }
  if (tool === 'optimize_parameter') {
    return <OptimizeResult result={result} />;
  }
  if (tool === 'compare_scenarios') {
    return <CompareResult result={result} />;
  }
  if (tool === 'run_uncertainty_quantification') {
    return <UQResult result={result} />;
  }
  if (tool === 'chain_modules') {
    return <ChainResult result={result} />;
  }
  if (tool === 'list_modules' || tool === 'describe_module') {
    return (
      <pre className="text-[11px] font-mono text-white/45 bg-[#0d0f14]/40 rounded-lg p-3 overflow-auto max-h-60">
        {JSON.stringify(result, null, 2)}
      </pre>
    );
  }
  return (
    <pre className="text-[11px] font-mono text-white/45 bg-[#0d0f14]/40 rounded-lg p-3 overflow-auto max-h-40">
      {JSON.stringify(result, null, 2).slice(0, 2000)}
    </pre>
  );
}

function RunModuleResult({ result }: { result: ToolResultPayload }) {
  const success = result.success;
  const fields = result.fields_stats || {};
  const stateFields = result.state?.fields || {};
  const metadata = result.state?.metadata || {};
  const position = metadata.position_cm as number[] | undefined;
  const ivVoltages = metadata.iv_voltages as number[] | undefined;
  const ivCurrents = metadata.iv_currents as number[] | undefined;

  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Badge variant={success ? 'full' : 'fail'}>{success ? 'Success' : 'Failed'}</Badge>
        <span className="text-[11px] font-mono text-white/40">
          {result.module_name} — {formatNumber(result.elapsed_time_s || result._elapsed_s || 0)}s
          {metadata.temperature ? ` · ${metadata.temperature} K` : ''}
        </span>
      </div>
      {result.error && <p className="text-xs text-red-300/70">{result.error}</p>}
      {Object.keys(fields).length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-2">
          {Object.entries(fields).map(([fname, stats]) => (
            <MetricCard
              key={fname}
              label={fname}
              value={`[${formatNumber(stats.min)}, ${formatNumber(stats.max)}]`}
              sub={`mean: ${formatNumber(stats.mean)}`}
            />
          ))}
        </div>
      )}
      {/* I-V characteristic — the gold standard for PN junction analysis */}
      {ivVoltages && ivCurrents && ivVoltages.length >= 2 && (
        <IVCurvePlot
          voltages={ivVoltages}
          currents={ivCurrents}
          temperature={metadata.temperature as number | undefined}
          materialName={metadata.material_name as string | undefined}
        />
      )}
      {/* Spatial field plots with physical position axis */}
      {Object.entries(stateFields).map(([fname, fdata]) => {
        if (!fdata.data) return null;
        if (Array.isArray(fdata.data[0])) {
          return (
            <FieldHeatmap
              key={fname}
              fieldName={fname}
              data={fdata.data as number[][]}
              unit={fdata.unit}
            />
          );
        }
        return (
          <FieldLinePlot
            key={fname}
            fieldName={fname}
            data={fdata.data as number[]}
            unit={fdata.unit}
            position={position}
          />
        );
      })}
    </>
  );
}

function SweepResult({ result }: { result: ToolResultPayload }) {
  const n = (result.results || []).length;
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Badge variant="full">Sweep</Badge>
        <span className="text-[11px] font-mono text-white/40">
          {result.param_path} — {n} points
        </span>
      </div>
      <SweepChart result={result} />
    </>
  );
}

function OptimizeResult({ result }: { result: ToolResultPayload }) {
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Badge variant="full">Optimized</Badge>
        <span className="text-[11px] font-mono text-white/40">{result.param_path}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 mt-2">
        <MetricCard
          label="Optimal Value"
          value={formatNumber(result.optimal_param_value ?? 0)}
        />
        <MetricCard
          label="Metric"
          value={formatNumber(result.optimal_metric ?? 0)}
        />
      </div>
      <OptimizationChart result={result} />
    </>
  );
}

function CompareResult({ result }: { result: ToolResultPayload }) {
  const comp = result.comparison || [];
  const allFields = new Set<string>();
  comp.forEach((c) => Object.keys(c.fields || {}).forEach((f) => allFields.add(f)));

  const fieldList = [...allFields].slice(0, 4);
  const labels = comp.map((c) => c.label);
  const colors = ['#22d3ee', '#6ee7b7', '#f59e0b', '#a78bfa', '#f87171'];

  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Badge variant="full">Compared</Badge>
        <span className="text-[11px] font-mono text-white/40">{comp.length} scenarios</span>
      </div>
      {fieldList.map((fname) => {
        const values = comp.map((c) => c.fields?.[fname]?.mean ?? 0);
        return (
          <div key={fname} className="mt-2">
            <Plot
              data={[
                {
                  x: labels,
                  y: values,
                  type: 'bar',
                  marker: {
                    color: comp.map((_, i) => colors[i % colors.length]),
                    line: { width: 0 },
                  },
                  hovertemplate: '%{x}<br>' + fname + ': %{y:.4g}<extra></extra>',
                },
              ]}
              layout={{
                width: 500,
                height: 220,
                margin: { l: 60, r: 20, t: 25, b: 50 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                title: {
                  text: fname,
                  font: { color: 'rgba(255,255,255,0.4)', size: 11, family: 'JetBrains Mono, monospace' },
                  x: 0.02,
                  xanchor: 'left',
                },
                xaxis: {
                  tickfont: { color: 'rgba(255,255,255,0.4)', size: 9 },
                  gridcolor: 'rgba(255,255,255,0.04)',
                },
                yaxis: {
                  tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
                  gridcolor: 'rgba(255,255,255,0.04)',
                  tickformat: '.3g',
                },
                showlegend: false,
              }}
              config={{
                displayModeBar: false,
                responsive: true,
              }}
              className="rounded-lg border border-white/[0.06] overflow-hidden"
            />
          </div>
        );
      })}
    </>
  );
}

function UQResult({ result }: { result: ToolResultPayload }) {
  const stats = result.statistics || {};
  const entries = Object.entries(stats);

  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Badge variant="full">UQ</Badge>
        <span className="text-[11px] font-mono text-white/40">{result.n_samples} samples</span>
      </div>
      {entries.map(([key, s]) => (
        <MetricCard
          key={key}
          label={key}
          value={`${formatNumber(s.mean)} \u00B1 ${formatNumber(s.std)} (CV=${s.cv_percent.toFixed(1)}%)`}
          sub={`90% CI: [${formatNumber(s.p5)}, ${formatNumber(s.p95)}]`}
        />
      ))}
      {entries.length > 0 && (
        <div className="mt-2">
          <Plot
            data={entries.map(([key, s], idx) => ({
              x: [key],
              y: [s.mean],
              error_y: {
                type: 'data' as const,
                symmetric: false,
                array: [s.p95 - s.mean],
                arrayminus: [s.mean - s.p5],
                color: 'rgba(110, 231, 183, 0.5)',
                thickness: 2,
              },
              type: 'bar' as const,
              name: key,
              marker: {
                color: ['#22d3ee', '#6ee7b7', '#f59e0b', '#a78bfa', '#f87171'][idx % 5],
              },
              hovertemplate: `${key}<br>mean: %{y:.4g}<br>90% CI: [${formatNumber(s.p5)}, ${formatNumber(s.p95)}]<extra></extra>`,
            }))}
            layout={{
              width: 500,
              height: 240,
              margin: { l: 60, r: 20, t: 10, b: 50 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              xaxis: {
                tickfont: { color: 'rgba(255,255,255,0.4)', size: 9 },
                gridcolor: 'rgba(255,255,255,0.04)',
              },
              yaxis: {
                title: { text: 'mean \u00B1 90% CI', font: { color: 'rgba(255,255,255,0.3)', size: 10 } },
                tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
                gridcolor: 'rgba(255,255,255,0.04)',
                tickformat: '.3g',
              },
              showlegend: false,
              barmode: 'group',
            }}
            config={{ displayModeBar: false, responsive: true }}
            className="rounded-lg border border-white/[0.06] overflow-hidden"
          />
        </div>
      )}
    </>
  );
}

function ChainResult({ result }: { result: ToolResultPayload }) {
  const chain = result.chain_results || [];
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Badge variant="full">Chain</Badge>
        <span className="text-[11px] font-mono text-white/40">{chain.length} steps</span>
      </div>
      {chain.map((step) => {
        const ok = step.success !== false && !step.error;
        return (
          <div
            key={step.step}
            className={`mt-2 pl-3 border-l-2 ${ok ? 'border-[#6ee7b7]/20' : 'border-red-400/20'}`}
          >
            <p className="text-[10px] font-mono text-white/40">
              {step.step_label || step.module_name || `step ${step.step}`}
            </p>
            {step.error && <p className="text-xs text-red-300/60">{step.error}</p>}
          </div>
        );
      })}
    </>
  );
}
