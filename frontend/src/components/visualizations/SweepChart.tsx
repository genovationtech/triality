import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { ToolResultPayload } from '../../types';

interface Props {
  result: ToolResultPayload;
}

const COLORS = ['#22d3ee', '#6ee7b7', '#f59e0b', '#a78bfa', '#f87171'];

/** Metadata scalars worth plotting across a sweep (key → display label). */
const METADATA_SCALARS: Record<string, string> = {
  current_density: 'current density (A/cm²)',
  built_in_potential: 'built-in potential (V)',
  max_field: 'max E-field (V/cm)',
  depletion_width: 'depletion width (cm)',
};

/** Fields where log scale is the right default. */
const LOG_DEFAULT = new Set(['current_density', 'electron_density', 'hole_density']);

export function SweepChart({ result }: Props) {
  const results = result.results || [];
  const paramPath = result.param_path || 'param';

  if (results.length < 2) return null;

  // Detect which metadata scalars vary across the sweep
  const metadataKeys = useMemo(() => {
    const keys: string[] = [];
    for (const key of Object.keys(METADATA_SCALARS)) {
      const vals = results
        .filter((r: any) => r.success && r.metadata?.[key] != null)
        .map((r: any) => r.metadata[key] as number);
      if (vals.length >= 2) keys.push(key);
    }
    return keys;
  }, [results]);

  // Build field traces (means)
  const fieldNames = Object.keys(results[0]?.fields || {});

  // Determine if any trace suggests log scale
  const hasLogCandidate = useMemo(() => {
    for (const key of metadataKeys) {
      if (LOG_DEFAULT.has(key)) return true;
    }
    for (const fname of fieldNames) {
      if (LOG_DEFAULT.has(fname)) return true;
    }
    return false;
  }, [metadataKeys, fieldNames]);

  const [logScale, setLogScale] = useState(hasLogCandidate);

  // Priority: metadata scalars first (the real physics), then field means
  const traces: any[] = [];
  let colorIdx = 0;

  for (const key of metadataKeys) {
    const xVals: number[] = [];
    const yVals: number[] = [];
    for (const r of results) {
      if (r.success && r.metadata?.[key] != null) {
        xVals.push(r.param_value);
        const v = r.metadata[key] as number;
        yVals.push(logScale ? Math.abs(v) : v);
      }
    }
    traces.push({
      x: xVals,
      y: yVals,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: METADATA_SCALARS[key],
      line: { color: COLORS[colorIdx % COLORS.length], width: 2.5 },
      marker: { size: 7, color: COLORS[colorIdx % COLORS.length] },
      hovertemplate: `${paramPath}: %{x:.4g}<br>${METADATA_SCALARS[key]}: %{y:.4g}<extra></extra>`,
    });
    colorIdx++;
  }

  // Field mean traces — de-emphasize if we have metadata scalars
  const fieldTraceStyle = metadataKeys.length > 0
    ? { width: 1.5, dash: 'dot' as const, size: 4, opacity: 0.5 }
    : { width: 2, dash: 'solid' as const, size: 6, opacity: 1 };

  for (const fname of fieldNames.slice(0, 5 - metadataKeys.length)) {
    const xVals: number[] = [];
    const yVals: number[] = [];
    for (const r of results) {
      if (r.success && r.fields?.[fname]) {
        xVals.push(r.param_value);
        const v = r.fields[fname].mean;
        yVals.push(logScale ? Math.abs(v) : v);
      }
    }
    traces.push({
      x: xVals,
      y: yVals,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: `${fname} (mean)`,
      line: {
        color: COLORS[colorIdx % COLORS.length],
        width: fieldTraceStyle.width,
        dash: fieldTraceStyle.dash,
      },
      marker: { size: fieldTraceStyle.size, color: COLORS[colorIdx % COLORS.length] },
      opacity: fieldTraceStyle.opacity,
      hovertemplate: `${paramPath}: %{x:.4g}<br>${fname} mean: %{y:.4g}<extra></extra>`,
    });
    colorIdx++;
  }

  if (!traces.length) return null;

  return (
    <div className="mt-2">
      <div className="flex items-center justify-end mb-1">
        <button
          onClick={() => setLogScale(!logScale)}
          className={`text-[9px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
            logScale
              ? 'border-cyan-400/30 text-cyan-400/60 bg-cyan-400/5'
              : 'border-white/10 text-white/25 hover:text-white/40'
          }`}
        >
          log
        </button>
      </div>
      <Plot
        data={traces}
        layout={{
          width: 520,
          height: 300,
          margin: { l: 65, r: 20, t: 10, b: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: { text: paramPath, font: { color: 'rgba(255,255,255,0.4)', size: 11 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            tickformat: '.3g',
          },
          yaxis: {
            title: {
              text: metadataKeys.length === 1
                ? METADATA_SCALARS[metadataKeys[0]]
                : 'value',
              font: { color: 'rgba(255,255,255,0.4)', size: 11 },
            },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            tickformat: '.3g',
            type: logScale ? 'log' : ('linear' as any),
          },
          legend: {
            font: { color: 'rgba(255,255,255,0.4)', size: 9, family: 'JetBrains Mono, monospace' },
            bgcolor: 'transparent',
            x: 1,
            xanchor: 'right' as const,
            y: 1,
          },
          showlegend: traces.length > 1,
        }}
        config={{
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
          displaylogo: false,
          responsive: true,
        }}
        className="rounded-lg border border-white/[0.06] overflow-hidden"
      />
    </div>
  );
}
