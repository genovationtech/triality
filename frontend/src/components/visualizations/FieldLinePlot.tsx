import { useState } from 'react';
import Plot from 'react-plotly.js';

interface Props {
  fieldName: string;
  data: number[];
  unit?: string;
  position?: number[];
}

const LOG_FIELDS = new Set([
  'electron_density',
  'hole_density',
  'current_density_total',
  'electron_current_density',
  'hole_current_density',
]);

export function FieldLinePlot({ fieldName, data, unit, position }: Props) {
  const shouldDefaultLog = LOG_FIELDS.has(fieldName);
  const [logScale, setLogScale] = useState(shouldDefaultLog);

  if (data.length < 2) return null;

  const xVals = position && position.length === data.length
    ? position.map(v => v * 1e4) // cm → µm for readability
    : data.map((_, i) => i);
  const xLabel = position ? 'position (µm)' : 'index';

  const plotData = logScale ? data.map(v => Math.abs(v)) : data;

  return (
    <div className="mt-2">
      <div className="flex items-center justify-between mb-1">
        <p className="text-[10px] font-mono text-white/30">
          {fieldName} {unit ? `(${unit})` : ''} — {data.length} pts
        </p>
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
        data={[
          {
            x: xVals,
            y: plotData,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6ee7b7', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(110, 231, 183, 0.08)',
            hovertemplate: `${xLabel}: %{x:.4g}<br>${fieldName}: %{y:.4g}<extra></extra>`,
          },
        ]}
        layout={{
          width: 460,
          height: 220,
          margin: { l: 55, r: 20, t: 10, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: { text: xLabel, font: { color: 'rgba(255,255,255,0.3)', size: 10 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
          },
          yaxis: {
            title: { text: fieldName, font: { color: 'rgba(255,255,255,0.3)', size: 10 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            tickformat: '.3g',
            type: logScale ? 'log' : 'linear',
          },
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
