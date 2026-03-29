import Plot from 'react-plotly.js';
import type { ToolResultPayload } from '../../types';

interface Props {
  result: ToolResultPayload;
}

export function OptimizationChart({ result }: Props) {
  const evals = result.evaluations || [];
  if (evals.length < 2) return null;

  const xVals = evals.map((e) => e.param_value);
  const yVals = evals.map((e) => e.metric);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [
    {
      x: xVals,
      y: yVals,
      type: 'scatter',
      mode: 'markers',
      name: 'Evaluations',
      marker: { size: 7, color: 'rgba(110, 231, 183, 0.5)', line: { width: 1, color: '#6ee7b7' } },
      hovertemplate: `${result.param_path}: %{x:.4g}<br>${result.objective_field}: %{y:.4g}<extra></extra>`,
    },
  ];

  if (result.optimal_param_value != null && result.optimal_metric != null) {
    traces.push({
      x: [result.optimal_param_value],
      y: [result.optimal_metric],
      type: 'scatter',
      mode: 'markers',
      name: 'Optimal',
      marker: {
        size: 14,
        color: '#6ee7b7',
        symbol: 'star',
        line: { width: 2, color: '#fff' },
      },
      hovertemplate: `OPTIMAL<br>${result.param_path}: %{x:.4g}<br>${result.objective_field}: %{y:.4g}<extra></extra>`,
    });
  }

  return (
    <div className="mt-2">
      <p className="text-[10px] font-mono text-white/30 mb-1">
        Optimization: {result.param_path} → {result.objective_field} ({result.objective})
      </p>
      <Plot
        data={traces}
        layout={{
          width: 500,
          height: 280,
          margin: { l: 60, r: 20, t: 10, b: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: { text: result.param_path || 'parameter', font: { color: 'rgba(255,255,255,0.4)', size: 11 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            tickformat: '.3g',
          },
          yaxis: {
            title: { text: result.objective_field || 'metric', font: { color: 'rgba(255,255,255,0.4)', size: 11 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            tickformat: '.3g',
          },
          showlegend: true,
          legend: {
            font: { color: 'rgba(255,255,255,0.4)', size: 9 },
            bgcolor: 'transparent',
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
