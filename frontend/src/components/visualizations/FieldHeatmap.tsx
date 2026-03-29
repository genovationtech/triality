import Plot from 'react-plotly.js';
import { formatNumber } from '../../utils/format';

interface Props {
  fieldName: string;
  data: number[][];
  unit?: string;
}

export function FieldHeatmap({ fieldName, data, unit }: Props) {
  const rows = data.length;
  const cols = data[0]?.length || 0;
  if (!rows || !cols) return null;

  // Check for degenerate data (mostly zeros = solver likely diverged)
  const flat = data.flat();
  const nonzero = flat.filter((v) => Math.abs(v) > 1e-15).length;
  const coverage = nonzero / flat.length;
  const isDegenerate = coverage < 0.05;

  return (
    <div className="mt-2">
      <p className="text-[10px] font-mono text-white/30 mb-1">
        {fieldName} {unit ? `(${unit})` : ''} — {rows}x{cols}
        {isDegenerate && (
          <span className="ml-2 text-amber-400/60">
            ({(coverage * 100).toFixed(1)}% non-zero — solver may have diverged)
          </span>
        )}
      </p>
      <Plot
        data={[
          {
            z: data,
            type: 'heatmap',
            colorscale: [
              [0, '#0d1b2a'],
              [0.25, '#1b4965'],
              [0.5, '#2a9d8f'],
              [0.75, '#e9c46a'],
              [1, '#e76f51'],
            ],
            colorbar: {
              tickfont: { color: 'rgba(255,255,255,0.4)', size: 9, family: 'JetBrains Mono, monospace' },
              thickness: 12,
              outlinewidth: 0,
              tickformat: '.3g',
            },
            hovertemplate: 'row: %{y}<br>col: %{x}<br>value: %{z:.4g}<extra></extra>',
          },
        ]}
        layout={{
          width: 460,
          height: 340,
          margin: { l: 40, r: 60, t: 10, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: { text: 'x', font: { color: 'rgba(255,255,255,0.3)', size: 10 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
          },
          yaxis: {
            title: { text: 'y', font: { color: 'rgba(255,255,255,0.3)', size: 10 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            autorange: 'reversed',
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
