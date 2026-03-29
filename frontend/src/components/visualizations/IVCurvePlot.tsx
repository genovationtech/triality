import { useState } from 'react';
import Plot from 'react-plotly.js';

interface Props {
  voltages: number[];
  currents: number[];
  temperature?: number;
  materialName?: string;
}

export function IVCurvePlot({ voltages, currents, temperature, materialName }: Props) {
  const [logScale, setLogScale] = useState(true);

  if (voltages.length < 2) return null;

  const plotCurrents = logScale ? currents.map(v => Math.abs(v)) : currents;

  const label = [
    'I–V characteristic',
    materialName && materialName,
    temperature && `${temperature} K`,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between mb-1">
        <p className="text-[10px] font-mono text-cyan-400/50">{label}</p>
        <button
          onClick={() => setLogScale(!logScale)}
          className={`text-[9px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
            logScale
              ? 'border-cyan-400/30 text-cyan-400/60 bg-cyan-400/5'
              : 'border-white/10 text-white/25 hover:text-white/40'
          }`}
        >
          {logScale ? 'semi-log' : 'linear'}
        </button>
      </div>
      <Plot
        data={[
          {
            x: voltages,
            y: plotCurrents,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#22d3ee', width: 2.5 },
            marker: { size: 5, color: '#22d3ee' },
            hovertemplate: 'V: %{x:.3f} V<br>J: %{y:.4g} A/cm²<extra></extra>',
          },
        ]}
        layout={{
          width: 480,
          height: 280,
          margin: { l: 65, r: 20, t: 10, b: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: { text: 'voltage (V)', font: { color: 'rgba(255,255,255,0.4)', size: 11 } },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.08)',
          },
          yaxis: {
            title: {
              text: logScale ? '|J| (A/cm²)' : 'J (A/cm²)',
              font: { color: 'rgba(255,255,255,0.4)', size: 11 },
            },
            tickfont: { color: 'rgba(255,255,255,0.3)', size: 9 },
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.08)',
            tickformat: '.2g',
            type: logScale ? 'log' : 'linear',
          },
        }}
        config={{
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
          displaylogo: false,
          responsive: true,
        }}
        className="rounded-lg border border-cyan-400/[0.08] overflow-hidden"
      />
    </div>
  );
}
