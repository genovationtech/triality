# Triality Frontend

React + TypeScript web interface for the Triality Agent.

## Stack

- **React 18** with TypeScript
- **Vite** for dev server and builds
- **Tailwind CSS** for styling
- **Plotly.js** for interactive visualizations (field heatmaps, I-V curves, optimization charts)
- **KaTeX** for LaTeX math rendering

## Development

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` with hot module replacement.

## Build

```bash
npm run build
```

Output is written to `dist_temp/`. To deploy, copy the built assets to `triality_app/static/dist/`.

## Structure

```
src/
  components/
    chat/       # Agent message components (thinking, reflection, convergence, etc.)
    layout/     # Header, Sidebar, ChatArea, ChatInput
    ui/         # Badge, MetricCard, Pill
    viz/        # FieldHeatmap, FieldLinePlot, IVCurvePlot, OptimizationChart, SweepChart
  hooks/        # useAgent, useCatalog
  utils/        # Formatting and markdown helpers
  types/        # TypeScript type definitions
  App.tsx       # Root application component
  main.tsx      # Entry point
```
