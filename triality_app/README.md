# Triality App

Dedicated FastAPI frontend/backend app for running Triality runtime modules from a professional web UI.

## Run

```bash
uvicorn triality_app.main:app --host 127.0.0.1 --port 8510 --reload
```

Open `http://127.0.0.1:8510`.

## What it does

- Serves a polished static frontend modeled on the provided Triality design
- Routes prompts into Triality runtime modules or templates
- Executes the real Triality SDK, not mock placeholder logic
- Computes domain-specific observables (peak values, safety margins, pass/fail verdicts) via the Observable Layer
- Returns structured config, metrics, observables, field summaries, warnings, and raw payload previews
