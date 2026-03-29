from __future__ import annotations

import json
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

DEPTH_LABELS = {
    "D0": "package import failed",
    "D1": "package import succeeded",
    "D2": "package + solver module imported",
    "D3": "at least one public callable executed",
    "D4": "at least one workflow-style method executed",
}

PROBE_SCRIPT = r"""
import importlib
import inspect
import json
import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

module_name = os.environ["TRIALITY_MODULE"]
root = Path(os.environ["TRIALITY_ROOT"])
pkg_name = f"triality.{module_name}"
solver_path = root / module_name / "solver.py"

result = {
    "module": module_name,
    "depth": "D0",
    "package_import": "not_attempted",
    "solver_import": "not_attempted",
    "callable_execution": "not_attempted",
    "workflow_execution": "not_attempted",
    "successful_callable": None,
    "successful_workflow": None,
    "notes": [],
}

def short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"

def is_zero_arg_callable(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    required = []
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            continue
        required.append(param)
    return len(required) == 0

def public_members(module):
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        yield name, obj

try:
    pkg = importlib.import_module(pkg_name)
    result["package_import"] = "ok"
    result["depth"] = "D1"
except Exception as exc:
    result["package_import"] = short_error(exc)
    print(json.dumps(result))
    raise SystemExit(0)

try:
    runtime_mod = importlib.import_module("triality.runtime")
    load_module = getattr(runtime_mod, "load_module")
    try:
        handle = load_module(module_name)
        runtime_solver = handle.from_demo_case()
        result["notes"].append("runtime SDK demo case available")
        result["callable_execution"] = "ok"
        result["successful_callable"] = f"load_module('{module_name}').from_demo_case()"
        result["depth"] = "D3"
        runtime_solver.solve()
        result["workflow_execution"] = "ok"
        result["successful_workflow"] = f"load_module('{module_name}').from_demo_case().solve()"
        result["depth"] = "D4"
        print(json.dumps(result))
        raise SystemExit(0)
    except KeyError:
        pass
    except Exception as runtime_exc:
        result["notes"].append(f"runtime SDK path failed: {short_error(runtime_exc)}")
except Exception:
    pass

solver_module = None
if solver_path.exists():
    try:
        solver_module = importlib.import_module(f"{pkg_name}.solver")
        result["solver_import"] = "ok"
        result["depth"] = "D2"
    except Exception as exc:
        result["solver_import"] = short_error(exc)
        result["notes"].append("solver.py exists but failed to import")
else:
    result["notes"].append("no solver.py present")

candidates = []
seen = set()
for source_module in [pkg, solver_module]:
    if source_module is None:
        continue
    for name, obj in public_members(source_module):
        key = (source_module.__name__, name)
        if key in seen:
            continue
        seen.add(key)
        if inspect.isfunction(obj) and getattr(obj, "__module__", None) == source_module.__name__ and is_zero_arg_callable(obj):
            candidates.append(("function", f"{source_module.__name__}.{name}", obj))
        elif inspect.isclass(obj) and getattr(obj, "__module__", None) == source_module.__name__:
            candidates.append(("class", f"{source_module.__name__}.{name}", obj))

workflow_names = ("solve", "run", "analyze", "compute", "simulate", "execute", "evaluate", "route", "predict")

for kind, label, obj in candidates[:20]:
    try:
        if kind == "function":
            obj()
            result["callable_execution"] = "ok"
            result["successful_callable"] = label
            result["depth"] = "D3"
            break

        init = getattr(obj, "__init__", None)
        if init is None or not is_zero_arg_callable(init):
            continue
        instance = obj()
        result["callable_execution"] = "ok"
        result["successful_callable"] = f"{label}()"
        result["depth"] = "D3"

        for workflow_name in workflow_names:
            method = getattr(instance, workflow_name, None)
            if method is None or not callable(method) or not is_zero_arg_callable(method):
                continue
            try:
                method()
                result["workflow_execution"] = "ok"
                result["successful_workflow"] = f"{label}().{workflow_name}()"
                result["depth"] = "D4"
                raise StopIteration
            except StopIteration:
                raise
            except Exception as workflow_exc:
                result["notes"].append(f"{label}().{workflow_name}() failed: {short_error(workflow_exc)}")
        break
    except StopIteration:
        break
    except Exception as exc:
        result["notes"].append(f"{label} failed: {short_error(exc)}")

if result["callable_execution"] == "not_attempted":
    result["callable_execution"] = "no_zero_arg_success"
if result["workflow_execution"] == "not_attempted":
    result["workflow_execution"] = "no_workflow_success"

print(json.dumps(result))
"""


def module_dirs() -> list[Path]:
    return sorted(
        p
        for p in ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith((".", "__"))
    )


def run_probe(module_name: str) -> dict[str, object]:
    env = {
        **os.environ,
        "TRIALITY_MODULE": module_name,
        "TRIALITY_ROOT": str(ROOT),
    }
    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SCRIPT],
        cwd=ROOT.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    stdout = proc.stdout.strip().splitlines()
    payload = stdout[-1] if stdout else "{}"
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {
            "module": module_name,
            "depth": "D0",
            "package_import": f"probe_json_error (rc={proc.returncode})",
            "solver_import": "unknown",
            "callable_execution": "unknown",
            "workflow_execution": "unknown",
            "successful_callable": None,
            "successful_workflow": None,
            "notes": [proc.stderr.strip() or proc.stdout.strip() or "probe produced invalid JSON"],
        }


def build_report(rows: list[dict[str, object]]) -> str:
    depth_counts = Counter(row["depth"] for row in rows)
    failed_imports = [row for row in rows if row["depth"] == "D0"]
    no_callable = [row for row in rows if row["depth"] in {"D1", "D2"}]
    workflow_success = [row for row in rows if row["depth"] == "D4"]

    lines = [
        "# Triality Module Execution Depth Report",
        "",
        "This report was generated by executing probe code against every importable `triality` module in the current working tree. The goal is to measure *execution depth*, not just static code presence.",
        "",
        "## Depth rubric",
        "",
    ]
    for code, desc in DEPTH_LABELS.items():
        lines.append(f"- **{code}** → {desc}")

    lines.extend(
        [
            "",
            "## Method",
            "",
            "For each module, the probe script:",
            "",
            "1. imports the package (`triality.<module>`),",
            "2. imports `solver.py` if present,",
            "3. looks for public zero-argument functions or zero-argument class constructors,",
            "4. executes the first successful callable it can find, and",
            "5. if a zero-arg instance exists, attempts a workflow method such as `solve()`, `run()`, `analyze()`, `compute()`, `simulate()`, or `execute()`.",
            "",
            "This is intentionally conservative: a low score means the module either failed to import or did not expose an easy-to-execute runtime surface under a generic probe, which is useful for spotting incomplete or hard-to-exercise code paths.",
            "",
            "## Portfolio summary",
            "",
        ]
    )

    for code in ["D0", "D1", "D2", "D3", "D4"]:
        lines.append(f"- **{code}**: {depth_counts[code]} modules ({DEPTH_LABELS[code]})")

    lines.extend(
        [
            "",
            "## Headline findings",
            "",
            f"- **Workflow depth (`D4`) reached:** {len(workflow_success)} modules exposed at least one generic workflow-style method that could actually be executed.",
            f"- **Import failures (`D0`):** {len(failed_imports)} modules are immediately broken at package-import time under this environment.",
            f"- **Shallow-but-importable (`D1`/`D2`):** {len(no_callable)} modules import but do not expose an easy generic zero-arg execution path, which usually means custom setup is still required or the module is incomplete at the runtime API level.",
            "",
        ]
    )

    if failed_imports:
        lines.extend(
            [
                "## Broken on import",
                "",
            ]
        )
        for row in failed_imports:
            lines.append(f"- `{row['module']}`: {row['package_import']}.")
        lines.append("")

    lines.extend(
        [
            "## Module-by-module results",
            "",
            "| Module | Depth | Package import | Solver import | Callable execution | Workflow execution | Successful callable | Successful workflow | Notes |",
            "|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        notes = "; ".join(row["notes"][:3]) if row["notes"] else ""
        lines.append(
            f"| `{row['module']}` | {row['depth']} | {row['package_import']} | {row['solver_import']} | {row['callable_execution']} | {row['workflow_execution']} | {row['successful_callable'] or ''} | {row['successful_workflow'] or ''} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Follow-up recommendations",
            "",
            "1. Fix all `D0` modules first; import-time breakage is the clearest sign of incomplete code.",
            "2. For `D1`/`D2` modules, add one small smoke-testable public entry point per module so automated depth testing can exercise real behavior instead of only imports.",
            "3. For higher-value solver modules, add lightweight zero-configuration demos or smoke tests that reach a `solve()` / `run()` path deterministically.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = [run_probe(module_dir.name) for module_dir in module_dirs()]
    report = build_report(rows)
    (ROOT / "MODULE_EXECUTION_DEPTH_REPORT.md").write_text(report + "\n")
    print(f"Wrote execution depth report for {len(rows)} modules")


if __name__ == "__main__":
    main()
