"""
Multi-Step Task Decomposition for Triality
===========================================
Breaks complex analysis prompts into discrete, ordered tasks.
Each task gets its own focused LLM call for configuration, then
execution, then the results feed into the next task's context.

Pipeline:
    1. Decompose: LLM breaks prompt into ordered analysis tasks
    2. For each task:
       a. Configure: Focused LLM call with prior results as context
       b. Execute: Run the tool
       c. Summarize: Brief result summary for next task's context
    3. Consolidate: Final LLM call synthesizes all results into the answer
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("triality.task_decomposer")


# ---------------------------------------------------------------------------
#  Decomposition Prompt
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = """Break the user's physics analysis request into ordered tasks.

User's request: "{prompt}"

ALLOWED TOOLS (use ONLY these exact names):
- run_module — run a single simulation
- sweep_parameter — sweep one parameter over a range
- optimize_parameter — find optimal value of a parameter
- compare_scenarios — compare multiple configurations

DO NOT use any other tool names. "visualize", "plot", "show fields" are NOT tools — they happen automatically when you use run_module.

Rules:
1. Each task = ONE tool from the list above
2. "Show fields" or "visualize" = just run_module (fields are shown automatically)
3. Do ONLY what the user explicitly asked for. Do NOT add extra sweeps, optimizations, or comparisons unless the user requested them.
4. If the user says "check", "run", or "analyze" with parameters but no sweep/optimize language, use ONLY run_module with those parameters.
5. If conversation history is provided, use it to understand the context (e.g. what module, what type of analysis).
6. Most simple requests need just 1 task. Only add more if the user explicitly asked for multiple things.

Available physics modules: {modules}

Respond with ONLY valid JSON:
{{
    "tasks": [
        {{"step": 1, "tool": "run_module", "description": "Run baseline with given parameters", "purpose": "Establish baseline"}}
    ],
    "final_question": "What is the user's actual question?"
}}
"""


CONFIGURE_TASK_PROMPT = """Configure the arguments for a {tool_name} call.

Task: {task_description}
Purpose: {task_purpose}
User's request: {prompt}

{module_info}

Previous results:
{prior_results}

ARGUMENT FORMATS (follow exactly):

For run_module:
{{"module_name": "navier_stokes", "config": {{"solver": {{...}}, "solve": {{...}}}}}}

For sweep_parameter:
{{"module_name": "navier_stokes", "config": {{"solver": {{...}}, "solve": {{...}}}}, "param_path": "solver.nu", "start": 0.005, "end": 0.05, "steps": 5}}

For optimize_parameter:
{{"module_name": "navier_stokes", "config": {{"solver": {{...}}, "solve": {{...}}}}, "param_path": "solver.nu", "bounds": [0.005, 0.05], "objective_field": "velocity", "objective": "maximize", "n_evals": 10}}

For compare_scenarios:
{{"module_name": "navier_stokes", "scenarios": [{{"label": "case1", "config": {{...}}}}, {{"label": "case2", "config": {{...}}}}]}}

RULES:
- config MUST include solver and solve sections with all required parameters
- All physical values must be positive (viscosity, density, velocity, temperature)
- For sweep/optimize: param_path uses dot notation like "solver.nu"
- Copy the config from previous results if available

Return ONLY the JSON arguments:"""


CONSOLIDATE_PROMPT = """You are a senior lead engineer. The user asked a question and multiple analysis steps were executed to answer it.

## User's Question
{prompt}

## Analysis Steps and Results
{all_task_results}

## Programmatic Insights
{insights}

{industry_context}

## Your Job
Synthesize ALL results into a clear, complete answer. Structure:

1. **Answer** — Lead with the direct answer to the user's question. Be specific with numbers.
2. **Key Findings** — What each analysis step revealed (cite numbers from results only)
3. **Physics Interpretation** — Why the results look this way
4. **Limitations** — Grid resolution, 2D assumption, model limits
5. **Recommendations** — Specific next steps if needed

STRICT: Only cite numbers from the results above. Never invent values.
Do NOT say "further study needed" if the analysis already answered the question.
FORMAT: Use LaTeX math notation for all units, exponents, and expressions — e.g. $10^{{17}}$ cm$^{{-3}}$, $1.57 \\times 10^{{5}}$ A/cm$^{{2}}$, $V_{{bi}} = 0.79$ V. This is required for proper rendering.
"""


# ---------------------------------------------------------------------------
#  Task Decomposer
# ---------------------------------------------------------------------------
def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM output, handling markdown code blocks."""
    import re as _re
    if not text:
        return None
    json_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    brace_match = _re.search(r"\{.*\}", text, _re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def decompose_prompt(
    prompt: str,
    modules: List[str],
    call_llm: Callable,
    token: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Break a user prompt into ordered analysis tasks via LLM."""
    decompose = DECOMPOSE_PROMPT.format(
        prompt=prompt,
        modules=json.dumps(modules),
    )
    output = call_llm(
        "You are a physics analysis planner. Break complex requests into simple, ordered steps. "
        "Each step is ONE tool call. Keep parameters physically reasonable.",
        decompose,
        token=token,
        model=model,
    )
    if not output:
        return None

    result = _parse_json(output)
    if result and result.get("tasks"):
        return result
    return None


def configure_task(
    task: Dict[str, Any],
    prompt: str,
    module_info: str,
    prior_results: str,
    call_llm: Callable,
    token: str,
    model: str,
    parse_json: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Configure a single task's tool arguments via focused LLM call."""
    config_prompt = CONFIGURE_TASK_PROMPT.format(
        task_description=task.get("description", ""),
        tool_name=task.get("tool", "run_module"),
        task_purpose=task.get("purpose", ""),
        prompt=prompt,
        module_info=module_info,
        prior_results=prior_results if prior_results else "No previous results — this is the first analysis step.",
    )

    output = call_llm(
        "You are a physics tool configurator. Return ONLY the JSON arguments for the specified tool. "
        "All parameters must be physically reasonable. Never use negative values for viscosity, "
        "force, temperature, or other physical quantities that must be positive.",
        config_prompt,
        token=token,
        model=model,
    )
    if not output:
        return None

    _parser = parse_json or _parse_json
    args = _parser(output)
    if args:
        # Safety: sanitize obviously wrong parameters
        args = _sanitize_args(task.get("tool", ""), args)
    return args


def build_task_result_summary(
    task: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    """Build a concise summary of a task's result for the next task's context."""
    lines = []
    tool = task.get("tool", "")
    lines.append(f"### Step {task.get('step', '?')}: {task.get('description', '')}")

    if result.get("error"):
        lines.append(f"  ERROR: {result['error']}")
        return "\n".join(lines)

    if tool == "run_module":
        # Summarize key fields and observables
        fields_stats = result.get("fields_stats", {})
        for fname, stats in list(fields_stats.items())[:5]:
            lines.append(f"  {fname}: min={stats.get('min', 0):.4g}, max={stats.get('max', 0):.4g}, mean={stats.get('mean', 0):.4g}")

        observables = result.get("observables", [])
        if isinstance(observables, list):
            for obs in observables[:8]:
                v = obs.get("value")
                if isinstance(v, (int, float)):
                    lines.append(f"  {obs.get('name', '?')}: {v:.4g} {obs.get('unit', '')}")
                elif isinstance(v, bool):
                    lines.append(f"  {obs.get('name', '?')}: {'YES' if v else 'NO'}")

    elif tool == "sweep_parameter":
        sweep_results = result.get("results", [])
        param_path = result.get("param_path", "")
        lines.append(f"  Swept {param_path} over {len(sweep_results)} points")
        for sr in sweep_results:
            if sr.get("success"):
                pv = sr.get("param_value", "?")
                obs_summary = []
                for oname, odata in sr.get("observables", {}).items():
                    if isinstance(odata, dict):
                        v = odata.get("value")
                        if isinstance(v, (int, float)):
                            obs_summary.append(f"{oname}={v:.4g}")
                if obs_summary:
                    lines.append(f"  {param_path}={pv:.4g}: {', '.join(obs_summary[:4])}")

    elif tool == "optimize_parameter":
        lines.append(f"  Optimal {result.get('param_path', '?')} = {result.get('optimal_param_value', '?')}")
        lines.append(f"  Metric ({result.get('objective_field', '?')}): {result.get('optimal_metric', '?')}")

    elif tool == "compare_scenarios":
        comp = result.get("comparison", [])
        for c in comp:
            lines.append(f"  {c.get('label', '?')}: success={c.get('success', False)}")

    return "\n".join(lines)


def _normalize_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM-generated args to match the actual tool signatures.

    Fixes common LLM mistakes:
    - Strips 'tool' key from inside args
    - Renames 'module' → 'module_name'
    - Unwraps nested 'args' dict
    - Fixes 'num' → 'steps', 'run_config' → 'config', etc.
    """
    # If LLM wrapped the whole thing in {"tool": ..., "args": {...}}, unwrap
    if "tool" in args and "args" in args:
        inner = args["args"]
        if isinstance(inner, dict):
            args = inner
    elif "tool" in args:
        # Remove stray 'tool' key
        args = {k: v for k, v in args.items() if k != "tool"}

    # Rename common LLM mistakes
    renames = {
        "module": "module_name",
        "num": "steps",
        "num_steps": "steps",
        "n_steps": "steps",
        "run_config": "config",
        "base_config": "config",
        "parameter_path": "param_path",
        "parameter": "param_path",
        "n_samples": "n_samples",
    }
    for wrong, right in renames.items():
        if wrong in args and right not in args:
            args[right] = args.pop(wrong)

    # For sweep: ensure 'values' or 'start'+'end' exist
    if tool_name == "sweep_parameter":
        # Remove non-standard keys the LLM might add
        valid_keys = {"module_name", "config", "param_path", "values", "start", "end", "steps"}
        args = {k: v for k, v in args.items() if k in valid_keys}

    # For run_module: ensure 'module_name' and 'config' exist
    if tool_name == "run_module":
        valid_keys = {"module_name", "config"}
        args = {k: v for k, v in args.items() if k in valid_keys}

    # For optimize: normalize
    if tool_name == "optimize_parameter":
        valid_keys = {"module_name", "config", "param_path", "bounds", "objective_field", "objective", "n_evals"}
        args = {k: v for k, v in args.items() if k in valid_keys}

    # For compare_scenarios: normalize
    if tool_name == "compare_scenarios":
        valid_keys = {"module_name", "scenarios"}
        args = {k: v for k, v in args.items() if k in valid_keys}

    return args


def _sanitize_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Fix obviously wrong parameters that the LLM might generate."""
    # First normalize the structure
    args = _normalize_args(tool_name, args)
    if tool_name == "sweep_parameter":
        # Ensure start < end and both positive for physical parameters
        start = args.get("start")
        end = args.get("end")
        if start is not None and end is not None:
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                # Physical parameters should be positive
                param_path = args.get("param_path", "")
                physical_params = ["nu", "viscosity", "temperature", "force", "pressure",
                                   "rho", "density", "velocity", "U_lid", "power", "current"]
                if any(p in param_path.lower() for p in physical_params):
                    start = abs(start) if start != 0 else 0.001
                    end = abs(end) if end != 0 else 1.0
                    args["start"] = start
                    args["end"] = end
                # Ensure start < end
                if start > end:
                    args["start"], args["end"] = end, start
                # Ensure they're not equal
                if args["start"] == args["end"]:
                    args["end"] = args["start"] * 2 if args["start"] > 0 else 1.0

        # Ensure steps is reasonable
        steps = args.get("steps")
        if steps is not None and isinstance(steps, (int, float)):
            args["steps"] = max(3, min(int(steps), 50))

    elif tool_name == "optimize_parameter":
        bounds = args.get("bounds", [])
        if isinstance(bounds, list) and len(bounds) == 2:
            param_path = args.get("param_path", "")
            physical_params = ["nu", "viscosity", "temperature", "force", "pressure",
                               "rho", "density", "velocity", "U_lid"]
            if any(p in param_path.lower() for p in physical_params):
                bounds = [abs(b) if isinstance(b, (int, float)) else b for b in bounds]
                if bounds[0] == 0:
                    bounds[0] = 0.001
                if bounds[1] == 0:
                    bounds[1] = 1.0
                args["bounds"] = sorted(bounds)

        n_evals = args.get("n_evals")
        if n_evals is not None:
            args["n_evals"] = max(5, min(int(n_evals), 20))

    elif tool_name == "run_module":
        config = args.get("config", {})
        solver = config.get("solver", {})
        # Ensure positive viscosity
        if "nu" in solver and isinstance(solver["nu"], (int, float)) and solver["nu"] <= 0:
            solver["nu"] = abs(solver["nu"]) if solver["nu"] != 0 else 0.01
        # Ensure positive density
        if "rho" in solver and isinstance(solver["rho"], (int, float)) and solver["rho"] <= 0:
            solver["rho"] = abs(solver["rho"]) if solver["rho"] != 0 else 1.0

    return args


def build_module_info_for_task(tool_name: str, modules_info: Dict, available_modules: List[str]) -> str:
    """Build focused module info string for a specific task."""
    lines = []
    for m in available_modules[:15]:  # Limit to avoid token overflow
        mi = modules_info.get(m, {})
        lines.append(f"**{m}** ({mi.get('domain', 'Physics')}): {mi.get('description', 'N/A')}")
        if mi.get("config_keys"):
            lines.append(f"  Config: {json.dumps(mi['config_keys'])}")
        if mi.get("defaults"):
            lines.append(f"  Defaults: {json.dumps(mi['defaults'])}")
    return "\n".join(lines)
