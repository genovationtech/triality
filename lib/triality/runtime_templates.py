from pathlib import Path
from typing import List

from triality.runtime_graph import RuntimeGraph

_TEMPLATE_DIR = Path(__file__).with_name("runtime_templates")


def available_runtime_templates() -> List[str]:
    return sorted(path.stem for path in _TEMPLATE_DIR.glob("*.json"))


def load_runtime_template(name: str) -> RuntimeGraph:
    path = _TEMPLATE_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(available_runtime_templates())
        raise KeyError(f"Unknown runtime template '{name}'. Available: {available}")
    return RuntimeGraph.load_json(path)
