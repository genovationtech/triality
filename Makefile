# ============================================================================
#  Triality — Makefile
#  Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS
# ============================================================================

.DEFAULT_GOAL := help
.PHONY: help setup install dev test lint clean docker docker-dev run

VENV     := .venv
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
PORT     := 8510
HOST     := 0.0.0.0

# ---------------------------------------------------------------------------
#  Help
# ---------------------------------------------------------------------------
help: ## Show this help
	@echo ""
	@echo "  Triality — Available Commands"
	@echo "  ─────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
#  Setup & Install
# ---------------------------------------------------------------------------
setup: ## Run full automated setup
	@./setup.sh

install: $(VENV) ## Install dependencies into virtual environment
	cd lib && $(PIP) install -e ".[plot,test]"
	$(PIP) install fastapi "uvicorn[standard]" pydantic httpx

$(VENV): ## Create virtual environment
	python3 -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

# ---------------------------------------------------------------------------
#  Development
# ---------------------------------------------------------------------------
run: ## Start the application server
	cd triality_app && $(CURDIR)/$(VENV)/bin/uvicorn main:app --host $(HOST) --port $(PORT) --reload

test: ## Run all tests
	cd lib/triality && $(PYTHON) -m pytest -v

test-layer1: ## Run Layer 1 tests (electrostatics)
	cd lib/triality && $(PYTHON) -m pytest test_electrostatics.py -v

test-layer2: ## Run Layer 2 tests (field-aware routing)
	cd lib/triality && $(PYTHON) -m pytest test_field_aware_routing.py -v

test-layer3: ## Run Layer 3 tests (drift-diffusion)
	cd lib/triality && $(PYTHON) -m pytest test_drift_diffusion.py -v

test-coverage: ## Run tests with coverage report
	cd lib/triality && $(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term

lint: ## Run linter
	$(PIP) install ruff --quiet
	ruff check lib/triality/ triality_app/

verify: ## Quick verification — solve a PDE
	@$(PYTHON) -c "\
	from triality import *; \
	u = Field('u'); \
	sol = solve(laplacian(u) == 1, Interval(0,1), bc={'left':0,'right':0}); \
	print(f'Solved: {sol.grid.shape[0]} pts, max={sol.values.max():.4f}'); \
	print('All systems nominal.')"

# ---------------------------------------------------------------------------
#  Rust Engine
# ---------------------------------------------------------------------------
rust-build: ## Build Rust acceleration engine
	cd lib/triality/triality_engine && maturin develop --release

rust-clean: ## Clean Rust build artifacts
	cd lib/triality/triality_engine && cargo clean

# ---------------------------------------------------------------------------
#  Docker
# ---------------------------------------------------------------------------
docker: ## Build and run with Docker
	docker compose up --build

docker-dev: ## Build and run in dev mode (hot-reload)
	docker compose --profile dev up --build triality-dev

docker-build: ## Build Docker image only
	docker build -t triality .

# ---------------------------------------------------------------------------
#  Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage
	@echo "Cleaned."
