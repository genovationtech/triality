#!/usr/bin/env bash
# ============================================================================
#  Triality — Professional Setup Script
#  Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS
#  Copyright (c) 2024-2026. Licensed under the MIT License.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Terminal Colors & Formatting (ANSI escape codes — no tput dependency)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD="\033[1m"
    DIM="\033[2m"
    RESET="\033[0m"
    RED="\033[31m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    BLUE="\033[34m"
    CYAN="\033[36m"
    WHITE="\033[97m"
    BG_RESET="\033[49m"
else
    BOLD="" DIM="" RESET=""
    RED="" GREEN="" YELLOW="" BLUE="" CYAN="" WHITE="" BG_RESET=""
fi

# ---------------------------------------------------------------------------
#  Animated ASCII Banner — White / Blue / Green tri-color
# ---------------------------------------------------------------------------
show_banner() {
    clear
    echo ""

    # Top border — green
    echo -e "${GREEN}     ╔══════════════════════════════════════════════════════════════════╗${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║                                                                ║${RESET}"
    sleep 0.03

    # TRIALITY letters — top rows white, mid rows blue, bottom rows green
    echo -e "${GREEN}     ║${RESET}   ${BOLD}${WHITE}████████╗██████╗ ██╗ █████╗ ██╗     ██╗████████╗██╗   ██╗${RESET}   ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}   ${BOLD}${WHITE}╚══██╔══╝██╔══██╗██║██╔══██╗██║     ██║╚══██╔══╝╚██╗ ██╔╝${RESET}  ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}      ${BOLD}${BLUE}██║   ██████╔╝██║███████║██║     ██║   ██║    ╚████╔╝${RESET}   ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}      ${BOLD}${BLUE}██║   ██╔══██╗██║██╔══██║██║     ██║   ██║     ╚██╔╝${RESET}    ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}      ${BOLD}${GREEN}██║   ██║  ██║██║██║  ██║███████╗██║   ██║      ██║${RESET}     ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}      ${BOLD}${GREEN}╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝${RESET}     ${GREEN}║${RESET}"
    sleep 0.03

    # Divider
    echo -e "${GREEN}     ║                                                                ║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}   ${DIM}${GREEN}─────────────────────────────────────────────────────────${RESET}    ${GREEN}║${RESET}"
    sleep 0.03

    # Info block
    echo -e "${GREEN}     ║${RESET}   ${BOLD}${WHITE}Production Physics Simulation Framework${RESET}            ${DIM}v0.2.0${RESET}  ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}   ${DIM}${WHITE}Genovation Technological Solutions${RESET} ${DIM}· Powered by Mentis OS${RESET} ${GREEN}║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ║${RESET}   ${DIM}${CYAN}\"We build systems that understand reality.\"${RESET}              ${GREEN}║${RESET}"
    sleep 0.03

    # Divider
    echo -e "${GREEN}     ║${RESET}   ${DIM}${GREEN}─────────────────────────────────────────────────────────${RESET}    ${GREEN}║${RESET}"
    sleep 0.03

    # Feature tags
    echo -e "${GREEN}     ║${RESET}   ${WHITE}PDE Solving${RESET}  ${DIM}│${RESET}  ${BLUE}Physics Routing${RESET}  ${DIM}│${RESET}  ${GREEN}Semiconductor Analysis${RESET}  ${GREEN}║${RESET}"
    sleep 0.03

    # Bottom border
    echo -e "${GREEN}     ║                                                                ║${RESET}"
    sleep 0.03
    echo -e "${GREEN}     ╚══════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

# ---------------------------------------------------------------------------
#  Status Helpers
# ---------------------------------------------------------------------------
step=0
total_steps=7

progress() {
    ((step++)) || true
    echo ""
    echo -e "  ${BOLD}${GREEN}[$step/$total_steps]${RESET} ${BOLD}${WHITE}$1${RESET}"
    echo -e "  ${DIM}$(printf '%.0s─' $(seq 1 62))${RESET}"
}

ok()   { echo -e "       ${GREEN}✓${RESET} $1"; }
warn() { echo -e "       ${YELLOW}!${RESET} $1"; }
fail() { echo -e "       ${RED}✗${RESET} $1"; }
info() { echo -e "       ${DIM}$1${RESET}"; }

# ---------------------------------------------------------------------------
#  Pre-flight Checks
# ---------------------------------------------------------------------------
preflight() {
    progress "Pre-flight checks"

    # Python
    if command -v python3 &>/dev/null; then
        local pyver
        pyver=$(python3 --version 2>&1 | awk '{print $2}')
        local pymajor pyminor
        pymajor=$(echo "$pyver" | cut -d. -f1)
        pyminor=$(echo "$pyver" | cut -d. -f2)
        if [ "$pymajor" -ge 3 ] && [ "$pyminor" -ge 8 ]; then
            ok "Python ${pyver}"
        else
            fail "Python ${pyver} found — requires >= 3.8"
            exit 1
        fi
    else
        fail "Python 3 not found"
        echo ""
        echo "  Install Python 3.8+ from https://python.org"
        exit 1
    fi

    # pip
    if python3 -m pip --version &>/dev/null; then
        ok "pip $(python3 -m pip --version 2>&1 | awk '{print $2}')"
    else
        fail "pip not found"
        exit 1
    fi

    # git (optional)
    if command -v git &>/dev/null; then
        ok "git $(git --version | awk '{print $3}')"
    else
        warn "git not found (optional)"
    fi

    # Rust (optional)
    if command -v rustc &>/dev/null; then
        ok "Rust $(rustc --version | awk '{print $2}') (enables high-performance engine)"
    else
        warn "Rust not found (optional — install for native acceleration)"
        info "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    fi

    # Docker (optional)
    if command -v docker &>/dev/null; then
        ok "Docker $(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')"
    else
        warn "Docker not found (optional — available for containerized deployment)"
    fi
}

# ---------------------------------------------------------------------------
#  Virtual Environment
# ---------------------------------------------------------------------------
setup_venv() {
    progress "Setting up Python virtual environment"

    local VENV_DIR=".venv"
    if [ -d "$VENV_DIR" ]; then
        ok "Virtual environment exists at ${VENV_DIR}/"
    else
        python3 -m venv "$VENV_DIR"
        ok "Created virtual environment at ${VENV_DIR}/"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    ok "Activated virtual environment"
    info "Python: $(which python)"
}

# ---------------------------------------------------------------------------
#  Install Core Dependencies
# ---------------------------------------------------------------------------
install_core() {
    progress "Installing core dependencies"

    pip install --upgrade pip setuptools wheel --quiet 2>&1 | tail -1 || true
    ok "Updated pip, setuptools, wheel"

    if [ -f "lib/setup.py" ] || [ -f "lib/pyproject.toml" ]; then
        (cd lib && pip install -e ".[plot,test]" --quiet 2>&1 | tail -1) || {
            warn "Some optional extras failed — installing base package"
            (cd lib && pip install -e . --quiet 2>&1 | tail -1) || true
        }
        ok "Installed triality package (numpy, scipy, matplotlib, pytest)"
    else
        warn "lib/setup.py not found — skipping triality package install"
    fi

    pip install fastapi uvicorn[standard] pydantic httpx --quiet 2>&1 | tail -1 || true
    ok "Installed web framework (FastAPI, Uvicorn, Pydantic)"
}

# ---------------------------------------------------------------------------
#  Build Rust Engine (optional)
# ---------------------------------------------------------------------------
build_rust() {
    progress "Building Rust acceleration engine"

    if ! command -v rustc &>/dev/null; then
        warn "Skipped — Rust toolchain not installed"
        info "The framework runs fully on Python; Rust provides optional acceleration"
        return 0
    fi

    if ! command -v maturin &>/dev/null; then
        pip install maturin --quiet 2>&1 | tail -1 || true
    fi

    if [ -d "lib/triality/triality_engine" ]; then
        (cd lib/triality/triality_engine && maturin develop --release 2>/dev/null) && {
            ok "Built triality_engine (Rust + PyO3)"
        } || {
            warn "Rust build failed — continuing with Python-only mode"
        }
    else
        warn "Rust engine directory not found — skipping"
    fi
}

# ---------------------------------------------------------------------------
#  Validate Installation
# ---------------------------------------------------------------------------
validate() {
    progress "Validating installation"

    # Test core import
    if python3 -c "from triality import solve, Field, laplacian, Interval; print('Core OK')" 2>/dev/null; then
        ok "Core framework imports"
    else
        warn "Core framework import incomplete — some modules may need configuration"
    fi

    # Test solver
    if python3 -c "
from triality import *
u = Field('u')
sol = solve(laplacian(u) == 1, Interval(0, 1), bc={'left': 0, 'right': 0})
assert sol.grid.shape[0] > 0
print('Solver OK')
" 2>/dev/null; then
        ok "PDE solver operational (Layer 1)"
    else
        warn "PDE solver test did not pass cleanly"
    fi

    # Test FastAPI import
    if python3 -c "from triality_app.main import app; print('App OK')" 2>/dev/null; then
        ok "FastAPI application loads"
    else
        warn "FastAPI app import issue — check triality_app/"
    fi

    # Test Rust engine
    if python3 -c "import triality_engine; print('Rust OK')" 2>/dev/null; then
        ok "Rust acceleration engine loaded"
    else
        info "Rust engine not available (Python-only mode)"
    fi
}

# ---------------------------------------------------------------------------
#  Run Tests
# ---------------------------------------------------------------------------
run_tests() {
    progress "Running test suite"

    if command -v pytest &>/dev/null || python3 -m pytest --version &>/dev/null 2>&1; then
        local test_output
        test_output=$(cd lib/triality && python3 -m pytest -x -q --tb=no 2>&1) || true
        local passed failed
        passed=$(echo "$test_output" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
        failed=$(echo "$test_output" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")

        if [ "$failed" = "0" ] || [ -z "$failed" ]; then
            ok "${passed:-0} tests passed"
        else
            warn "${passed:-0} passed, ${failed} failed"
            info "Run 'make test' for details"
        fi
    else
        warn "pytest not available — skipping tests"
    fi
}

# ---------------------------------------------------------------------------
#  Final Summary
# ---------------------------------------------------------------------------
summary() {
    progress "Setup complete"

    echo ""
    echo -e "  ${BOLD}${GREEN}Triality is ready.${RESET}"
    echo ""
    echo -e "  ${BOLD}${WHITE}Quick Start:${RESET}"
    echo ""
    echo -e "    ${BLUE}# Create virtual environment (if not already created)${RESET}"
    echo -e "    ${WHITE}python3 -m venv .venv${RESET}"
    echo ""
    echo -e "    ${BLUE}# Activate environment${RESET}"
    echo -e "    ${WHITE}source .venv/bin/activate${RESET}"
    echo ""
    echo -e "    ${BLUE}# Start the application${RESET}"
    echo -e "    ${WHITE}make run${RESET}"
    echo ""
    echo -e "    ${BLUE}# Open in browser${RESET}"
    echo -e "    ${WHITE}http://<your-server-ip>:8510${RESET}"
    echo ""
    echo -e "    ${BLUE}# Or use Docker${RESET}"
    echo -e "    ${WHITE}docker compose up${RESET}"
    echo ""
    echo -e "    ${BLUE}# Run a quick simulation${RESET}"
    echo -e "    ${WHITE}python3 -c \"${RESET}"
    echo -e "    ${WHITE}from triality import *${RESET}"
    echo -e "    ${WHITE}u = Field('u')${RESET}"
    echo -e "    ${WHITE}sol = solve(laplacian(u) == 1, Interval(0, 1), bc={'left': 0, 'right': 0})${RESET}"
    echo -e "    ${WHITE}print(f'Solved: {sol.grid.shape[0]} points, max = {sol.values.max():.4f}')${RESET}"
    echo -e "    ${WHITE}\"${RESET}"
    echo ""
    echo -e "    ${BLUE}# Run tests${RESET}"
    echo -e "    ${WHITE}make test${RESET}"
    echo ""
    echo -e "  ${GREEN}─────────────────────────────────────────────────────────────────${RESET}"
    echo -e "  ${DIM}Docs:   docs/               │  Architecture:  docs/architecture.md${RESET}"
    echo -e "  ${DIM}Tests:  make test            │  API Reference: docs/api_reference.md${RESET}"
    echo -e "  ${GREEN}─────────────────────────────────────────────────────────────────${RESET}"
    echo ""
}

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
main() {
    show_banner
    preflight
    setup_venv
    install_core
    build_rust
    validate
    run_tests
    summary
}

main "$@"
