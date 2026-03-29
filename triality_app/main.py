"""
Triality Physics Reasoning Engine — FastAPI Backend
=============================================
Full agentic loop: Goal Extract → Analytical Estimate → Plan → Execute → Evaluate → Adapt → Converge → Answer
with LLM integration (Replicate), SSE streaming, 8 tool types,
goal-driven convergence engine, heuristic fallback planner, and programmatic insights.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple

import numpy as np
import requests as http_requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from triality import (
    RuntimeExecutionResult,
    available_runtime_modules,
    available_runtime_templates,
    load_module,
    load_runtime_template,
)
from triality.runtime_graph import RuntimeGraphResult, RuntimeNodeResult

try:
    from triality_app.goal_engine import (
        AnalysisGoal,
        AnalyticalEstimate,
        ConvergenceAction,
        GoalDrivenRunner,
        GoalEvaluation,
        GoalEvaluator,
        GoalType,
        GOAL_EXTRACTION_PROMPT,
        ANALYTICAL_ESTIMATION_PROMPT,
        extract_goal_from_llm_response,
        extract_goal_heuristic,
        parse_analytical_estimate,
        infer_goal_from_plan,
        detect_unresolved_findings,
        build_goal_from_finding,
        extract_goal_from_scenario,
        ConvergenceStrategy,
        SearchState,
        check_convergence,
        expand_bounds,
        compute_confidence,
        commit_answer_optimum,
        commit_answer_threshold,
    )
except ImportError:
    from goal_engine import (  # type: ignore[no-redef]
        AnalysisGoal,
        AnalyticalEstimate,
        ConvergenceAction,
        GoalDrivenRunner,
        GoalEvaluation,
        GoalEvaluator,
        GoalType,
        GOAL_EXTRACTION_PROMPT,
        ANALYTICAL_ESTIMATION_PROMPT,
        extract_goal_from_llm_response,
        extract_goal_heuristic,
        parse_analytical_estimate,
        infer_goal_from_plan,
        detect_unresolved_findings,
        build_goal_from_finding,
        extract_goal_from_scenario,
        ConvergenceStrategy,
        SearchState,
        check_convergence,
        expand_bounds,
        compute_confidence,
        commit_answer_optimum,
        commit_answer_threshold,
    )

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
REACT_DIST_DIR = BASE_DIR / "static" / "dist"
STATIC_DIR = REACT_DIST_DIR if REACT_DIST_DIR.exists() else BASE_DIR / "static"
DEFAULT_MODEL = "meta/llama-4-maverick-instruct"
DEFAULT_REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
MAX_REFLECTION_STEPS = 2
logger = logging.getLogger("triality")

# ---------------------------------------------------------------------------
#  Module Info Catalog — descriptions, defaults, config keys
# ---------------------------------------------------------------------------
TRIALITY_MODULES_INFO: Dict[str, Dict[str, Any]] = {
    "navier_stokes": {
        "description": "2D/3D laminar Navier-Stokes solver (projection method). Lid-driven cavity, channel flow, backward-facing step.",
        "domain": "Fluid Dynamics",
        "config_keys": {
            "solver": ["nx", "ny", "Lx", "Ly", "rho", "nu", "U_lid", "cfl", "quasi_3d", "z_length"],
            "solve": ["t_end", "dt", "max_steps", "pressure_iters", "pressure_tol"],
        },
        "defaults": {
            "solver": {"nx": 64, "ny": 64, "Lx": 1.0, "Ly": 1.0, "rho": 1.0, "nu": 0.01, "U_lid": 1.0},
            "solve": {"t_end": 0.5, "dt": 0.001, "max_steps": 500, "pressure_iters": 100, "pressure_tol": 1e-5},
        },
    },
    "drift_diffusion": {
        "description": "Semiconductor PN junction solver. Poisson + continuity + drift-diffusion transport with SRH recombination.",
        "domain": "Semiconductor Physics",
        "config_keys": {
            "solver": ["length", "n_points", "material", "temperature", "enable_srh", "enable_field_mobility", "tau_n", "tau_p"],
            "solve": ["applied_voltage", "max_iterations", "tolerance", "under_relaxation"],
            "doping": ["type", "N_d_level", "N_a_level", "junction_pos"],
        },
        "defaults": {
            "solver": {"n_points": 100, "temperature": 300.0},
            "solve": {"applied_voltage": 0.7},
            "doping": {"N_d_level": 1e17, "N_a_level": 5e16},
        },
    },
    "sensing": {
        "description": "Radar detection probability mapping. Radar range equation + SNR estimation over spatial grid.",
        "domain": "Sensing & Detection",
        "config_keys": {
            "radar": ["frequency_ghz", "power_w", "aperture_diameter_m", "bandwidth_mhz", "pulse_width_us", "prf_hz", "noise_figure_db", "n_pulses_integrated", "losses_db"],
            "target": ["rcs_m2", "target_strength_db"],
            "grid": ["grid_x_km", "grid_y_km", "grid_nx", "grid_ny", "auto_grid", "pfa", "weather", "water_temp_c"],
        },
        "defaults": {
            "radar": {"frequency_ghz": 10.0, "power_w": 1000, "aperture_diameter_m": 1.0, "bandwidth_mhz": 10.0, "noise_figure_db": 3.0, "n_pulses_integrated": 10, "losses_db": 3.0},
            "target": {"rcs_m2": 1.0},
            "auto_grid": True,
        },
    },
    "electrostatics": {
        "description": "2D Laplace/Poisson electrostatic and conduction solver. Electric field, current density, Joule heating.",
        "domain": "Electromagnetism",
        "config_keys": {
            "solver": ["x_min", "x_max", "y_min", "y_max", "resolution", "permittivity", "mode"],
            "solve": ["method", "boundary_value"],
        },
        "defaults": {
            "solver": {"x_min": 0.0, "x_max": 0.1, "y_min": 0.0, "y_max": 0.1, "resolution": 50, "mode": "electrostatic"},
            "solve": {"method": "gmres", "boundary_value": 1000.0},
        },
    },
    "aero_loads": {
        "description": "Hypersonic/supersonic aerodynamic loads and heating. Newtonian impact + flat-plate heating + force integration.",
        "domain": "Aerodynamics",
        "config_keys": {
            "solver": ["body_length_m", "nose_radius_m", "base_radius_m", "n_panels"],
            "solve": ["velocity", "density", "temperature", "pressure", "alpha_deg", "wall_temperature_K"],
        },
        "defaults": {
            "solver": {"body_length_m": 3.0, "nose_radius_m": 0.15, "base_radius_m": 0.15, "n_panels": 100},
            "solve": {"velocity": 2000.0, "density": 0.1, "temperature": 300.0, "pressure": 1e4, "alpha_deg": 5.0, "wall_temperature_K": 300.0},
        },
    },
    "uav_aerodynamics": {
        "description": "Vortex Lattice Method for finite wing aerodynamics. Lift, induced drag, spanwise circulation distribution.",
        "domain": "Aerodynamics",
        "config_keys": {
            "solver": ["span", "root_chord", "tip_chord", "n_span", "alpha_deg", "V_inf", "rho"],
        },
        "defaults": {
            "solver": {"span": 10.0, "root_chord": 1.0, "n_span": 40, "alpha_deg": 5.0, "V_inf": 30.0, "rho": 1.225},
        },
    },
    "spacecraft_thermal": {
        "description": "Multi-node transient spacecraft thermal analysis. Radiative exchange, conduction, heaters, heat pipes.",
        "domain": "Thermal",
        "config_keys": {
            "solver": ["n_nodes", "T_space", "T_min_limit", "T_max_limit", "internal_power_W"],
            "solve": ["t_end", "dt"],
        },
        "defaults": {
            "solver": {"n_nodes": 4, "T_space": 4.0, "T_min_limit": 233.0, "T_max_limit": 333.0, "internal_power_W": 50.0},
            "solve": {"t_end": 3600.0, "dt": 10.0},
        },
    },
    "automotive_thermal": {
        "description": "Transient thermal analysis for automotive power electronics. Multi-component lumped thermal network.",
        "domain": "Thermal",
        "config_keys": {
            "solver": ["n_components", "T_ambient", "current_A", "h_convection"],
            "solve": ["t_end", "dt"],
        },
        "defaults": {
            "solver": {"n_components": 3, "T_ambient": 298.0, "current_A": 200.0, "h_convection": 100.0},
            "solve": {"t_end": 60.0, "dt": 0.1},
        },
    },
    "battery_thermal": {
        "description": "EV battery pack thermal management. Drive-cycle analysis, cooling sizing, thermal runaway risk.",
        "domain": "Thermal",
        "config_keys": {
            "solver": ["n_cells", "cooling_type", "cell_chemistry", "T_init"],
            "solve": ["discharge_current_A", "duration_s", "dt"],
        },
        "defaults": {
            "solver": {"n_cells": 96, "cooling_type": "liquid_indirect", "cell_chemistry": "NMC", "T_init": 298.0},
            "solve": {"discharge_current_A": 50.0, "duration_s": 600.0, "dt": 0.5},
        },
    },
    "structural_analysis": {
        "description": "FEM beam static analysis with buckling checks and composite laminate evaluation.",
        "domain": "Structures",
        "config_keys": {
            "solver": ["material_name", "length", "n_elements", "I", "A", "c", "support"],
            "solve": ["tip_force_N", "distributed_load_N_m"],
        },
        "defaults": {
            "solver": {"material_name": "AL7075-T6", "length": 1.0, "n_elements": 20, "I": 1e-6, "A": 1e-4, "c": 0.01, "support": "cantilever"},
            "solve": {"tip_force_N": -5000.0},
        },
    },
    "structural_dynamics": {
        "description": "Newmark-beta time integration for structural dynamics. Modal analysis, SRS, random vibration.",
        "domain": "Structures",
        "config_keys": {
            "solver": ["n_dof", "stiffness_diag", "damping_ratio", "force_amplitude", "force_frequency"],
            "solve": ["t_end", "dt", "compute_srs"],
        },
        "defaults": {
            "solver": {"n_dof": 3, "damping_ratio": 0.02, "force_amplitude": 100.0, "force_frequency": 10.0},
            "solve": {"t_end": 2.0, "dt": 0.001},
        },
    },
    "flight_mechanics": {
        "description": "6-DOF rigid body flight mechanics. Attitude control, sensors, actuators, orbital/suborbital.",
        "domain": "Dynamics",
        "config_keys": {
            "solver": ["mass", "Ixx", "Iyy", "Izz", "dt", "gravity"],
            "solve": ["t_final", "omega_x", "omega_y", "omega_z"],
        },
        "defaults": {
            "solver": {"mass": 500.0, "Ixx": 100.0, "Iyy": 120.0, "Izz": 80.0, "dt": 0.01, "gravity": True},
            "solve": {"t_final": 60.0, "omega_x": 0.01, "omega_y": -0.02, "omega_z": 0.005},
        },
    },
    "coupled_physics": {
        "description": "Coupled neutronics-thermal transient. Point kinetics + heat conduction with Doppler/moderator feedback.",
        "domain": "Nuclear",
        "config_keys": {
            "solver": ["n_points", "length_cm", "beta_eff", "neutron_lifetime", "lambda_precursor", "feedback_mode"],
            "solve": ["t_end", "dt", "initial_power", "reactivity_insertion_pcm"],
        },
        "defaults": {
            "solver": {"n_points": 50, "length_cm": 200.0, "beta_eff": 0.0065, "feedback_mode": "full"},
            "solve": {"t_end": 10.0, "dt": 0.01, "initial_power": 1e6, "reactivity_insertion_pcm": 100.0},
        },
    },
    "geospatial": {
        "description": "Facility location optimisation on geospatial networks. Maximal coverage with travel-time constraints.",
        "domain": "Logistics & Infrastructure",
        "config_keys": {
            "solver": ["max_facilities", "time_limit_hours", "target_coverage", "road_type", "swap_iterations"],
        },
        "defaults": {
            "solver": {"max_facilities": 3, "time_limit_hours": 24.0, "target_coverage": 0.95, "road_type": "state_highway", "swap_iterations": 50},
        },
    },
    "field_aware_routing": {
        "description": "EM-aware PCB/cable routing. Solves 2D Laplace/Poisson for electric field cost maps.",
        "domain": "Electromagnetism",
        "config_keys": {
            "solver": ["nx", "ny", "x_max", "y_max", "bc_left_V", "bc_right_V", "mode"],
        },
        "defaults": {
            "solver": {"nx": 64, "ny": 64, "x_max": 1.0, "y_max": 1.0, "bc_left_V": 1000.0, "bc_right_V": 0.0, "mode": "electrostatic"},
        },
    },
    "neutronics": {
        "description": "Two-group neutron diffusion for reactor core analysis. k-eff eigenvalue, flux distribution, power peaking, transients.",
        "domain": "Nuclear",
        "config_keys": {
            "solver": ["core_length", "n_spatial", "fuel_material", "reflector_material", "reflector_thickness", "fuel_type", "generation_time", "initial_power"],
            "solve": ["t_final", "dt", "reactivity_step_pcm"],
        },
        "defaults": {
            "solver": {"core_length": 200.0, "n_spatial": 50, "fuel_material": "FUEL_UO2_3PCT", "reflector_material": "REFLECTOR_H2O", "reflector_thickness": 30.0, "initial_power": 3e9},
            "solve": {"t_final": 0.0, "dt": 0.01, "reactivity_step_pcm": 0.0},
        },
    },
}

# ---------------------------------------------------------------------------
#  Reference Benchmarks for Validation Scoring
# ---------------------------------------------------------------------------
REFERENCE_BENCHMARKS: Dict[str, Dict[str, tuple]] = {
    "navier_stokes": {
        "velocity_x_max": (0.0, 0.50, "Re~100 lid cavity peak u near lid"),
        "velocity_x_min": (-0.38, -0.10, "Re~100 lid cavity reverse flow (Ghia 1982)"),
        "pressure_range": (None, None, "pressure drop depends on Re and geometry"),
    },
    "drift_diffusion": {
        "built_in_potential_v": (0.55, 0.85, "Si PN junction built-in potential at 300K"),
        "ideality_factor": (1.0, 2.0, "diode ideality (1=ideal Shockley)"),
    },
    "sensing": {
        "max_detection_range_km": (None, None, "depends on radar/target config"),
    },
    "electrostatics": {
        "max_field_V_m": (None, None, "depends on geometry and boundary voltage"),
    },
    "aero_loads": {
        "CL": (-0.5, 2.0, "lift coefficient range for typical bodies at angle of attack"),
        "CD": (0.01, 2.0, "drag coefficient range for blunt/streamlined bodies"),
        "peak_heat_flux_W_m2": (None, None, "depends on Mach number and geometry"),
    },
    "uav_aerodynamics": {
        "CL": (0.0, 1.8, "lift coefficient range for subsonic wings"),
        "CD_induced": (0.001, 0.1, "induced drag for finite wings"),
    },
    "spacecraft_thermal": {
        "max_temperature_K": (200, 400, "typical spacecraft operational range"),
        "min_temperature_K": (100, 300, "cold-side temperature range"),
    },
    "automotive_thermal": {
        "max_temperature_K": (298, 500, "power electronics operational range"),
    },
    "battery_thermal": {
        "max_cell_temperature_K": (298, 430, "cell temperature range before runaway"),
        "safety_score": (0.0, 1.0, "0=runaway, 1=fully safe"),
    },
    "structural_analysis": {
        "max_von_mises_Pa": (None, None, "depends on loading and material"),
        "min_buckling_ms": (0.0, None, "positive = safe against buckling"),
    },
    "structural_dynamics": {
        "peak_displacement_m": (None, None, "depends on forcing and stiffness"),
    },
    "flight_mechanics": {
        "max_angular_rate_rad_s": (0.0, 1.0, "typical small spacecraft angular rates"),
    },
    "coupled_physics": {
        "k_eff": (0.95, 1.05, "effective multiplication factor near criticality"),
        "temperature_max_K": (300, 2000, "fuel temperature range"),
    },
    "geospatial": {
        "coverage_fraction": (0.0, 1.0, "fraction of demand population served"),
    },
    "field_aware_routing": {
        "max_field_V_m": (None, None, "depends on boundary conditions and geometry"),
    },
    "neutronics": {
        "k_eff": (0.8, 1.5, "effective multiplication factor for typical reactor configurations"),
        "peaking_factor": (1.0, 3.0, "axial power peaking factor"),
    },
}

# ---------------------------------------------------------------------------
#  Industry Demo Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    # ---- Fundamentals ----
    {
        "id": "lid-driven-cavity",
        "title": "Lid-Driven Cavity Flow",
        "subtitle": "Laminar Navier-Stokes",
        "business_title": "Cooling System Flow Analysis",
        "industry_problem": "Equipment overheating from poorly mixed coolant costs unplanned downtime -- is the flow reaching dead zones?",
        "decision_focus": "Is recirculation strong enough to cool all corners, or does the geometry need baffles?",
        "description": "Classic 2D lid-driven cavity flow benchmark. Run, sweep Reynolds number, and validate against Ghia et al.",
        "prompt": "Analyze coolant flow in a 2D cavity. Lid velocity 1.0 m/s, 1m x 1m cavity, 32x32 grid, kinematic viscosity 0.01 m^2/s. Show velocity and pressure fields, then sweep viscosity from 0.005 to 0.05 in 5 steps.",
        "icon": "droplets",
        "category": "fundamentals",
    },
    {
        "id": "silicon-pn-junction",
        "title": "Silicon PN Junction",
        "subtitle": "Forward Bias Sweep",
        "business_title": "Diode Forward-Bias Performance",
        "industry_problem": "Power supply efficiency depends on diode turn-on characteristics -- wrong doping wastes energy as heat.",
        "decision_focus": "How much voltage headroom exists before thermal runaway, and what I-V curve does this doping produce?",
        "description": "Forward-biased silicon diode. Sweep voltage from 0 to 0.7V and analyse I-V characteristics.",
        "prompt": "Characterize a silicon PN junction at 300K with donor doping 1e17 and acceptor 5e16. Sweep applied voltage from 0.0 to 0.7 V in 5 steps and analyze the I-V curve.",
        "icon": "cpu",
        "category": "fundamentals",
    },
    {
        "id": "radar-sensor-network",
        "title": "Radar Sensor Network",
        "subtitle": "Detection Coverage Map",
        "business_title": "Perimeter Detection Coverage",
        "industry_problem": "Gaps in radar coverage create security blind spots that drive insurance premiums and incident risk.",
        "decision_focus": "How small a target can we reliably detect at the perimeter, and where are the blind zones?",
        "description": "X-band radar detection probability map with RCS sweep for stealth analysis.",
        "prompt": "Map radar detection coverage. Radar at 10 GHz, power_w=1000, aperture_diameter_m=1.0, bandwidth_mhz=10. Target RCS 1.0 m^2. Use auto_grid=true. Then compare RCS 0.1, 1.0, and 10.0 m^2.",
        "icon": "radar",
        "category": "fundamentals",
    },
    {
        "id": "reynolds-number-study",
        "title": "Reynolds Number Study",
        "subtitle": "Parameter Optimization",
        "business_title": "Mixing Efficiency Optimization",
        "industry_problem": "Suboptimal fluid mixing wastes energy and produces inconsistent product quality in processing equipment.",
        "decision_focus": "What viscosity setting maximizes vortex strength without exceeding pump capacity?",
        "description": "Find the optimal viscosity that maximises vortex strength in a cavity flow.",
        "prompt": "Optimize mixing in a lid-driven cavity. Find the kinematic viscosity between 0.005 and 0.05 that maximizes velocity magnitude. Use a 32x32 grid with lid velocity 1.0 m/s.",
        "icon": "target",
        "category": "fundamentals",
    },
    {
        "id": "diode-temperature-study",
        "title": "Diode Temperature Study",
        "subtitle": "Multi-Scenario Compare",
        "business_title": "Semiconductor Thermal Reliability",
        "industry_problem": "Junction performance degrades at temperature extremes -- field failures cost 10x factory rejects.",
        "decision_focus": "At what temperature does junction behavior degrade unacceptably, and how much margin exists?",
        "description": "Compare PN junction behaviour at different operating temperatures.",
        "prompt": "Compare PN junction behavior at 250K, 300K, 350K, and 400K with 0.6V applied bias, donor doping 1e17 and acceptor 5e16.",
        "icon": "thermometer",
        "category": "fundamentals",
    },
    # ---- Industry Problems ----
    {
        "id": "data-center-cold-aisle",
        "title": "Data Center Cold Aisle Containment",
        "subtitle": "Airflow Recirculation CFD",
        "business_title": "Data Center Cold Aisle Containment",
        "industry_problem": "Hot spots from poor airflow recirculation cause server throttling and unplanned outages -- costing $9,000/min in downtime.",
        "decision_focus": "Is the cold aisle supply reaching top-of-rack units, or do we need blanking panels and floor tile rearrangement?",
        "description": "Analyze cold aisle airflow distribution to identify recirculation hot spots that cause server throttling.",
        "prompt": "Analyze data center cold aisle airflow. Model as cavity flow with lid velocity 2.5 m/s, kinematic viscosity 1.5e-5 m^2/s, 32x32 grid. Show velocity and pressure, then sweep lid velocity from 1.0 to 5.0 m/s in 4 steps.",
        "icon": "server",
        "category": "industry",
    },
    {
        "id": "ev-battery-thermal-runaway",
        "title": "EV Battery Pack Thermal Runaway Margin",
        "subtitle": "Cell Temperature vs C-Rate",
        "business_title": "EV Battery Pack Thermal Runaway Margin",
        "industry_problem": "A single cell thermal event can cascade through the pack -- recall costs run $10K-$40K per vehicle and destroy brand trust.",
        "decision_focus": "At what C-rate does cell temperature exceed the separator melting point, and how much margin does the cooling plate provide?",
        "description": "Evaluate thermal margins in an EV battery cell under increasing discharge rates to predict runaway risk.",
        "prompt": "Assess EV battery thermal runaway. Compare PN junction at 300K, 350K, 400K, and 430K with 0.6V bias, donor 1e17, acceptor 5e16 to find the temperature where current spikes.",
        "icon": "battery-charging",
        "category": "industry",
    },
    {
        "id": "refinery-flare-exclusion",
        "title": "Refinery Flare Radiation Exclusion Zone",
        "subtitle": "Thermal Radiation Flux Mapping",
        "business_title": "Refinery Flare Radiation Exclusion Zone",
        "industry_problem": "Undersized exclusion zones risk worker burns and regulatory fines; oversized zones waste expensive plot space.",
        "decision_focus": "What is the thermal radiation flux at the fence line during emergency depressurization, and does it meet API 521 limits?",
        "description": "Map thermal radiation intensity around a refinery flare stack to define safe exclusion zone boundaries.",
        "prompt": "Map flare radiation exclusion zone. Use sensing module: frequency_ghz=10.0, power_w=5000, aperture_diameter_m=2.0, bandwidth_mhz=20. auto_grid=true, rcs_m2=1.0. Compare RCS 0.1, 1.0, and 10.0 m^2.",
        "icon": "flame",
        "category": "industry",
    },
    {
        "id": "pharma-mixing-scaleup",
        "title": "Pharmaceutical Mixing Vessel Scale-Up",
        "subtitle": "Blend Uniformity vs Impeller Speed",
        "business_title": "Pharmaceutical Mixing Vessel Scale-Up",
        "industry_problem": "Batch-to-batch variability from poor mixing at production scale delays FDA approval and scraps $500K API batches.",
        "decision_focus": "What impeller speed achieves 95% blend uniformity within 120 seconds without exceeding shear limits on the active ingredient?",
        "description": "Optimize mixing vessel parameters to achieve blend uniformity at production scale without exceeding shear limits.",
        "prompt": "Optimize pharma mixing vessel. Model as cavity flow, find optimal viscosity between 0.005 and 0.05 that maximizes velocity on a 32x32 grid with lid velocity 2.0 m/s. Sweep viscosity in 5 steps.",
        "icon": "flask-round",
        "category": "industry",
    },
    {
        "id": "gas-turbine-creep-life",
        "title": "Gas Turbine Blade Creep Life Prediction",
        "subtitle": "Temperature-Dependent Degradation",
        "business_title": "Gas Turbine Blade Creep Life Prediction",
        "industry_problem": "Blade replacements during unplanned outages cost $2M+ per event and take the unit offline for weeks.",
        "decision_focus": "At current firing temperatures, what is the remaining creep life of the first-stage blades, and how much derating extends the next inspection interval by 6 months?",
        "description": "Predict turbine blade material degradation across operating temperature ranges to schedule maintenance.",
        "prompt": "Predict gas turbine blade creep. Compare PN junction at 250K, 300K, 350K, and 400K with 0.65V bias, donor 1e17, acceptor 5e16. Find the degradation threshold.",
        "icon": "cog",
        "category": "industry",
    },
    {
        "id": "lng-sloshing-loads",
        "title": "LNG Sloshing Loads in Membrane Tanks",
        "subtitle": "Fill Level vs Wave Impact",
        "business_title": "LNG Sloshing Loads in Membrane Tanks",
        "industry_problem": "Partial fills during transit create sloshing impacts that crack membrane containment -- repair drydocks cost $5M+ and idle the vessel for months.",
        "decision_focus": "At what fill level and sea state do sloshing pressures exceed the membrane's fatigue limit, and should we mandate minimum fill restrictions?",
        "description": "Analyze sloshing dynamics in LNG membrane tanks at various fill levels to determine safe operating envelopes.",
        "prompt": "Analyze LNG sloshing loads. Model as cavity flow on 32x32 grid with viscosity 0.01. Sweep lid velocity from 0.5 to 5.0 m/s in 5 steps. Compare viscosities 0.005, 0.01, and 0.05.",
        "icon": "ship",
        "category": "industry",
    },
    {
        "id": "cleanroom-laminar-flow",
        "title": "Cleanroom Laminar Flow Validation",
        "subtitle": "FFU Velocity & ISO 14644 Compliance",
        "business_title": "Cleanroom Laminar Flow Validation",
        "industry_problem": "Particle contamination from turbulent zones on the wafer kills yield -- each 1% yield drop costs $50M/year on a high-volume line.",
        "decision_focus": "Where does the laminar flow break down above the wafer stage, and do the FFU velocities meet ISO 14644 Class 1 requirements?",
        "description": "Validate cleanroom airflow patterns to identify turbulent zones that compromise wafer yield.",
        "prompt": "Validate cleanroom laminar flow. Model as cavity with lid velocity 0.45 m/s, viscosity 0.01, 32x32 grid. Sweep viscosity from 0.005 to 0.02 in 4 steps.",
        "icon": "scan-line",
        "category": "industry",
    },
    {
        "id": "mine-ventilation-methane",
        "title": "Mine Ventilation Methane Dilution",
        "subtitle": "Ventilation Rate vs Dead Zones",
        "business_title": "Mine Ventilation Methane Dilution",
        "industry_problem": "Methane pockets in longwall panels reach explosive concentrations before sensors trigger -- responsible for 60% of underground mine fatalities.",
        "decision_focus": "Is the ventilation rate sufficient to keep methane below 1% LEL at the face, and where do recirculation dead zones form behind the shearer?",
        "description": "Analyze mine ventilation patterns to identify methane accumulation dead zones near the longwall face.",
        "prompt": "Analyze mine ventilation. Cavity flow with lid velocity 4.0 m/s, viscosity 0.01, 32x32 grid. Show velocity and pressure, then optimize viscosity between 0.005 and 0.02 to maximize velocity.",
        "icon": "hard-hat",
        "category": "industry",
    },
    {
        "id": "automotive-crumple-zone",
        "title": "Automotive Crash Crumple Zone Absorption",
        "subtitle": "Energy Absorption vs Intrusion",
        "business_title": "Automotive Crash Crumple Zone Absorption",
        "industry_problem": "Failing NCAP frontal offset tests delays launch by 6-12 months and requires expensive tooling changes to the body-in-white.",
        "decision_focus": "Does the current rail gauge and fold pattern absorb enough energy to keep cabin intrusion under 50mm at 64 km/h?",
        "description": "Evaluate crumple zone energy absorption characteristics under frontal impact loading conditions.",
        "prompt": "Evaluate crumple zone impact. Cavity flow with viscosity 0.01, 32x32 grid. Sweep lid velocity from 8 to 20 m/s in 5 steps. Optimize viscosity between 0.005 and 0.05 to maximize velocity.",
        "icon": "car",
        "category": "industry",
    },
    {
        "id": "district-heating-pressure",
        "title": "District Heating Network Pressure Drop",
        "subtitle": "Branch Sizing & Pump Compensation",
        "business_title": "District Heating Network Pressure Drop",
        "industry_problem": "Undersized pipes in branch lines starve end-of-network buildings of heat -- tenant complaints trigger regulatory penalties and retrofit costs.",
        "decision_focus": "Which branches exceed the allowable pressure drop at peak demand, and can variable-speed pumps compensate or do we need to upsize pipe?",
        "description": "Analyze pressure drop across a district heating network to identify undersized branches.",
        "prompt": "Analyze district heating pressure drop. Cavity flow with lid velocity 2.8 m/s, viscosity 0.01, 32x32 grid. Sweep viscosity from 0.005 to 0.02 in 4 steps.",
        "icon": "gauge",
        "category": "industry",
    },
    {
        "id": "offshore-wave-fatigue",
        "title": "Offshore Platform Wave Load Fatigue",
        "subtitle": "Cumulative Damage at K-Joints",
        "business_title": "Offshore Platform Wave Load Fatigue",
        "industry_problem": "Jacket weld joints accumulate fatigue damage from millions of wave cycles -- a missed crack grows to failure in one storm season.",
        "decision_focus": "What is the cumulative fatigue damage ratio at the critical K-joints after 25 years of West Africa sea states, and does it stay below the DNV threshold of 0.1?",
        "description": "Evaluate cumulative wave load fatigue damage on offshore platform jacket K-joints.",
        "prompt": "Evaluate offshore wave fatigue. Compare cavity flow at lid velocities 2.0, 5.0, 10.0, and 14.0 m/s on 32x32 grid with viscosity 0.01. Find the sea state exceeding the fatigue threshold.",
        "icon": "anchor",
        "category": "industry",
    },
    {
        "id": "steel-caster-mold-heat",
        "title": "Steel Continuous Caster Mold Heat Flux",
        "subtitle": "Mold Water Flow Uniformity",
        "business_title": "Steel Continuous Caster Mold Heat Flux",
        "industry_problem": "Uneven mold cooling causes shell thinning and breakouts -- a single breakout dumps molten steel, costs $1M in lost production, and risks worker safety.",
        "decision_focus": "Is the mold water flow rate producing uniform heat extraction across all four faces, or are the narrow faces starved?",
        "description": "Analyze cooling water flow distribution in a continuous caster mold to prevent shell thinning breakouts.",
        "prompt": "Analyze steel caster mold cooling. Cavity flow with lid velocity 3.0 m/s, viscosity 0.01, 32x32 grid. Sweep viscosity from 0.005 to 0.02 in 4 steps.",
        "icon": "factory",
        "category": "industry",
    },
    {
        "id": "hvac-smoke-evacuation",
        "title": "HVAC Smoke Evacuation in High-Rise Atrium",
        "subtitle": "Smoke Layer Height vs Exhaust Rate",
        "business_title": "HVAC Smoke Evacuation in High-Rise Atrium",
        "industry_problem": "Smoke stratification in tall atriums traps occupants above the smoke layer -- fire code violations block the certificate of occupancy.",
        "decision_focus": "Does the smoke exhaust system maintain a clear layer above 2.5m at the highest occupied floor within 3 minutes of fire ignition?",
        "description": "Validate smoke exhaust system performance in a high-rise atrium to meet fire code egress requirements.",
        "prompt": "Validate HVAC smoke evacuation. Cavity flow with lid velocity 5.0 m/s, viscosity 0.01, 32x32 grid. Sweep lid velocity from 1.0 to 8.0 m/s in 4 steps to find minimum exhaust rate.",
        "icon": "building",
        "category": "industry",
    },
    {
        "id": "wind-farm-wake-loss",
        "title": "Wind Farm Wake Interference Loss",
        "subtitle": "Turbine Spacing & Array Efficiency",
        "business_title": "Wind Farm Wake Interference Loss",
        "industry_problem": "Downstream turbines sit in upstream wakes and lose 10-20% annual energy production -- turning a profitable project unfinanceable.",
        "decision_focus": "What turbine spacing and yaw offset strategy minimizes array losses while keeping total capacity above the grid connection limit?",
        "description": "Optimize wind farm turbine spacing to minimize wake interference losses across the array.",
        "prompt": "Analyze wind farm wake loss. Cavity flow with lid velocity 12.0 m/s, 32x32 grid. Sweep viscosity from 1.0 to 10.0 in 5 steps. Optimize viscosity between 1.0 and 10.0 to maximize downstream velocity.",
        "icon": "wind",
        "category": "industry",
    },
    {
        "id": "desalination-membrane-fouling",
        "title": "Desalination RO Membrane Fouling Prediction",
        "subtitle": "Transmembrane Pressure vs Recovery",
        "business_title": "Desalination RO Membrane Fouling Prediction",
        "industry_problem": "Membrane fouling cuts permeate flow and spikes energy costs -- unplanned CIP cycles reduce plant availability below the offtake contract minimum.",
        "decision_focus": "At current feed turbidity and recovery ratio, how many operating hours before transmembrane pressure exceeds the CIP trigger, and does reducing recovery to 42% meaningfully extend the interval?",
        "description": "Predict RO membrane fouling rates to optimize cleaning schedules and recovery ratios.",
        "prompt": "Predict desalination membrane fouling. Cavity flow with lid velocity 0.18 m/s, 32x32 grid. Sweep viscosity from 0.005 to 0.05 in 5 steps to see how pressure drop increases with fouling.",
        "icon": "filter",
        "category": "industry",
    },
    # ---- New Module Scenarios ----
    {
        "id": "pcb-trace-electrostatics",
        "title": "PCB Trace Electric Field Analysis",
        "subtitle": "Electrostatic Field Mapping",
        "business_title": "PCB High-Voltage Clearance Validation",
        "industry_problem": "High-voltage traces too close to signal paths cause arcing and EMI failures -- redesign after first article costs 6 weeks and $200K in NRE.",
        "decision_focus": "Does the field strength between the 1kV bus and the signal trace stay below the dielectric breakdown threshold?",
        "description": "Map electric field distribution on a PCB cross-section to validate trace clearances.",
        "prompt": "Validate PCB high-voltage clearances. Electrostatic field on 0.1m x 0.1m board at 30-point resolution with 1000V boundary. Sweep boundary voltage from 100 to 2000 V in 5 steps.",
        "icon": "zap",
        "category": "fundamentals",
    },
    {
        "id": "reentry-vehicle-heating",
        "title": "Reentry Vehicle Aero-Thermal Loads",
        "subtitle": "Hypersonic Heating Distribution",
        "business_title": "Reentry Vehicle TPS Sizing",
        "industry_problem": "Under-designed thermal protection burns through on reentry -- over-designed TPS adds mass that cuts payload by 15%.",
        "decision_focus": "What is the peak heat flux at the nose, and how does angle of attack redistribute heating along the body?",
        "description": "Compute distributed aerodynamic loads and heating on a reentry vehicle at Mach 5.",
        "prompt": "Size reentry vehicle TPS. Body 3m long, nose radius 0.15m, velocity 1500 m/s, density 0.1 kg/m^3, 50 panels. Run at 5 degrees AoA, then sweep alpha from 0 to 10 degrees in 4 steps.",
        "icon": "flame",
        "category": "fundamentals",
    },
    {
        "id": "uav-wing-design",
        "title": "UAV Wing Lift Distribution",
        "subtitle": "Vortex Lattice Analysis",
        "business_title": "UAV Wing Span Optimisation",
        "industry_problem": "Sub-optimal wing geometry reduces endurance by 20% -- each extra kg of induced drag costs $15K/year in fuel on a commercial UAV fleet.",
        "decision_focus": "What wingspan and taper ratio minimise induced drag while meeting the lift requirement at cruise?",
        "description": "Analyse wing lift and drag using the Vortex Lattice Method for a fixed-wing UAV.",
        "prompt": "Design a UAV wing. Span 10m, root chord 1.0m, tip chord 0.6m, 20 spanwise panels, 5 degrees AoA, 25 m/s. Sweep AoA from 0 to 10 degrees in 5 steps. Compare spans 8m, 10m, and 12m.",
        "icon": "plane",
        "category": "fundamentals",
    },
    {
        "id": "satellite-thermal-budget",
        "title": "Satellite Thermal Budget",
        "subtitle": "Multi-Node Transient Analysis",
        "business_title": "Satellite Thermal Design Verification",
        "industry_problem": "Electronics outside their survival temperature range during eclipse cause mission loss -- thermal redesign delays launch by 9 months.",
        "decision_focus": "Do all nodes stay within survival limits through a full orbit including eclipse, and how much heater power is needed?",
        "description": "Evaluate multi-node spacecraft thermal response through an orbital heating cycle.",
        "prompt": "Verify CubeSat thermal design. 3-node model, 30W electronics, 600s analysis, dt=5.0. Sweep internal power from 10 to 80W in 4 steps. Compare 2-node vs 4-node.",
        "icon": "satellite",
        "category": "fundamentals",
    },
    {
        "id": "ev-inverter-thermal",
        "title": "EV Inverter Power Module Thermal",
        "subtitle": "IGBT Junction Temperature",
        "business_title": "EV Inverter Thermal Derating",
        "industry_problem": "IGBT junction overtemperature triggers derating that cuts motor torque during hill climbs -- customer complaints drive warranty costs.",
        "decision_focus": "At 200A phase current, does the junction stay below 175C with the current heatsink, or do we need to upsize the cold plate?",
        "description": "Transient thermal analysis of an automotive power electronics assembly under load.",
        "prompt": "Evaluate EV inverter thermal. 2-component model at 150A, h=100 W/m^2K, 10 seconds, dt=0.5. Sweep current from 100 to 300A in 4 steps. Compare h=50, 100, and 200.",
        "icon": "thermometer",
        "category": "fundamentals",
    },
    {
        "id": "battery-fast-charge",
        "title": "Battery Fast-Charge Thermal Safety",
        "subtitle": "Cell Temperature vs C-Rate",
        "business_title": "Fast-Charge Protocol Safety Validation",
        "industry_problem": "Aggressive fast-charge profiles push cells past separator melting -- a single thermal event cascades through the pack costing $10K-$40K per vehicle recall.",
        "decision_focus": "At what C-rate does cell temperature exceed the NMC runaway threshold of 130C, and how much does liquid cooling extend the safe envelope?",
        "description": "Evaluate battery thermal safety under increasing discharge/charge rates.",
        "prompt": "Validate battery fast-charge. 24-cell NMC pack, liquid indirect cooling, 50A for 60s, dt=1.0. Sweep current from 30 to 150A in 4 steps.",
        "icon": "battery-charging",
        "category": "fundamentals",
    },
    {
        "id": "cantilever-beam-stress",
        "title": "Cantilever Beam Stress & Buckling",
        "subtitle": "Static Structural FEM",
        "business_title": "Structural Member Sizing",
        "industry_problem": "Undersized structural members fail in service while oversized ones waste material and add weight -- both cost money.",
        "decision_focus": "Does the current cross-section provide positive buckling margin under the design load, and what is the weight penalty for a 1.5x safety factor?",
        "description": "Static stress analysis with buckling check for a loaded cantilever beam.",
        "prompt": "Check cantilever beam. AL7075-T6, 1.0m long, 10 elements, I=1e-6, A=1e-4, c=0.01, tip load 5000N. Sweep force from 1000 to 10000N in 4 steps. Compare cantilever vs simply_supported.",
        "icon": "ruler",
        "category": "fundamentals",
    },
    {
        "id": "vibration-response",
        "title": "Structural Vibration Response",
        "subtitle": "Newmark-Beta Time Integration",
        "business_title": "Equipment Vibration Qualification",
        "industry_problem": "Resonance at natural frequencies causes fatigue failures in vibration-sensitive equipment -- replacing a failed unit costs 10x prevention.",
        "decision_focus": "Do any natural frequencies coincide with the excitation spectrum, and what is the peak acceleration at the mounting point?",
        "description": "Time-domain dynamic response of a multi-DOF structural system under harmonic forcing.",
        "prompt": "Qualify equipment for vibration. 3-DOF, stiffness [1000, 2000, 3000] N/m, 2% damping, 100N at 10 Hz for 0.5 seconds, dt=0.005. Sweep frequency from 5 to 30 Hz in 4 steps. Compare damping 0.01, 0.02, and 0.05.",
        "icon": "activity",
        "category": "fundamentals",
    },
    {
        "id": "satellite-detumble",
        "title": "Satellite Detumble Manoeuvre",
        "subtitle": "6-DOF Attitude Dynamics",
        "business_title": "Post-Deployment Attitude Acquisition",
        "industry_problem": "Failed detumble after deployment leaves the satellite tumbling -- no power generation means mission loss within hours.",
        "decision_focus": "How long does the B-dot controller take to detumble from worst-case tip-off rates, and is the fuel budget sufficient?",
        "description": "Evaluate 6-DOF rigid body dynamics for a satellite detumble scenario.",
        "prompt": "Verify satellite detumble. 500 kg, Ixx=100, Iyy=120, Izz=80, omega_x=0.01, omega_y=-0.02, omega_z=0.005, dt=0.05 for 10 seconds. Compare omega_x = 0.01, 0.05, and 0.1 rad/s.",
        "icon": "rotate-3d",
        "category": "fundamentals",
    },
    {
        "id": "reactor-rod-withdrawal",
        "title": "Reactor Rod Withdrawal Transient",
        "subtitle": "Coupled Neutronics-Thermal",
        "business_title": "Nuclear Reactor Safety Analysis",
        "industry_problem": "Uncontrolled reactivity insertion causes power excursion and fuel damage -- conservative limits reduce plant capacity factor by 5%.",
        "decision_focus": "For a 100 pcm rod withdrawal, does Doppler feedback arrest the power excursion before fuel temperature limits are reached?",
        "description": "Evaluate a coupled neutronics-thermal transient with reactivity feedback.",
        "prompt": "Analyze rod withdrawal transient. 30-point, 200cm core, full feedback, 100 pcm insertion for 2 seconds, dt=0.01. Sweep reactivity from 50 to 400 pcm in 4 steps. Compare full, doppler_only, and no_feedback modes.",
        "icon": "atom",
        "category": "fundamentals",
    },
    {
        "id": "warehouse-network",
        "title": "Warehouse Network Optimisation",
        "subtitle": "Facility Location Coverage",
        "business_title": "Distribution Network Design",
        "industry_problem": "Poor warehouse placement leaves 30% of demand outside 24h delivery -- lost sales exceed $2M/year per uncovered region.",
        "decision_focus": "How many warehouses are needed to achieve 95% population coverage within 24 hours, and where should they be located?",
        "description": "Optimize warehouse locations to maximize population coverage under travel-time constraints.",
        "prompt": "Design distribution network. Find optimal 3 warehouse locations from 8 candidates, 24h travel limit, state highways. Sweep max_facilities from 1 to 4. Compare state_highway vs national_highway.",
        "icon": "warehouse",
        "category": "fundamentals",
    },
    {
        "id": "pcb-em-routing",
        "title": "PCB EM-Aware Trace Routing",
        "subtitle": "Electric Field Cost Map",
        "business_title": "EMI-Aware PCB Layout",
        "industry_problem": "Signal traces routed through high-field regions pick up EMI -- board respins cost $150K and 8 weeks each.",
        "decision_focus": "Where are the high-field zones between the power and ground planes, and which routing corridors have field strength below the coupling threshold?",
        "description": "Generate electric field cost maps for physics-aware PCB trace routing.",
        "prompt": "Plan PCB trace routes. Map electric field on 1.0m x 1.0m at 32x32 resolution, 1000V left boundary. Sweep voltage from 200 to 1500V in 4 steps.",
        "icon": "circuit-board",
        "category": "fundamentals",
    },
    {
        "id": "reactor-core-criticality",
        "title": "Reactor Core Criticality Analysis",
        "subtitle": "Two-Group Neutron Diffusion",
        "business_title": "Reactor Core Design Verification",
        "industry_problem": "Under-moderated cores waste fuel while over-moderated ones hit safety limits -- each 1% k-eff uncertainty costs months of licensing review.",
        "decision_focus": "Is the core k-eff within the target band, and how much reflector thickness is needed to maintain criticality with 3% enriched fuel?",
        "description": "Solve the two-group neutron diffusion eigenvalue problem for a reactor core with reflectors.",
        "prompt": "Verify reactor core criticality. 200 cm core, 30 spatial points, FUEL_UO2_3PCT, REFLECTOR_H2O (30 cm sides). Report k-eff and peaking factor. Compare FUEL_UO2_3PCT vs FUEL_UO2_5PCT. Sweep reflector thickness from 10 to 50 cm in 4 steps.",
        "icon": "atom",
        "category": "fundamentals",
    },
    # ---- Advanced Multi-Physics ----
    {
        "id": "hypersonic-failure-envelope",
        "title": "Hypersonic Vehicle Failure Envelope",
        "subtitle": "Aero-Thermal-Structural Coupling",
        "business_title": "Hypersonic Vehicle Failure Envelope",
        "industry_problem": "Hypersonic vehicles fail from coupled thermal + structural + aero effects in seconds — design margins are razor-thin.",
        "decision_focus": "At Mach 8, what angle of attack causes structural failure due to thermal stress before lift becomes unstable?",
        "description": "Coupled aero-thermal-structural failure boundary for a hypersonic vehicle.",
        "prompt": "Find the hypersonic failure envelope. First run aero_loads: body 3m, nose radius 0.15m, velocity 2500 m/s, density 0.05 kg/m^3, 50 panels. Sweep alpha from 0 to 15 degrees in 5 steps to map heat flux vs angle of attack. Then run structural_analysis: AL7075-T6 cantilever, 1.0m, 10 elements, sweep tip force from 5000 to 50000N in 5 steps to find the failure load. Compare the aero heating at each AoA against the structural limit.",
        "icon": "rocket",
        "category": "advanced",
    },
    {
        "id": "missile-interception-feasibility",
        "title": "Missile Interception Feasibility",
        "subtitle": "Sensing + Kinematics Kill Chain",
        "business_title": "Missile Interception Physics Kill Switch",
        "industry_problem": "Not all interception scenarios are physically possible — committing resources to impossible intercepts wastes assets.",
        "decision_focus": "Given interceptor speed, radar latency, and target maneuverability — is interception physically possible?",
        "description": "Determine whether an interception is physically feasible from radar detection to kinematic closure.",
        "prompt": "Assess missile interception feasibility. First run sensing: radar at frequency_ghz=10, power_w=5000, aperture_diameter_m=2.0, auto_grid=true, target rcs_m2=0.1. Check if detection range exceeds 5 km. Then run flight_mechanics: interceptor mass=50 kg, Ixx=5, Iyy=5, Izz=3, dt=0.05, for 5 seconds with omega_x=0.1 to check if the interceptor can manoeuvre to the target within the detection window. Compare detection at rcs_m2=0.01, 0.1, and 1.0.",
        "icon": "crosshair",
        "category": "advanced",
    },
    {
        "id": "tsunami-impact-zones",
        "title": "Tsunami Impact Propagation",
        "subtitle": "Wave Energy + Coastal Geometry",
        "business_title": "Tsunami Coastal Damage Zone Mapping",
        "industry_problem": "Predicting which coastal zones exceed destructive wave energy thresholds takes days of analysis.",
        "decision_focus": "Which coastal zones exceed the destructive wave energy threshold for a given earthquake magnitude?",
        "description": "Map wave energy propagation to identify high-risk coastal zones.",
        "prompt": "Map tsunami impact zones. Model the wave front as a cavity flow with lid velocity representing wave speed. Run navier_stokes: 32x32 grid, lid velocity 5.0 m/s (shallow water wave), viscosity 0.01. Sweep lid velocity from 2.0 to 10.0 m/s in 5 steps representing different earthquake magnitudes. Identify where velocity magnitude exceeds the 2.0 m/s destructive threshold.",
        "icon": "waves",
        "category": "advanced",
    },
    {
        "id": "power-grid-cascade",
        "title": "Power Grid Cascade Failure",
        "subtitle": "Network + Thermal + Load Dynamics",
        "business_title": "Power Grid Cascade Failure Propagation",
        "industry_problem": "Grid failures cascade unpredictably — one overloaded substation can black out a region in seconds.",
        "decision_focus": "If node X fails, which substations overload within 5 seconds?",
        "description": "Model thermal overload propagation through an electrical network.",
        "prompt": "Model power grid cascade failure. Use automotive_thermal: 4-component model representing substations, current 300A (overload), h=50 W/m^2K, T_ambient=313K (40C summer), for 5 seconds dt=0.1. Sweep current from 200 to 500A in 5 steps to find the cascade threshold. Compare h=20 (poor cooling) vs h=100 (good cooling).",
        "icon": "plug-zap",
        "category": "advanced",
    },
    {
        "id": "plasma-containment-stability",
        "title": "Plasma Containment Stability",
        "subtitle": "MHD + Thermal + EM Coupling",
        "business_title": "Fusion Plasma Confinement Stability Boundary",
        "industry_problem": "Plasma instability destroys magnetic confinement — each disruption damages the tokamak wall.",
        "decision_focus": "At what magnetic field strength does plasma become unstable for given density and temperature?",
        "description": "Find the plasma confinement stability boundary using coupled EM and thermal models.",
        "prompt": "Analyze plasma confinement stability. Use electrostatics as a proxy for the confining field: 0.5m x 0.5m domain at 30-point resolution. Sweep boundary voltage from 500 to 5000V in 5 steps (representing B-field strength). At each level, check if peak field strength exceeds 1e6 V/m (proxy for plasma pressure exceeding magnetic pressure — beta limit). Find the minimum confinement field needed.",
        "icon": "atom",
        "category": "advanced",
    },
    {
        "id": "drone-swarm-detectability",
        "title": "Drone Swarm RF Detectability",
        "subtitle": "Radar + Formation Geometry",
        "business_title": "Drone Swarm Stealth Envelope",
        "industry_problem": "Small drone swarms evade traditional radar — operators don't know their detection blind spots.",
        "decision_focus": "At what range and formation density does a drone swarm become undetectable to radar?",
        "description": "Map radar detection limits for small-RCS drone swarms at varying ranges.",
        "prompt": "Assess drone swarm detectability. Run sensing: radar at frequency_ghz=10, power_w=2000, aperture_diameter_m=1.5, bandwidth_mhz=15, auto_grid=true. Sweep target rcs_m2 from 0.001 to 1.0 in 5 steps (representing single micro-drone to full swarm aggregate RCS). Find the RCS below which detection probability falls under 50% at 5 km range.",
        "icon": "radio",
        "category": "advanced",
    },
    {
        "id": "battery-runaway-propagation",
        "title": "Battery Thermal Runaway Propagation",
        "subtitle": "Cell-to-Cell Heat Cascade",
        "business_title": "Battery Pack Thermal Runaway Chain Reaction",
        "industry_problem": "A single cell at 150°C can cascade through the pack — each cell that ignites adds 40kJ of energy.",
        "decision_focus": "If one cell hits runaway, how many adjacent cells exceed the threshold and what cooling stops the cascade?",
        "description": "Model thermal runaway propagation across a battery pack under different cooling scenarios.",
        "prompt": "Analyze battery runaway propagation. Run battery_thermal: 24 cells, NMC, liquid_indirect cooling, discharge at 150A for 60s, dt=1.0. Compare cooling types: liquid_indirect vs air_forced vs liquid_immersion. Sweep discharge current from 100 to 250A in 4 steps to find the threshold where runaway_risk becomes true.",
        "icon": "flame",
        "category": "advanced",
    },
    {
        "id": "factory-layout-optimization",
        "title": "Factory Layout Throughput Optimization",
        "subtitle": "Spatial + Flow + Thermal Coupling",
        "business_title": "Factory Layout Multi-Objective Optimization",
        "industry_problem": "Bad layout kills throughput — machines overheat, material flow bottlenecks form, and output drops 30%.",
        "decision_focus": "Which machine placement minimizes flow bottlenecks and thermal hotspots simultaneously?",
        "description": "Optimize factory layout using geospatial routing and thermal analysis.",
        "prompt": "Optimize factory layout. Run geospatial: find optimal 3 facility locations from 8 candidates within 12h travel time. Then run field_aware_routing: 32x32 grid, 500V left boundary to map flow cost fields. Compare 2-facility vs 3-facility vs 4-facility configurations for coverage vs cost trade-off.",
        "icon": "layout-grid",
        "category": "advanced",
    },
    {
        "id": "wind-turbine-gust-failure",
        "title": "Wind Turbine Gust Failure Threshold",
        "subtitle": "Aero + Structural Dynamics",
        "business_title": "Wind Turbine Extreme Gust Failure Envelope",
        "industry_problem": "Turbines fail under transient gusts — one blade failure in a 100-turbine farm costs $2M and 6 months downtime.",
        "decision_focus": "At what gust speed does blade stress exceed fatigue limits?",
        "description": "Find the gust speed threshold where structural dynamics exceed fatigue limits.",
        "prompt": "Find wind turbine gust failure threshold. Run structural_dynamics: 3-DOF system with stiffness [5000, 10000, 20000] N/m (blade modes), 1% damping, force_amplitude=500N, force_frequency=2 Hz (gust frequency), for 0.5s dt=0.005. Sweep force_amplitude from 200 to 2000N in 5 steps (representing gust speeds 15-50 m/s). Find the amplitude where peak acceleration exceeds 50 m/s^2 fatigue limit.",
        "icon": "wind",
        "category": "advanced",
    },
    {
        "id": "satellite-blackout-zones",
        "title": "Satellite Communication Blackout Zones",
        "subtitle": "Orbit + RF + Atmosphere",
        "business_title": "Satellite Signal Blackout Zone Mapping",
        "industry_problem": "Signal dropouts during critical orbital passes cause data loss and missed commands.",
        "decision_focus": "Where does signal-to-noise fall below the communication threshold during orbit?",
        "description": "Map communication blackout zones from orbital geometry and RF propagation.",
        "prompt": "Map satellite blackout zones. Run sensing: ground station radar at frequency_ghz=2.0, power_w=500, aperture_diameter_m=3.0, bandwidth_mhz=5, auto_grid=true, rcs_m2=10 (satellite effective area). Check coverage fraction at 90% threshold. Then run spacecraft_thermal: 3-node model, 20W electronics, 600s, dt=5 to verify the satellite electronics stay within operating range during the pass.",
        "icon": "satellite-dish",
        "category": "advanced",
    },
    {
        "id": "surgical-thermal-damage",
        "title": "Surgical Thermal Damage Boundary",
        "subtitle": "Heat Diffusion + Safety Margin",
        "business_title": "Surgical Laser Thermal Damage Threshold",
        "industry_problem": "Heat from surgical tools damages surrounding tissue — the safe power window is narrow.",
        "decision_focus": "At what laser power does tissue temperature exceed the safe 43°C threshold within 2 seconds?",
        "description": "Find the thermal damage boundary for surgical laser power using heat diffusion.",
        "prompt": "Find surgical thermal damage threshold. Use automotive_thermal as tissue model: 2-component (tissue + tool contact), T_ambient=310K (body temperature), h=10 W/m^2K (poor tissue convection), for 2 seconds dt=0.05. Sweep current_A from 50 to 300A in 5 steps (representing laser power 1-10W via I^2R heating). Find the power where peak temperature exceeds 316K (43°C tissue damage threshold).",
        "icon": "heart-pulse",
        "category": "advanced",
    },
    {
        "id": "crash-survivability-envelope",
        "title": "Crash Survivability Envelope",
        "subtitle": "Structural + Energy Absorption",
        "business_title": "Crash Survivability Speed Envelope",
        "industry_problem": "Crash outcomes depend on speed, material, and geometry — the survivability boundary is unknown for new designs.",
        "decision_focus": "At what impact speed does structural intrusion exceed the survivability threshold?",
        "description": "Find the crash survivability speed envelope from structural analysis.",
        "prompt": "Find crash survivability envelope. Run structural_analysis: AL7075-T6 cantilever, 0.5m long (crumple zone), 10 elements, I=5e-6, A=5e-4, c=0.02. Sweep tip_force_N from 10000 to 100000N in 5 steps (representing 20-80 km/h impact). Find the force where max deflection exceeds 50mm (cabin intrusion limit). Compare cantilever vs simply_supported (different crush modes).",
        "icon": "shield-alert",
        "category": "advanced",
    },
    {
        "id": "building-fire-collapse",
        "title": "Building Collapse Under Fire + Load",
        "subtitle": "Thermal Weakening + Structural Failure",
        "business_title": "Fire-Induced Structural Collapse Threshold",
        "industry_problem": "Fire weakens steel structure — collapse happens when load capacity drops below the applied load.",
        "decision_focus": "At what fire temperature does the load-bearing capacity drop below the safety factor?",
        "description": "Find the fire temperature where structural strength falls below the load requirement.",
        "prompt": "Analyze fire-induced collapse risk. Run structural_analysis at different temperatures (proxy via material degradation): AL7075-T6 beam, 2.0m, 10 elements, I=1e-5, A=1e-3, c=0.02, tip_force_N=-20000N. Compare simply_supported at load 20000N vs 40000N vs 60000N. Sweep force from 10000 to 80000N in 5 steps to find the collapse margin. The safety factor is max_von_mises / yield_stress.",
        "icon": "building-2",
        "category": "advanced",
    },
    {
        "id": "reactor-explosion-risk",
        "title": "Chemical Reactor Explosion Risk",
        "subtitle": "Reaction + Heat + Pressure Coupling",
        "business_title": "Chemical Reactor Runaway Explosion Threshold",
        "industry_problem": "Exothermic runaway reactions cause explosions — the safe operating envelope is narrow and temperature-dependent.",
        "decision_focus": "At what temperature and power does the reaction become uncontrollable?",
        "description": "Find the thermal runaway threshold for a chemical reactor using coupled thermal analysis.",
        "prompt": "Analyze reactor explosion risk. Use coupled_physics as a proxy for exothermic reaction dynamics: 30-point core, 200cm, full feedback, initial_power=1e6. Sweep reactivity_insertion_pcm from 100 to 600 pcm in 5 steps (representing increasing reaction rate). Find the threshold where feedback cannot arrest the excursion (converged=false). Compare full vs doppler_only vs no_feedback modes.",
        "icon": "flask-round",
        "category": "advanced",
    },
    {
        "id": "lightning-strike-damage",
        "title": "Lightning Strike Current Path",
        "subtitle": "EM Field + Material Response",
        "business_title": "Lightning Strike Damage Path Analysis",
        "industry_problem": "Lightning current follows the lowest-impedance path — components in that path can be destroyed.",
        "decision_focus": "Where does current density exceed safe limits during a lightning strike?",
        "description": "Map the current density distribution during a high-voltage transient to identify damage zones.",
        "prompt": "Analyze lightning strike damage path. Run electrostatics in conduction mode: 0.5m x 0.5m domain, 30-point resolution, mode=conduction, boundary 50000V (lightning impulse). Identify where peak field exceeds 1e6 V/m (component damage threshold). Then run field_aware_routing: 32x32, 50000V left boundary to map the current path and find low-field safe corridors. Sweep voltage from 10000 to 100000V in 5 steps.",
        "icon": "zap",
        "category": "advanced",
    },
]

# ---------------------------------------------------------------------------
#  Pydantic Models
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    prompt: str = ""
    scenario_id: Optional[str] = None
    replicate_api_token: Optional[str] = None
    llm_model: str = DEFAULT_MODEL


class AgentRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    replicate_api_token: str
    llm_model: str = DEFAULT_MODEL
    scenario_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None  # [{role, content}, ...]


# ---------------------------------------------------------------------------
#  Utility — safe serialization
# ---------------------------------------------------------------------------
def _safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value[:200]]
    if is_dataclass(value):
        return _safe_value(asdict(value))
    if hasattr(value, "__dict__"):
        return _safe_value(vars(value))
    return repr(value)


def _array_summary(values: Any) -> Dict[str, Any]:
    array = np.asarray(values)
    flat = array.flatten()
    summary: Dict[str, Any] = {"shape": list(array.shape), "size": int(array.size)}
    if flat.size and np.issubdtype(array.dtype, np.number):
        finite = flat[np.isfinite(flat)]
        summary["min"] = float(np.min(finite)) if finite.size > 0 else 0.0
        summary["max"] = float(np.max(finite)) if finite.size > 0 else 0.0
        summary["mean"] = float(np.mean(finite)) if finite.size > 0 else 0.0
    return summary


def _set_nested(d: Dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation: 'solver.nu' -> d['solver']['nu']."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: Dict, path: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot notation."""
    keys = path.split(".")
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
#  Serialize Triality results
# ---------------------------------------------------------------------------
def _serialize_state(result: RuntimeExecutionResult) -> Dict[str, Any]:
    state = result.generated_state
    if state is None:
        return {"fields": {}, "metadata": {}}
    fields = {}
    for fname, field in state.fields.items():
        arr = np.asarray(field.data, dtype=float)
        entry: Dict[str, Any] = {"unit": field.unit, "summary": _array_summary(arr)}
        # Include full data for visualization (2D heatmaps, 1D line plots)
        # Replace NaN/Inf with 0 so Plotly renders correctly
        if arr.size <= 10000:
            clean = np.where(np.isfinite(arr), arr, 0.0)
            entry["data"] = clean.tolist()
        fields[fname] = entry
    return {
        "solver_name": state.solver_name,
        "time": state.time,
        "metadata": _safe_value(state.metadata),
        "fields": fields,
    }


def _serialize_runtime_result(result: RuntimeExecutionResult) -> Dict[str, Any]:
    return {
        "module_name": result.module_name,
        "success": result.success,
        "status": result.status,
        "warnings": result.warnings,
        "error": result.error,
        "elapsed_time_s": result.elapsed_time_s,
        "residuals": _safe_value(result.residuals),
        "convergence": _safe_value(result.convergence),
        "state": _serialize_state(result),
    }


def _format_result(result: RuntimeExecutionResult, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Format a RuntimeExecutionResult into the JSON the agent sees and the LLM summarizes."""
    from triality.observables import compute_observables

    serialized = _serialize_runtime_result(result)

    # Add field stats for backwards compatibility
    fields_stats = {}
    state = result.generated_state
    if state:
        for fname, fld in state.fields.items():
            arr = np.asarray(fld.data, dtype=float)
            finite = arr[np.isfinite(arr)]
            fields_stats[fname] = {
                "shape": list(arr.shape),
                "min": float(np.min(finite)) if finite.size > 0 else 0.0,
                "max": float(np.max(finite)) if finite.size > 0 else 0.0,
                "mean": float(np.mean(finite)) if finite.size > 0 else 0.0,
            }
    serialized["fields_stats"] = fields_stats

    # --- Observable layer: compute domain-specific derived quantities ---
    if state and result.module_name:
        observables = compute_observables(
            result.module_name, state, config or {},
            native_result=result.result_payload,
        )
        serialized["observables"] = [o.to_dict() for o in observables]
    else:
        serialized["observables"] = []

    return serialized


# ---------------------------------------------------------------------------
#  TOOL EXECUTION FUNCTIONS
# ---------------------------------------------------------------------------
def tool_list_modules(**kwargs) -> Dict[str, Any]:
    """List all available runtime modules with descriptions."""
    modules = available_runtime_modules()
    info = {}
    for m in modules:
        mi = TRIALITY_MODULES_INFO.get(m, {})
        info[m] = {
            "description": mi.get("description", "No description"),
            "domain": mi.get("domain", "General"),
            "config_keys": mi.get("config_keys", {}),
        }
    return {"modules": info, "templates": available_runtime_templates()}


def tool_describe_module(module_name: str, **kwargs) -> Dict[str, Any]:
    """Describe a specific module's capabilities, config keys, and defaults."""
    if module_name not in available_runtime_modules():
        return {"error": f"Module '{module_name}' not found. Available: {available_runtime_modules()}"}
    handle = load_module(module_name)
    desc = handle.describe()
    mi = TRIALITY_MODULES_INFO.get(module_name, {})
    return {
        "module_name": module_name,
        "description": mi.get("description", ""),
        "domain": mi.get("domain", ""),
        "config_keys": mi.get("config_keys", {}),
        "defaults": mi.get("defaults", {}),
        "runtime_description": _safe_value(desc),
    }


def _solve_with_progress(solver, est_steps: int, label: str = ""):
    """Run a solver in a background thread, yielding real progress dicts.

    Yields zero or more ``{"_progress": True, ...}`` dicts while the solver
    runs, then yields the ``RuntimeExecutionResult`` as the final item.
    For fast solves (est_steps < 100) it runs inline with no thread overhead.
    """
    if est_steps < 100:
        # Fast path — no threading
        yield solver.solve_safe()
        return

    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = solver.solve_safe()
        except Exception as exc:
            error_holder[0] = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    started = time.perf_counter()
    thread.start()

    last_step = 0
    while thread.is_alive():
        thread.join(timeout=1.0)
        elapsed = time.perf_counter() - started
        real = solver.drain_progress()
        if real:
            latest = real[-1]
            last_step = latest.get("step", last_step)
            est_steps = latest.get("total", est_steps)
        if not thread.is_alive():
            break
        yield {"_progress": True, "step": last_step, "total": est_steps,
               "elapsed_s": round(elapsed, 1),
               "detail": f"{label}step {last_step}/{est_steps} — {elapsed:.0f}s" if label else f"step {last_step}/{est_steps} — {elapsed:.0f}s"}

    solver.drain_progress()

    if error_holder[0]:
        from triality import RuntimeExecutionResult
        yield RuntimeExecutionResult(
            module_name=solver.module_name, success=False, status="execution_error",
            warnings=[], residuals={}, convergence={}, elapsed_time_s=time.perf_counter() - started,
            result_payload=None, generated_state=None, description=solver.describe(),
            error=error_holder[0],
        )
    else:
        yield result_holder[0]


def tool_run_module(module_name: str, config: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Execute a single Triality runtime module with given config."""
    final = None
    for item in _iter_run_module(module_name, config):
        final = item
    return final or {"error": "No result produced."}


def _estimate_timesteps(config: Dict) -> int:
    """Estimate the number of time steps from a merged module config.

    Scans both top-level and nested 'solver'/'solve' sections for
    common time-stepping key patterns (t_end/dt, t_final/dt,
    duration_s/dt, max_steps) and returns the best estimate.
    Returns 0 if no time-stepping keys are found (steady-state solver).
    """
    # Flatten: look in top-level, solver, and solve sections
    pools = [config, config.get("solver", {}), config.get("solve", {})]

    t_end = None
    dt = None
    max_steps = None

    for pool in pools:
        for k, v in pool.items():
            kl = k.lower()
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if kl in ("t_end", "t_final", "duration_s") and t_end is None:
                t_end = fv
            elif kl == "dt" and dt is None:
                dt = fv
            elif kl == "max_steps" and max_steps is None:
                max_steps = int(fv)

    if t_end is not None and dt is not None and dt > 0:
        steps = int(t_end / dt)
        if max_steps is not None:
            steps = min(steps, max_steps)
        return steps
    if max_steps is not None:
        return max_steps
    return 0


def _iter_run_module(module_name: str, config: Optional[Dict] = None):
    """Generator: yields progress events during solve, then the final result.

    For solves estimated to take many steps, runs the solver in a background
    thread and emits honest time-based progress events every few seconds.
    """
    if module_name not in available_runtime_modules():
        yield {"error": f"Module '{module_name}' not found. Available: {available_runtime_modules()}"}
        return

    handle = load_module(module_name)
    defaults = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))
    merged = _deep_merge(defaults, config or {})
    solver = handle.from_config(merged) if merged else handle.from_demo_case()

    desc = solver.describe()
    est_steps = _estimate_timesteps(merged)

    yield {"_progress": True, "step": 0, "total": max(est_steps, 1),
           "status": "configuring", "module": module_name,
           "detail": f"Configuring {desc.get('domain', module_name)}..."}

    # Use shared progress helper
    result = None
    for item in _solve_with_progress(solver, est_steps, label=f"{module_name}: "):
        if hasattr(item, 'success'):
            result = item
        else:
            item["module"] = module_name
            yield item

    if result is None:
        yield {"error": "No result produced."}
        return

    yield {"_progress": True, "step": max(est_steps, 1), "total": max(est_steps, 1),
           "status": "done", "module": module_name,
           "elapsed_s": round(result.elapsed_time_s, 3),
           "detail": f"{'Converged' if result.success else 'Failed'} in {result.elapsed_time_s:.1f}s"}

    yield _format_result(result, config=merged)


def tool_sweep_parameter(
    module_name: str,
    config: Dict,
    param_path: str,
    values: Optional[List[float]] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    steps: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Sweep a parameter over a range and collect results."""
    # Consume the generator to get the final result
    final = None
    for item in _iter_sweep_parameter(module_name, config, param_path, values, start, end, steps):
        final = item
    return final or {"error": "No results produced."}


def _iter_sweep_parameter(
    module_name: str,
    config: Dict,
    param_path: str,
    values: Optional[List[float]] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    steps: int = 5,
):
    """Generator that yields progress dicts then the final result."""
    if values is None:
        if start is None or end is None:
            yield {"error": "Must provide either 'values' list or 'start'+'end' range."}
            return
        values = np.linspace(start, end, steps).tolist()

    if module_name not in available_runtime_modules():
        yield {"error": f"Module '{module_name}' not found."}
        return

    defaults = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))
    base_config = _deep_merge(defaults, config or {})
    sweep_results = []
    total = len(values)

    for step_i, val in enumerate(values):
        cfg = copy.deepcopy(base_config)
        _set_nested(cfg, param_path, val)
        try:
            handle = load_module(module_name)
            solver = handle.from_config(cfg)
            est = _estimate_timesteps(cfg)
            # Run inner solve with sub-step progress
            result = None
            for item in _solve_with_progress(solver, est, label=f"[{step_i+1}/{total}] "):
                if hasattr(item, 'success'):
                    result = item
                else:
                    item["step"] = step_i
                    item["total"] = total
                    yield {"_progress": True, **item}
            if result is None:
                sweep_results.append({"param_value": val, "success": False, "error": "No result"})
                continue
            entry: Dict[str, Any] = {
                "param_value": val,
                "success": result.success,
                "elapsed_s": result.elapsed_time_s,
                "fields": {},
                "metadata": {},
            }
            if result.generated_state:
                for fname, field in result.generated_state.fields.items():
                    arr = np.asarray(field.data, dtype=float)
                    finite = arr[np.isfinite(arr)]
                    entry["fields"][fname] = {
                        "min": float(np.min(finite)) if finite.size > 0 else 0.0,
                        "max": float(np.max(finite)) if finite.size > 0 else 0.0,
                        "mean": float(np.mean(finite)) if finite.size > 0 else 0.0,
                    }
                entry["metadata"] = _safe_value(result.generated_state.metadata)
                # Compute observables for this sweep point
                from triality.observables import compute_observables
                obs = compute_observables(module_name, result.generated_state, cfg, native_result=result.result_payload)
                entry["observables"] = {o.name: o.to_dict() for o in obs}
            sweep_results.append(entry)
        except Exception as exc:
            sweep_results.append({"param_value": val, "success": False, "error": str(exc)})

        # Yield progress (not the final result)
        yield {"_progress": True, "step": step_i + 1, "total": total,
               "param_value": val, "elapsed_s": sweep_results[-1].get("elapsed_s", 0),
               "status": "ok" if sweep_results[-1].get("success") else "error"}

    # Yield final result
    yield {
        "module_name": module_name,
        "param_path": param_path,
        "values": values,
        "results": sweep_results,
    }


def tool_optimize_parameter(
    module_name: str,
    config: Dict,
    param_path: str,
    bounds: List[float],
    objective_field: str,
    objective: str = "maximize",
    n_evals: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """Golden-section search optimization of a single parameter."""
    final = None
    for item in _iter_optimize_parameter(module_name, config, param_path, bounds, objective_field, objective, n_evals):
        final = item
    return final or {"error": "No results produced."}


def _iter_optimize_parameter(
    module_name: str,
    config: Dict,
    param_path: str,
    bounds: List[float],
    objective_field: str,
    objective: str = "maximize",
    n_evals: int = 10,
):
    """Generator that yields progress then final result."""
    if module_name not in available_runtime_modules():
        yield {"error": f"Module '{module_name}' not found."}
        return

    defaults = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))
    base_config = _deep_merge(defaults, config or {})
    a, b = bounds[0], bounds[1]
    gr = (math.sqrt(5) - 1) / 2
    evaluations = []
    eval_count = 0
    total_evals = n_evals * 2 + 1  # 2 evals per iteration + 1 final

    _pending_sub_progress = []  # sub-step progress from inner solves

    def evaluate(val: float) -> float:
        cfg = copy.deepcopy(base_config)
        _set_nested(cfg, param_path, val)
        handle = load_module(module_name)
        solver = handle.from_config(cfg)
        est = _estimate_timesteps(cfg)
        result = None
        for item in _solve_with_progress(solver, est, label=f"eval "):
            if hasattr(item, 'success'):
                result = item
            else:
                _pending_sub_progress.append(item)
        if result is None or not result.success or not result.generated_state:
            return float("-inf") if objective == "maximize" else float("inf")
        for fname, field in result.generated_state.fields.items():
            if objective_field.lower() in fname.lower():
                arr = np.asarray(field.data)
                finite = np.abs(arr)[np.isfinite(arr)]
                metric = float(np.max(finite)) if finite.size > 0 else 0.0
                evaluations.append({"param_value": val, "metric": metric})
                return metric
        first_field = list(result.generated_state.fields.values())[0]
        arr = np.asarray(first_field.data)
        finite = np.abs(arr)[np.isfinite(arr)]
        metric = float(np.max(finite)) if finite.size > 0 else 0.0
        evaluations.append({"param_value": val, "metric": metric})
        return metric

    for step_i in range(n_evals):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = evaluate(c)
        eval_count += 1
        # Yield any sub-step progress from the inner solve
        for sp in _pending_sub_progress:
            sp["step"] = step_i
            sp["total"] = n_evals
            yield {"_progress": True, **sp}
        _pending_sub_progress.clear()
        fd = evaluate(d)
        eval_count += 1
        for sp in _pending_sub_progress:
            sp["step"] = step_i
            sp["total"] = n_evals
            yield {"_progress": True, **sp}
        _pending_sub_progress.clear()
        if objective == "maximize":
            if fc > fd:
                b = d
            else:
                a = c
        else:
            if fc < fd:
                b = d
            else:
                a = c

        best_so_far = max(evaluations, key=lambda e: e["metric"]) if objective == "maximize" else min(evaluations, key=lambda e: e["metric"])
        yield {"_progress": True, "step": step_i + 1, "total": n_evals,
               "eval_count": eval_count, "best_metric": best_so_far["metric"],
               "best_param": best_so_far["param_value"], "status": "ok"}

    optimal_val = (a + b) / 2
    # Run final evaluation and capture observables
    cfg_opt = copy.deepcopy(base_config)
    _set_nested(cfg_opt, param_path, optimal_val)
    handle = load_module(module_name)
    solver = handle.from_config(cfg_opt)
    result_opt = solver.solve_safe()
    optimal_metric = 0.0
    optimal_observables = []
    if result_opt.success and result_opt.generated_state:
        for fname, field in result_opt.generated_state.fields.items():
            if objective_field.lower() in fname.lower():
                arr = np.asarray(field.data)
                finite = np.abs(arr)[np.isfinite(arr)]
                optimal_metric = float(np.max(finite)) if finite.size > 0 else 0.0
                break
        else:
            first_field = list(result_opt.generated_state.fields.values())[0]
            arr = np.asarray(first_field.data)
            finite = np.abs(arr)[np.isfinite(arr)]
            optimal_metric = float(np.max(finite)) if finite.size > 0 else 0.0
        evaluations.append({"param_value": optimal_val, "metric": optimal_metric})
        from triality.observables import compute_observables
        obs = compute_observables(module_name, result_opt.generated_state, cfg_opt, native_result=result_opt.result_payload)
        optimal_observables = [o.to_dict() for o in obs]

    yield {
        "module_name": module_name,
        "param_path": param_path,
        "objective": objective,
        "objective_field": objective_field,
        "optimal_param_value": optimal_val,
        "optimal_metric": optimal_metric,
        "bounds": bounds,
        "evaluations": evaluations,
        "observables": optimal_observables,
    }


def tool_compare_scenarios(
    module_name: str,
    scenarios: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Run N configs side-by-side and return a comparison table."""
    final = None
    for item in _iter_compare_scenarios(module_name, scenarios):
        final = item
    return final or {"error": "No results produced."}


def _iter_compare_scenarios(
    module_name: str,
    scenarios: List[Dict[str, Any]],
):
    """Generator that yields progress then final result."""
    if module_name not in available_runtime_modules():
        yield {"error": f"Module '{module_name}' not found."}
        return

    defaults = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))
    comparison = []
    total = len(scenarios)

    for i, scenario in enumerate(scenarios):
        label = scenario.get("label", f"scenario_{i}")
        cfg = _deep_merge(defaults, scenario.get("config", {}))
        try:
            handle = load_module(module_name)
            solver = handle.from_config(cfg)
            est = _estimate_timesteps(cfg)
            result = None
            for item in _solve_with_progress(solver, est, label=f"[{i+1}/{total}] "):
                if hasattr(item, 'success'):
                    result = item
                else:
                    item["step"] = i
                    item["total"] = total
                    yield {"_progress": True, **item}
            if result is None:
                comparison.append({"label": label, "success": False, "error": "No result"})
                continue
            entry: Dict[str, Any] = {
                "label": label,
                "config": _safe_value(cfg),
                "success": result.success,
                "elapsed_s": result.elapsed_time_s,
                "fields": {},
                "metadata": {},
            }
            if result.generated_state:
                for fname, field in result.generated_state.fields.items():
                    arr = np.asarray(field.data, dtype=float)
                    finite = arr[np.isfinite(arr)]
                    entry["fields"][fname] = {
                        "min": float(np.min(finite)) if finite.size > 0 else 0.0,
                        "max": float(np.max(finite)) if finite.size > 0 else 0.0,
                        "mean": float(np.mean(finite)) if finite.size > 0 else 0.0,
                    }
                entry["metadata"] = _safe_value(result.generated_state.metadata)
                from triality.observables import compute_observables
                obs = compute_observables(module_name, result.generated_state, cfg, native_result=result.result_payload)
                entry["observables"] = {o.name: o.to_dict() for o in obs}
            comparison.append(entry)
        except Exception as exc:
            comparison.append({"label": label, "success": False, "error": str(exc)})

        yield {"_progress": True, "step": i + 1, "total": total,
               "label": label, "elapsed_s": comparison[-1].get("elapsed_s", 0),
               "status": "ok" if comparison[-1].get("success", False) else "error"}

    yield {"module_name": module_name, "comparison": comparison}


def tool_run_uncertainty_quantification(
    module_name: str,
    config: Dict,
    param_distributions: Dict[str, Dict[str, float]],
    n_samples: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """Monte Carlo uncertainty quantification. param_distributions: {path: {mean, std}}."""
    final = None
    for item in _iter_uncertainty_quantification(module_name, config, param_distributions, n_samples):
        final = item
    return final or {"error": "No results produced."}


def _iter_uncertainty_quantification(
    module_name: str,
    config: Dict,
    param_distributions: Dict[str, Dict[str, float]],
    n_samples: int = 10,
):
    """Generator that yields progress then final result."""
    if module_name not in available_runtime_modules():
        yield {"error": f"Module '{module_name}' not found."}
        return

    defaults = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))
    base_config = _deep_merge(defaults, config or {})
    all_metrics: Dict[str, List[float]] = {}

    for sample_i in range(n_samples):
        cfg = copy.deepcopy(base_config)
        for path, dist in param_distributions.items():
            if not isinstance(dist, dict) or "mean" not in dist or "std" not in dist:
                continue  # skip non-numeric distributions (discrete, categorical)
            try:
                mean_val = float(dist["mean"])
                std_val = float(dist["std"])
            except (TypeError, ValueError):
                continue  # skip non-numeric parameters
            sample = np.random.normal(mean_val, std_val)
            if mean_val > 0:
                sample = max(sample, 1e-12)
            _set_nested(cfg, path, float(sample))
        try:
            handle = load_module(module_name)
            solver = handle.from_config(cfg)
            est = _estimate_timesteps(cfg)
            result = None
            for item in _solve_with_progress(solver, est, label=f"[{sample_i+1}/{n_samples}] "):
                if hasattr(item, 'success'):
                    result = item
                else:
                    item["step"] = sample_i
                    item["total"] = n_samples
                    yield {"_progress": True, **item}
            if result is not None and result.success and result.generated_state:
                for fname, field in result.generated_state.fields.items():
                    arr = np.asarray(field.data)
                    key = f"{fname}_mean"
                    finite = arr[np.isfinite(arr)]
                    all_metrics.setdefault(key, []).append(float(np.mean(finite)) if finite.size > 0 else 0.0)
                    key = f"{fname}_max"
                    abs_finite = np.abs(arr)[np.isfinite(arr)]
                    all_metrics.setdefault(key, []).append(float(np.max(abs_finite)) if abs_finite.size > 0 else 0.0)
                # Also collect observable-level metrics for UQ
                from triality.observables import compute_observables
                obs = compute_observables(module_name, result.generated_state, cfg, native_result=result.result_payload)
                for o in obs:
                    if o.is_scalar and isinstance(o.value, (int, float)):
                        all_metrics.setdefault(f"obs:{o.name}", []).append(float(o.value))
        except Exception:
            pass

        yield {"_progress": True, "step": sample_i + 1, "total": n_samples, "status": "ok"}

    stats = {}
    for key, vals in all_metrics.items():
        a = np.array(vals)
        stats[key] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "cv_percent": float(np.std(a) / max(np.mean(a), 1e-30) * 100),
            "n_samples": len(vals),
        }

    yield {
        "module_name": module_name,
        "param_distributions": param_distributions,
        "n_samples": n_samples,
        "statistics": stats,
    }


def tool_chain_modules(
    steps: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Chain multiple module executions, passing state between them."""
    chain_results = []
    prev_state: Optional[Dict] = None

    for i, step in enumerate(steps):
        module_name = step.get("module_name", "")
        config = step.get("config", {})
        coupling = step.get("coupling", {})

        if module_name not in available_runtime_modules():
            chain_results.append({"step": i, "module": module_name, "error": "Module not found"})
            continue

        # Apply coupling from previous step
        if prev_state and coupling:
            for target_path, source_key in coupling.items():
                val = prev_state.get(source_key)
                if val is not None:
                    _set_nested(config, target_path, val)

        defaults = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))
        merged = _deep_merge(defaults, config)

        try:
            handle = load_module(module_name)
            solver = handle.from_config(merged)
            result = solver.solve_safe()
            formatted = _format_result(result, config=merged)
            formatted["step"] = i
            formatted["step_label"] = step.get("label", f"step_{i}")
            chain_results.append(formatted)

            # Capture state for next step
            prev_state = {}
            if result.generated_state:
                for fname, field in result.generated_state.fields.items():
                    arr = np.asarray(field.data)
                    finite = arr[np.isfinite(arr)]
                    prev_state[f"{fname}_mean"] = float(np.mean(finite)) if finite.size > 0 else 0.0
                    prev_state[f"{fname}_max"] = float(np.max(finite)) if finite.size > 0 else 0.0
                    prev_state[f"{fname}_min"] = float(np.min(finite)) if finite.size > 0 else 0.0
                prev_state.update(_safe_value(result.generated_state.metadata) or {})
        except Exception as exc:
            chain_results.append({"step": i, "module": module_name, "error": str(exc)})

    return {"chain_results": chain_results}


# ---------------------------------------------------------------------------
#  Tool Dispatch
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "list_modules": tool_list_modules,
    "describe_module": tool_describe_module,
    "run_module": tool_run_module,
    "sweep_parameter": tool_sweep_parameter,
    "optimize_parameter": tool_optimize_parameter,
    "compare_scenarios": tool_compare_scenarios,
    "run_uncertainty_quantification": tool_run_uncertainty_quantification,
    "chain_modules": tool_chain_modules,
}


def execute_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call and return the result."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"}
    try:
        return fn(**args)
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
#  Tool Signatures for LLM System Prompt
# ---------------------------------------------------------------------------
TOOL_DESCRIPTIONS = """
## Available Tools

1. **list_modules** — List all available physics modules and templates.
   Args: (none)

2. **describe_module** — Get detailed info about a module's capabilities.
   Args: {"module_name": "<name>"}

3. **run_module** — Execute a physics analysis.
   Args: {"module_name": "<name>", "config": {<nested config dict>}}

4. **sweep_parameter** — Sweep a parameter over a range.
   Args: {"module_name": "<name>", "config": {<base config>}, "param_path": "solver.nu", "start": 0.005, "end": 0.05, "steps": 5}
   OR: {"module_name": "<name>", "config": {<base config>}, "param_path": "solver.nu", "values": [0.005, 0.01, 0.02, 0.05]}

5. **optimize_parameter** — Find the optimal value of a parameter via golden-section search.
   Args: {"module_name": "<name>", "config": {<base config>}, "param_path": "solver.nu", "bounds": [0.005, 0.1], "objective_field": "velocity", "objective": "maximize", "n_evals": 10}

6. **compare_scenarios** — Run multiple configs side-by-side and compare.
   Args: {"module_name": "<name>", "scenarios": [{"label": "case1", "config": {...}}, {"label": "case2", "config": {...}}]}

7. **run_uncertainty_quantification** — Monte Carlo UQ analysis.
   Args: {"module_name": "<name>", "config": {<base config>}, "param_distributions": {"solver.nu": {"mean": 0.01, "std": 0.002}}, "n_samples": 10}

8. **chain_modules** — Chain multiple module executions with state passing.
   Args: {"steps": [{"module_name": "<name>", "config": {...}, "label": "step1"}, {"module_name": "<name>", "config": {...}, "coupling": {"solver.T_inlet": "temperature_max"}}]}
"""

# ---------------------------------------------------------------------------
#  LLM Integration — Replicate API
# ---------------------------------------------------------------------------
def _build_system_prompt(goal: Optional[AnalysisGoal] = None, analytical: Optional[AnalyticalEstimate] = None) -> str:
    modules = available_runtime_modules()

    module_details = ""
    for m in modules:
        mi = TRIALITY_MODULES_INFO.get(m, {})
        module_details += f"""
### {m} — {mi.get('domain', 'Physics')}
{mi.get('description', 'N/A')}
**Configuration keys:** {json.dumps(mi.get('config_keys', {}), indent=2)}
**Defaults:** {json.dumps(mi.get('defaults', {}), indent=2)}
"""

    templates = available_runtime_templates()
    templates_section = f"Available multi-physics templates: {json.dumps(templates)}" if templates else ""

    # Goal-aware context injection
    goal_context = ""
    if goal and goal.goal_type != GoalType.CHARACTERIZE:
        goal_context = f"""
## GOAL-DRIVEN MODE ACTIVE

The user's question has been analyzed and a structured goal has been extracted:
- **Goal type:** {goal.goal_type.value}
- **Metric to evaluate:** {goal.metric}
- **Operator:** {goal.operator or 'N/A'}
- **Threshold:** {goal.threshold or 'N/A'} {goal.unit or ''}
- **Search variable:** {goal.search_variable or 'N/A'}
- **Search bounds:** {list(goal.search_bounds) if goal.search_bounds else 'N/A'}
- **Module:** {goal.module_name or 'auto-detect'}

IMPORTANT: Your initial tool_calls should set up the FIRST iteration of the search.
For find_threshold goals: use sweep_parameter with the search bounds.
For maximize/minimize goals: use optimize_parameter with the search bounds.
The convergence engine will automatically iterate and adapt if the first pass doesn't find the answer.
"""
        if analytical and analytical.estimate is not None:
            goal_context += f"""
**Analytical pre-flight estimate:** {analytical.estimate:.4g} {analytical.unit or ''}
  Governing equation: {analytical.governing_equation or 'N/A'}
  Confidence: {analytical.confidence}
  Suggested bounds: {list(analytical.suggested_bounds) if analytical.suggested_bounds else 'N/A'}

Use this estimate to set intelligent initial search bounds. The numerical solver will refine from here.
"""

    return f"""You are Triality Agent, an expert physics reasoning assistant. You have two modes:

1. **Analysis mode**: When the user asks for an analysis, evaluation, sweep, optimization, or comparison, respond with a JSON plan containing tool_calls.
2. **Conversation mode**: When the user asks a follow-up question, asks you to explain results, wants physics discussion, or has a question that doesn't require running a new analysis, respond with a JSON object containing ONLY "thinking" and "summary_request" (no "tool_calls" key) where "summary_request" contains your detailed answer in markdown.

If the user's message includes CONVERSATION HISTORY, use that context to answer follow-up questions about previous results. You can reference specific numbers and observables from prior turns.

Example follow-up response (NO tool_calls):
{{"thinking": "The user is asking about the previous optimization result", "summary_request": "The optimal viscosity was found at 0.0498 m²/s because..."}}

Example analysis response (WITH tool_calls):
{{"thinking": "Need to run a fluid dynamics analysis", "plan": ["Run navier_stokes"], "tool_calls": [{{"tool": "run_module", "args": {{...}}}}]}}

## Available Physics Modules
{module_details}
{templates_section}

## Module Selection Guide

You must choose the correct module based on the PHYSICS of the problem:

- **navier_stokes** — Use for ANY fluid flow problem: cavity flows, channel flows, cooling analysis, mixing, Reynolds number studies, viscosity effects, lid-driven configurations, laminar flow.
  Key physics: velocity fields, pressure fields, vortex structures, recirculation zones.
  Key parameters: `solver.nu` (kinematic viscosity), `solver.U_lid` (lid velocity), `solver.nx`/`solver.ny` (grid resolution), `solver.Lx`/`solver.Ly` (domain size), `solver.rho` (density). Solve: `solve.t_end`, `solve.dt`, `solve.max_steps`, `solve.pressure_iters`, `solve.pressure_tol`.

- **drift_diffusion** — Use for ANY semiconductor/electronic device problem: PN junctions, diodes, doping analysis, I-V curves, junction temperature effects, carrier transport.
  Key physics: electric potential, electron/hole concentrations, current density, depletion width.
  Key parameters: `solver.temperature`, `solver.n_points`, `solver.length`, `solver.material`, `solve.applied_voltage`, `solve.max_iterations`, `solve.tolerance`, `doping.N_d_level` (donor doping), `doping.N_a_level` (acceptor doping), `doping.junction_pos`.

- **sensing** — Use for ANY radar/detection/sensor problem: radar coverage, detection probability, RCS analysis, SNR estimation, antenna design, surveillance.
  Key physics: detection probability maps, SNR fields, radar range equation.
  Key parameters: `radar.frequency_ghz`, `radar.power_w`, `radar.aperture_diameter_m`, `radar.bandwidth_mhz`, `radar.noise_figure_db`, `radar.n_pulses_integrated`, `radar.losses_db`, `target.rcs_m2`. Set `auto_grid: true` to auto-size the grid to the detection range. Grid: `grid_nx`, `grid_ny`, `grid_x_km`, `grid_y_km` (top-level keys, NOT nested).

When the user describes a physical problem WITHOUT naming a specific module, you must INFER the correct module from the physics involved. For example:
- "coolant flow in a cavity" → navier_stokes
- "characterize a diode" → drift_diffusion
- "radar detection range" → sensing
- "how viscosity affects mixing" → navier_stokes
- "junction behavior at different temperatures" → drift_diffusion

{TOOL_DESCRIPTIONS}

## Tool Selection Guide

Choose tools based on what the user wants to learn:
- **run_module** — Single analysis run. Use when the user wants to see fields/results for one configuration.
- **sweep_parameter** — Parameter study. Use when the user wants to understand how results change across a range of one parameter.
- **optimize_parameter** — Find the best value. Use when the user wants to maximize or minimize a quantity.
- **compare_scenarios** — Side-by-side comparison. Use when the user wants to compare different configurations (e.g., different temperatures, different designs).
- **run_uncertainty_quantification** — Confidence analysis. Use when the user wants to know how sensitive results are to parameter uncertainty.
- **chain_modules** — Multi-physics coupling. Use when the problem requires sequential evaluation of different physics.
- **list_modules** / **describe_module** — Information queries. Use when the user asks about capabilities.

## How to Respond

You MUST respond with valid JSON in this exact format:
{{
  "thinking": "Your reasoning: what physics is involved, which module fits, what parameters to use, and why",
  "plan": ["Step 1 description", "Step 2 description"],
  "tool_calls": [
    {{"tool": "tool_name", "args": {{...}}}}
  ],
  "summary_request": "What to highlight in the summary"
}}

{goal_context}

## Rules
1. ALWAYS use tool calls to run analyses. NEVER invent or estimate physics results.
2. ALWAYS select the module yourself based on the physics described. The user will describe their problem in engineering terms — you must map it to the correct module.
3. Use sensible defaults from the module info when the user doesn't specify parameters. Set appropriate grid sizes, time steps, and ranges for the physics involved.
4. For multi-step analysis (e.g., "run then sweep"), include multiple tool_calls in your plan.
5. If the user message is conversational (greeting, capability question), respond with empty tool_calls: {{"thinking": "conversational", "plan": [], "tool_calls": [], "summary_request": "Respond conversationally."}}
"""


def call_llm(
    system_prompt: str,
    user_message: str,
    token: str = "",
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> Optional[str]:
    """Call Replicate API and return the text output."""
    api_token = token or DEFAULT_REPLICATE_TOKEN
    if not api_token:
        return None

    headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
    full_prompt = f"{system_prompt}\n\nUser request:\n{user_message}" if system_prompt else user_message

    for attempt in range(max_retries):
        try:
            # Look up latest model version
            model_resp = http_requests.get(
                f"https://api.replicate.com/v1/models/{model}",
                headers=headers,
                timeout=20,
            )
            if model_resp.status_code != 200:
                logger.warning("Replicate model lookup failed (HTTP %s): %s", model_resp.status_code, model_resp.text[:200])
                continue
            version = model_resp.json().get("latest_version", {}).get("id", "")
            if not version:
                logger.warning("No version found for model %s", model)
                continue

            # Create prediction
            resp = http_requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json={
                    "version": version,
                    "input": {"prompt": full_prompt},
                },
                timeout=30,
            )
            if resp.status_code != 201:
                logger.warning("Replicate prediction create failed (HTTP %s): %s", resp.status_code, resp.text[:200])
                continue

            prediction = resp.json()
            poll_url = prediction.get("urls", {}).get("get", "")
            if not poll_url:
                poll_url = f"https://api.replicate.com/v1/predictions/{prediction['id']}"

            # Poll for completion
            for _ in range(120):
                time.sleep(1)
                poll_resp = http_requests.get(poll_url, headers=headers, timeout=20)
                if poll_resp.status_code != 200:
                    continue
                data = poll_resp.json()
                status = data.get("status", "")
                if status == "succeeded":
                    output = data.get("output", "")
                    if isinstance(output, list):
                        output = "".join(str(p) for p in output)
                    return output
                if status in ("failed", "canceled"):
                    logger.warning("Replicate prediction %s: %s", status, data.get("error", ""))
                    break
        except Exception as exc:
            logger.warning("call_llm attempt %d failed: %s", attempt, exc)
            continue
    logger.warning("call_llm returning None after %d retries", max_retries)
    return None


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM output, handling markdown code blocks."""
    if not text:
        return None
    # Try to find JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to find first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
#  Heuristic Fallback Planner — when LLM fails or no token
# ---------------------------------------------------------------------------
def _heuristic_plan(prompt: str, scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """Build a plan from keyword matching when LLM is unavailable."""
    text = (prompt or "").lower().strip()

    # Check scenario first
    if scenario_id:
        for s in SCENARIOS:
            if s["id"] == scenario_id:
                text = s["prompt"].lower()
                break

    # Detect module
    module_name = "navier_stokes"  # default
    module_keywords = {
        "drift_diffusion": ["diode", "pn junction", "semiconductor", "drift", "diffusion", "doping", "junction", "silicon", "voltage bias"],
        "sensing": ["radar", "sensing", "detection", "coverage", "rcs", "antenna", "snr", "sonar", "lidar"],
        "electrostatics": ["electrostatic", "electric field", "laplace", "poisson", "permittivity", "dielectric", "charge distribution"],
        "aero_loads": ["aero load", "hypersonic", "reentry", "heat flux", "newtonian flow", "nose radius", "thermal protection", "tps"],
        "uav_aerodynamics": ["uav", "wing", "vortex lattice", "vlm", "wingspan", "airfoil", "lift coefficient", "induced drag", "chord"],
        "spacecraft_thermal": ["spacecraft", "satellite", "cubesat", "orbit", "eclipse", "heater", "radiative", "deep space"],
        "automotive_thermal": ["igbt", "inverter", "busbar", "power electronics", "automotive thermal", "heatsink", "junction temperature"],
        "battery_thermal": ["battery", "cell thermal", "soc", "c-rate", "runaway", "pack thermal", "discharge current", "nmc", "lfp"],
        "structural_analysis": ["beam", "cantilever", "buckling", "stress", "structural analysis", "bending", "von mises", "simply supported"],
        "structural_dynamics": ["vibration", "modal", "newmark", "shock response", "srs", "resonance", "harmonic", "structural dynamics"],
        "flight_mechanics": ["6-dof", "6 dof", "attitude", "detumble", "quaternion", "angular rate", "rigid body", "inertia tensor"],
        "coupled_physics": ["neutronics-thermal", "coupled physics", "point kinetics", "reactivity insertion", "rod withdrawal", "doppler feedback"],
        "neutronics": ["neutron", "k-eff", "k_eff", "criticality", "diffusion solver", "fission", "reactor core", "fuel enrichment", "reflector"],
        "geospatial": ["warehouse", "facility", "coverage", "travel time", "geospatial", "logistics", "delivery", "population"],
        "field_aware_routing": ["pcb", "trace routing", "emi", "field-aware", "routing", "clearance", "crosstalk"],
        "navier_stokes": ["cavity", "navier", "stokes", "fluid", "flow", "laminar", "lid", "reynolds", "viscosity"],
    }
    for mod, keywords in module_keywords.items():
        if any(kw in text for kw in keywords):
            module_name = mod
            break

    # Detect intent
    intent = "run_module"
    if any(w in text for w in ["sweep", "vary", "range", "study", "steps", "from", "to"]):
        intent = "sweep_parameter"
    if any(w in text for w in ["optimize", "optimal", "best", "maximize", "minimize"]):
        intent = "optimize_parameter"
    if any(w in text for w in ["compare", "versus", " vs ", "different", "side by side"]):
        intent = "compare_scenarios"
    if any(w in text for w in ["uncertainty", "monte carlo", "confidence", "uq"]):
        intent = "run_uncertainty_quantification"

    # Build config from extracted parameters
    config = copy.deepcopy(TRIALITY_MODULES_INFO.get(module_name, {}).get("defaults", {}))

    # Extract numbers from prompt
    def _find(pattern: str) -> Optional[float]:
        m = re.search(pattern, prompt or "", re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    if module_name == "navier_stokes":
        grid = _find(r"(\d+)\s*x\s*\d+\s*grid")
        if grid:
            config["solver"]["nx"] = int(grid)
            config["solver"]["ny"] = int(grid)
        nu = _find(r"viscosity\s+(?:is\s+|of\s+)?([0-9.]+(?:e[+-]?\d+)?)")
        if nu:
            config["solver"]["nu"] = nu
        u_lid = _find(r"(?:lid\s+velocity|drives?\s+(?:air|flow|water|cold air)\s+(?:upward\s+)?at)\s+([0-9.]+(?:e[+-]?\d+)?)")
        if u_lid:
            config["solver"]["U_lid"] = u_lid
        # Also try "at X m/s" pattern for velocity
        if not u_lid:
            u_lid = _find(r"at\s+([0-9.]+)\s*m/s")
            if u_lid:
                config["solver"]["U_lid"] = u_lid
        re_num = _find(r"re(?:ynolds)?\s*(?:number)?\s*~?\s*(\d+)")
        if re_num and re_num > 0:
            u = config["solver"].get("U_lid", 0.1)
            L = config["solver"].get("Lx", 1.0)
            config["solver"]["nu"] = u * L / re_num

    elif module_name == "drift_diffusion":
        temp = _find(r"(\d+)\s*k(?:\b|elvin)")
        if temp:
            config["solver"]["temperature"] = temp
        voltage = _find(r"([0-9.]+)\s*v(?:olt)?")
        if voltage:
            config["solve"]["applied_voltage"] = voltage
        nd = _find(r"donor\s+(?:doping\s+)?([0-9.]+(?:e[+-]?\d+)?)")
        if nd:
            config.setdefault("doping", {})["N_d_level"] = nd
        na = _find(r"acceptor\s+(?:doping\s+)?([0-9.]+(?:e[+-]?\d+)?)")
        if na:
            config.setdefault("doping", {})["N_a_level"] = na

    elif module_name == "sensing":
        config.setdefault("radar", {})
        config.setdefault("target", {})
        config["auto_grid"] = True
        freq = _find(r"frequency_ghz\s*=\s*([0-9.e+-]+)") or _find(r"([0-9.]+)\s*ghz")
        if freq:
            config["radar"]["frequency_ghz"] = freq
        power = _find(r"power_w\s*=\s*([0-9.e+-]+)") or _find(r"([0-9.e+]+)\s*(?:w|watt)\s+(?:transmit|peak|power)")
        if power:
            config["radar"]["power_w"] = power
        aperture = _find(r"aperture_diameter_m\s*=\s*([0-9.e+-]+)") or _find(r"([0-9.]+)\s*m\s+(?:aperture|antenna|diameter)")
        if aperture:
            config["radar"]["aperture_diameter_m"] = aperture
        bw = _find(r"bandwidth_mhz\s*=\s*([0-9.e+-]+)") or _find(r"([0-9.]+)\s*mhz\s+bandwidth")
        if bw:
            config["radar"]["bandwidth_mhz"] = bw
        rcs = _find(r"rcs_m2\s*=\s*([0-9.e+-]+)") or _find(r"rcs\s+(?:of\s+)?([0-9.e+-]+)")
        if rcs:
            config["target"]["rcs_m2"] = rcs

    elif module_name == "electrostatics":
        res = _find(r"(\d+)\s*(?:-?\s*point|resolution)")
        if res:
            config.setdefault("solver", {})["resolution"] = int(res)
        bv = _find(r"(\d+(?:\.\d+)?)\s*(?:v|volt)")
        if bv:
            config.setdefault("solve", {})["boundary_value"] = bv

    elif module_name == "aero_loads":
        length = _find(r"(\d+(?:\.\d+)?)\s*m\s+(?:long|length|body)")
        if length:
            config.setdefault("solver", {})["body_length_m"] = length
        vel = _find(r"(?:velocity|speed)\s+(?:of\s+)?(\d+(?:\.\d+)?)")
        if vel:
            config.setdefault("solve", {})["velocity"] = vel
        alpha = _find(r"(\d+(?:\.\d+)?)\s*(?:deg|degrees?)\s*(?:angle|aoa|attack)")
        if alpha:
            config.setdefault("solve", {})["alpha_deg"] = alpha

    elif module_name == "uav_aerodynamics":
        span = _find(r"(\d+(?:\.\d+)?)\s*m\s+span")
        if span:
            config.setdefault("solver", {})["span"] = span
        aoa = _find(r"(\d+(?:\.\d+)?)\s*(?:deg|degrees?)")
        if aoa:
            config.setdefault("solver", {})["alpha_deg"] = aoa
        vinf = _find(r"(\d+(?:\.\d+)?)\s*m/s")
        if vinf:
            config.setdefault("solver", {})["V_inf"] = vinf

    elif module_name == "spacecraft_thermal":
        n_nodes = _find(r"(\d+)\s*(?:-?\s*node)")
        if n_nodes:
            config.setdefault("solver", {})["n_nodes"] = int(n_nodes)
        pwr = _find(r"(\d+(?:\.\d+)?)\s*w\s+(?:electronics|dissipation|internal|power)")
        if pwr:
            config.setdefault("solver", {})["internal_power_W"] = pwr
        tend = _find(r"(\d+(?:\.\d+)?)\s*s(?:econds?)?\s+(?:orbit|analysis|simulation)")
        if tend:
            config.setdefault("solve", {})["t_end"] = tend

    elif module_name == "automotive_thermal":
        cur = _find(r"(\d+(?:\.\d+)?)\s*a(?:mps?)?")
        if cur:
            config.setdefault("solver", {})["current_A"] = cur
        h = _find(r"h\s*=\s*(\d+(?:\.\d+)?)")
        if h:
            config.setdefault("solver", {})["h_convection"] = h

    elif module_name == "battery_thermal":
        ncells = _find(r"(\d+)\s*(?:-?\s*cell)")
        if ncells:
            config.setdefault("solver", {})["n_cells"] = int(ncells)
        current = _find(r"(\d+(?:\.\d+)?)\s*a\s+(?:discharge|current)")
        if current:
            config.setdefault("solve", {})["discharge_current_A"] = current

    elif module_name == "structural_analysis":
        force = _find(r"(\d+(?:\.\d+)?)\s*n\s+(?:tip|force|load)")
        if force:
            config.setdefault("solve", {})["tip_force_N"] = -abs(force)
        length = _find(r"(\d+(?:\.\d+)?)\s*m\s+(?:long|length|beam)")
        if length:
            config.setdefault("solver", {})["length"] = length

    elif module_name == "structural_dynamics":
        freq = _find(r"(\d+(?:\.\d+)?)\s*hz")
        if freq:
            config.setdefault("solver", {})["force_frequency"] = freq
        damp = _find(r"(\d+(?:\.\d+)?)\s*%?\s*damp")
        if damp:
            config.setdefault("solver", {})["damping_ratio"] = damp if damp < 1 else damp / 100.0

    elif module_name == "flight_mechanics":
        mass = _find(r"(\d+(?:\.\d+)?)\s*kg")
        if mass:
            config.setdefault("solver", {})["mass"] = mass
        tfinal = _find(r"(\d+(?:\.\d+)?)\s*s(?:econds?)?")
        if tfinal:
            config.setdefault("solve", {})["t_final"] = tfinal

    elif module_name == "coupled_physics":
        rho = _find(r"(\d+(?:\.\d+)?)\s*pcm")
        if rho:
            config.setdefault("solve", {})["reactivity_insertion_pcm"] = rho
        tend = _find(r"(\d+(?:\.\d+)?)\s*s(?:econds?)?")
        if tend:
            config.setdefault("solve", {})["t_end"] = tend

    elif module_name == "neutronics":
        core_len = _find(r"(\d+(?:\.\d+)?)\s*cm\s+core")
        if core_len:
            config.setdefault("solver", {})["core_length"] = core_len
        npts = _find(r"(\d+)\s+(?:spatial|mesh|point)")
        if npts:
            config.setdefault("solver", {})["n_spatial"] = int(npts)

    elif module_name == "geospatial":
        nfac = _find(r"(\d+)\s+(?:warehouse|facilit)")
        if nfac:
            config.setdefault("solver", {})["max_facilities"] = int(nfac)
        tlim = _find(r"(\d+(?:\.\d+)?)\s*(?:hour|h)\s+(?:delivery|travel|limit)")
        if tlim:
            config.setdefault("solver", {})["time_limit_hours"] = tlim

    elif module_name == "field_aware_routing":
        nx = _find(r"(\d+)\s*x\s*\d+\s*(?:grid|resolution)")
        if nx:
            config.setdefault("solver", {})["nx"] = int(nx)
            config.setdefault("solver", {})["ny"] = int(nx)
        bv = _find(r"(\d+(?:\.\d+)?)\s*v\s+(?:boundary|bus|left)")
        if bv:
            config.setdefault("solver", {})["bc_left_V"] = bv

    # Build tool calls based on intent
    tool_calls = []

    if intent == "run_module":
        tool_calls.append({"tool": "run_module", "args": {"module_name": module_name, "config": config}})

    elif intent == "sweep_parameter":
        # Detect sweep parameter and range
        param_path, start, end, steps = _detect_sweep_params(text, module_name, config)
        tool_calls.append({
            "tool": "run_module",
            "args": {"module_name": module_name, "config": config},
        })
        tool_calls.append({
            "tool": "sweep_parameter",
            "args": {
                "module_name": module_name,
                "config": config,
                "param_path": param_path,
                "start": start,
                "end": end,
                "steps": steps,
            },
        })

    elif intent == "optimize_parameter":
        param_path, lo, hi, _ = _detect_sweep_params(text, module_name, config)
        _objective_fields = {
            "navier_stokes": "velocity",
            "drift_diffusion": "electrostatic_potential",
            "sensing": "detection_probability",
            "electrostatics": "electric_field_magnitude",
            "aero_loads": "heat_flux",
            "uav_aerodynamics": "lift_coefficient",
            "spacecraft_thermal": "temperature",
            "automotive_thermal": "temperature",
            "battery_thermal": "temperature",
            "structural_analysis": "displacement",
            "structural_dynamics": "displacement",
            "flight_mechanics": "euler_angles",
            "coupled_physics": "temperature_field",
            "neutronics": "power_density",
            "geospatial": "coverage_fraction",
            "field_aware_routing": "electric_potential",
        }
        obj_field = _objective_fields.get(module_name, "velocity")
        tool_calls.append({
            "tool": "optimize_parameter",
            "args": {
                "module_name": module_name,
                "config": config,
                "param_path": param_path,
                "bounds": [lo, hi],
                "objective_field": obj_field,
                "objective": "maximize",
                "n_evals": 10,
            },
        })

    elif intent == "compare_scenarios":
        scenarios = _build_comparison_scenarios(text, module_name, config)
        tool_calls.append({
            "tool": "compare_scenarios",
            "args": {"module_name": module_name, "scenarios": scenarios},
        })

    elif intent == "run_uncertainty_quantification":
        param_path, _, _, _ = _detect_sweep_params(text, module_name, config)
        nominal = _get_nested(config, param_path, 0.01)
        tool_calls.append({
            "tool": "run_uncertainty_quantification",
            "args": {
                "module_name": module_name,
                "config": config,
                "param_distributions": {param_path: {"mean": nominal, "std": nominal * 0.1}},
                "n_samples": 10,
            },
        })

    thinking = f"Detected module={module_name}, intent={intent} from keywords in prompt."
    plan = [f"Execute {intent} on {module_name}"]

    return {
        "thinking": thinking,
        "plan": plan,
        "tool_calls": tool_calls,
        "summary_request": f"Summarize the {intent} results for {module_name}.",
    }


def _detect_sweep_params(text: str, module_name: str, config: Dict) -> Tuple[str, float, float, int]:
    """Detect which parameter to sweep and its range from prompt text."""
    # Try to extract explicit range
    range_match = re.search(r"from\s+([0-9.e+-]+)\s+to\s+([0-9.e+-]+)", text)
    steps_match = re.search(r"(\d+)\s+steps", text)
    steps = int(steps_match.group(1)) if steps_match else 5

    if module_name == "navier_stokes":
        if any(w in text for w in ["viscosity", "nu", "reynolds"]):
            param = "solver.nu"
            if range_match:
                return param, float(range_match.group(1)), float(range_match.group(2)), steps
            return param, 1e-5, 1e-2, steps
        if any(w in text for w in ["velocity", "speed", "lid"]):
            param = "solver.U_lid"
            if range_match:
                return param, float(range_match.group(1)), float(range_match.group(2)), steps
            return param, 0.5, 5.0, steps
        param = "solver.U_lid"
        if range_match:
            return param, float(range_match.group(1)), float(range_match.group(2)), steps
        return param, 0.5, 5.0, steps

    elif module_name == "drift_diffusion":
        if any(w in text for w in ["voltage", "bias", "v "]):
            param = "solve.applied_voltage"
            if range_match:
                return param, float(range_match.group(1)), float(range_match.group(2)), steps
            return param, 0.0, 0.7, steps
        if any(w in text for w in ["temperature", "temp"]):
            param = "solver.temperature"
            if range_match:
                return param, float(range_match.group(1)), float(range_match.group(2)), steps
            return param, 250, 400, steps
        return "solve.applied_voltage", 0.0, 0.7, steps

    elif module_name == "sensing":
        if any(w in text for w in ["rcs", "cross section"]):
            param = "target.rcs_m2"
            if range_match:
                return param, float(range_match.group(1)), float(range_match.group(2)), steps
            return param, 0.01, 1.0, steps
        if any(w in text for w in ["power"]):
            param = "radar.power_w"
            if range_match:
                return param, float(range_match.group(1)), float(range_match.group(2)), steps
            return param, 1000, 100000, steps
        return "target.rcs_m2", 0.01, 1.0, steps

    # --- New module sweep defaults ---
    _default_sweeps = {
        "electrostatics": ("solve.boundary_value", 100.0, 2000.0),
        "aero_loads": ("solve.alpha_deg", 0.0, 15.0),
        "uav_aerodynamics": ("solver.alpha_deg", 0.0, 12.0),
        "spacecraft_thermal": ("solver.internal_power_W", 10.0, 100.0),
        "automotive_thermal": ("solver.current_A", 50.0, 400.0),
        "battery_thermal": ("solve.discharge_current_A", 20.0, 200.0),
        "structural_analysis": ("solve.tip_force_N", -1000.0, -15000.0),
        "structural_dynamics": ("solver.force_frequency", 1.0, 50.0),
        "flight_mechanics": ("solve.omega_x", 0.01, 0.2),
        "coupled_physics": ("solve.reactivity_insertion_pcm", 50.0, 500.0),
        "neutronics": ("solver.reflector_thickness", 10.0, 60.0),
        "geospatial": ("solver.max_facilities", 1.0, 6.0),
        "field_aware_routing": ("solver.bc_left_V", 100.0, 2000.0),
    }
    if module_name in _default_sweeps:
        param, lo, hi = _default_sweeps[module_name]
        if range_match:
            return param, float(range_match.group(1)), float(range_match.group(2)), steps
        return param, lo, hi, steps

    return "solver.nx", 8, 32, steps


def _build_comparison_scenarios(text: str, module_name: str, config: Dict) -> List[Dict]:
    """Build comparison scenarios from prompt text."""
    # Extract explicit values (e.g., "250K, 300K, 350K, 400K")
    number_matches = re.findall(r"(\d+(?:\.\d+)?)\s*k(?:elvin)?", text, re.IGNORECASE)

    if module_name == "drift_diffusion":
        if number_matches and len(number_matches) >= 2:
            scenarios = []
            for t in number_matches:
                cfg = copy.deepcopy(config)
                cfg["solver"]["temperature"] = float(t)
                scenarios.append({"label": f"{t}K", "config": cfg})
            return scenarios
        # Default temperature comparison
        return [
            {"label": "250K", "config": _deep_merge(config, {"solver": {"temperature": 250}})},
            {"label": "300K", "config": _deep_merge(config, {"solver": {"temperature": 300}})},
            {"label": "350K", "config": _deep_merge(config, {"solver": {"temperature": 350}})},
        ]

    elif module_name == "sensing":
        rcs_matches = re.findall(r"(\d+(?:\.\d+)?)\s*m\^?2", text)
        if rcs_matches and len(rcs_matches) >= 2:
            return [{"label": f"RCS={r}m²", "config": _deep_merge(config, {"target": {"rcs_m2": float(r)}})} for r in rcs_matches]
        return [
            {"label": "RCS=0.01m²", "config": _deep_merge(config, {"target": {"rcs_m2": 0.01}})},
            {"label": "RCS=0.1m²", "config": _deep_merge(config, {"target": {"rcs_m2": 0.1}})},
            {"label": "RCS=1.0m²", "config": _deep_merge(config, {"target": {"rcs_m2": 1.0}})},
        ]

    elif module_name == "navier_stokes":
        return [
            {"label": "Re=10", "config": _deep_merge(config, {"solver": {"nu": 0.01}})},
            {"label": "Re=50", "config": _deep_merge(config, {"solver": {"nu": 0.002}})},
            {"label": "Re=100", "config": _deep_merge(config, {"solver": {"nu": 0.001}})},
        ]

    elif module_name == "electrostatics":
        return [
            {"label": "500V", "config": _deep_merge(config, {"solve": {"boundary_value": 500.0}})},
            {"label": "1000V", "config": _deep_merge(config, {"solve": {"boundary_value": 1000.0}})},
            {"label": "2000V", "config": _deep_merge(config, {"solve": {"boundary_value": 2000.0}})},
        ]

    elif module_name == "aero_loads":
        return [
            {"label": "Mach 3", "config": _deep_merge(config, {"solve": {"velocity": 1000.0}})},
            {"label": "Mach 5", "config": _deep_merge(config, {"solve": {"velocity": 2000.0}})},
            {"label": "Mach 7", "config": _deep_merge(config, {"solve": {"velocity": 2500.0}})},
        ]

    elif module_name == "uav_aerodynamics":
        return [
            {"label": "Span 8m", "config": _deep_merge(config, {"solver": {"span": 8.0}})},
            {"label": "Span 12m", "config": _deep_merge(config, {"solver": {"span": 12.0}})},
            {"label": "Span 16m", "config": _deep_merge(config, {"solver": {"span": 16.0}})},
        ]

    elif module_name == "spacecraft_thermal":
        return [
            {"label": "20W", "config": _deep_merge(config, {"solver": {"internal_power_W": 20.0}})},
            {"label": "50W", "config": _deep_merge(config, {"solver": {"internal_power_W": 50.0}})},
            {"label": "100W", "config": _deep_merge(config, {"solver": {"internal_power_W": 100.0}})},
        ]

    elif module_name == "automotive_thermal":
        return [
            {"label": "100A", "config": _deep_merge(config, {"solver": {"current_A": 100.0}})},
            {"label": "200A", "config": _deep_merge(config, {"solver": {"current_A": 200.0}})},
            {"label": "400A", "config": _deep_merge(config, {"solver": {"current_A": 400.0}})},
        ]

    elif module_name == "battery_thermal":
        return [
            {"label": "50A (1C)", "config": _deep_merge(config, {"solve": {"discharge_current_A": 50.0}})},
            {"label": "100A (2C)", "config": _deep_merge(config, {"solve": {"discharge_current_A": 100.0}})},
            {"label": "200A (4C)", "config": _deep_merge(config, {"solve": {"discharge_current_A": 200.0}})},
        ]

    elif module_name == "structural_analysis":
        return [
            {"label": "5kN", "config": _deep_merge(config, {"solve": {"tip_force_N": -5000.0}})},
            {"label": "10kN", "config": _deep_merge(config, {"solve": {"tip_force_N": -10000.0}})},
            {"label": "15kN", "config": _deep_merge(config, {"solve": {"tip_force_N": -15000.0}})},
        ]

    elif module_name == "structural_dynamics":
        return [
            {"label": "1% damping", "config": _deep_merge(config, {"solver": {"damping_ratio": 0.01}})},
            {"label": "2% damping", "config": _deep_merge(config, {"solver": {"damping_ratio": 0.02}})},
            {"label": "5% damping", "config": _deep_merge(config, {"solver": {"damping_ratio": 0.05}})},
        ]

    elif module_name == "flight_mechanics":
        return [
            {"label": "200kg", "config": _deep_merge(config, {"solver": {"mass": 200.0}})},
            {"label": "500kg", "config": _deep_merge(config, {"solver": {"mass": 500.0}})},
            {"label": "1000kg", "config": _deep_merge(config, {"solver": {"mass": 1000.0}})},
        ]

    elif module_name == "coupled_physics":
        return [
            {"label": "50 pcm", "config": _deep_merge(config, {"solve": {"reactivity_insertion_pcm": 50.0}})},
            {"label": "100 pcm", "config": _deep_merge(config, {"solve": {"reactivity_insertion_pcm": 100.0}})},
            {"label": "300 pcm", "config": _deep_merge(config, {"solve": {"reactivity_insertion_pcm": 300.0}})},
        ]

    elif module_name == "neutronics":
        return [
            {"label": "3% UO2", "config": _deep_merge(config, {"solver": {"fuel_material": "FUEL_UO2_3PCT"}})},
            {"label": "5% UO2", "config": _deep_merge(config, {"solver": {"fuel_material": "FUEL_UO2_5PCT"}})},
        ]

    elif module_name == "geospatial":
        return [
            {"label": "2 facilities", "config": _deep_merge(config, {"solver": {"max_facilities": 2}})},
            {"label": "3 facilities", "config": _deep_merge(config, {"solver": {"max_facilities": 3}})},
            {"label": "5 facilities", "config": _deep_merge(config, {"solver": {"max_facilities": 5}})},
        ]

    elif module_name == "field_aware_routing":
        return [
            {"label": "500V", "config": _deep_merge(config, {"solver": {"bc_left_V": 500.0}})},
            {"label": "1000V", "config": _deep_merge(config, {"solver": {"bc_left_V": 1000.0}})},
            {"label": "2000V", "config": _deep_merge(config, {"solver": {"bc_left_V": 2000.0}})},
        ]

    return [{"label": "default", "config": config}]


# ---------------------------------------------------------------------------
#  Programmatic Insights — deterministic analysis of results
# ---------------------------------------------------------------------------
def _validate_against_benchmarks(module: str, fields: Dict) -> List[str]:
    """Compare field stats against reference benchmarks and return validation lines."""
    benchmarks = REFERENCE_BENCHMARKS.get(module, {})
    if not benchmarks:
        return []
    lines = ["", "**Validation against references:**"]
    for bench_key, (lo, hi, desc) in benchmarks.items():
        if lo is None or hi is None:
            continue
        # Try to match bench_key to a field stat
        actual = None
        if bench_key == "velocity_x_max":
            actual = fields.get("velocity_x", {}).get("max")
        elif bench_key == "velocity_x_min":
            actual = fields.get("velocity_x", {}).get("min")
        elif bench_key == "built_in_potential_v":
            pot = fields.get("electrostatic_potential", {})
            if pot:
                actual = abs(pot.get("max", 0) - pot.get("min", 0))
        if actual is None:
            continue
        in_range = lo <= actual <= hi
        grade = "A" if in_range else ("B" if abs(actual - (lo + hi) / 2) < abs(hi - lo) else "C")
        status = "PASS" if in_range else "MARGIN"
        lines.append(f"  - {bench_key} = {actual:.4g} | ref: [{lo:.3g}, {hi:.3g}] | {desc} | {status} [{grade}]")
    return lines if len(lines) > 2 else []


def _get_scenario_context(scenario_id: Optional[str]) -> Optional[Dict]:
    """Look up a scenario by ID and return its context if it's an industry scenario."""
    if not scenario_id:
        return None
    for s in SCENARIOS:
        if s["id"] == scenario_id and s.get("category") == "industry":
            return s
    return None


def _build_industry_framing(scenario: Optional[Dict]) -> str:
    """Build industry-specific framing instructions for the LLM summary."""
    if not scenario:
        return ""

    title = scenario.get("business_title", scenario.get("title", ""))
    problem = scenario.get("industry_problem", "")
    decision = scenario.get("decision_focus", "")

    return f"""
## Industry Context — Frame Results as Decisions

This analysis addresses a real industry problem. You MUST translate every result into
language that helps an engineer or manager make a decision. Do NOT use academic phrasing.

**Problem:** {problem}
**Decision needed:** {decision}

Translation rules:
1. **Grid positions → physical dimensions.** If the domain is L×H, then row j of N rows
   corresponds to height (j/N)*H. ALWAYS convert grid indices to metres (or the appropriate unit)
   and state which physical location they represent (e.g., "row 18 of 24 ≈ 1.5m above floor").
2. **Field values → compliance thresholds.** Compare every key result against the threshold
   implied by the decision focus. State explicitly whether the threshold is MET or NOT MET,
   and by how much margin.
3. **Sweep/optimize results → operating recommendations.** Don't just say "increases
   monotonically." Say at which parameter value the threshold is first met, and what
   operating range provides adequate margin (≥20% above/below the limit).
4. **Dead zones / recirculation → physical risk.** Map any stagnation or recirculation
   regions to their physical location and state the operational consequence (e.g.,
   "dead zone in the lower-left quadrant corresponds to the area behind the equipment
   where methane could accumulate").
5. **Use the decision question as your conclusion.** Your final paragraph must directly
   answer the decision_focus question with a clear YES/NO/CONDITIONAL recommendation,
   supported by the numbers from the analysis.
6. **Replacement section: instead of generic "Design Recommendations," use
   "Decision & Recommended Action"** — state the go/no-go and the specific action
   (e.g., "increase exhaust fan speed to ≥0.56 m/s" or "add blanking panels at rows 3–5").
"""


def _build_programmatic_insights(tool_results: List[Dict], scenario: Optional[Dict] = None) -> str:
    """Build deterministic insights from tool execution results."""
    insights = []

    # If industry scenario, prepend physical mapping context
    if scenario:
        insights.append(f"**Industry scenario:** {scenario.get('business_title', '')}")
        insights.append(f"**Decision needed:** {scenario.get('decision_focus', '')}")
        insights.append("")

    is_industry = scenario is not None

    for i, result in enumerate(tool_results):
        tool_name = result.get("_tool_name", "unknown")

        if tool_name == "run_module":
            module = result.get("module_name", "")
            if result.get("success"):
                # --- Observable-based insights (domain-specific) ---
                observables = result.get("observables", [])
                if observables:
                    insights.append(f"**Key observables ({module}):**")
                    for obs in observables:
                        v = obs.get("value")
                        u = obs.get("unit", "")
                        desc = obs.get("description", "")
                        rel = obs.get("relevance", "")
                        margin = obs.get("margin")

                        if isinstance(v, bool):
                            v_str = "YES" if v else "NO"
                        elif isinstance(v, float):
                            v_str = f"{v:.4g}"
                        elif isinstance(v, (list, dict)):
                            v_str = str(v)[:80]
                        else:
                            v_str = str(v)

                        line = f"- **{obs['name']}**: {v_str} {u}"
                        if desc:
                            line += f" — {desc}"
                        if margin is not None:
                            if isinstance(margin, (int, float)):
                                if margin < 0:
                                    line += f" ⚠️ MARGIN VIOLATED ({margin:+.3g})"
                                else:
                                    line += f" ✓ margin={margin:+.3g}"
                        if rel:
                            line += f" [{rel}]"
                        insights.append(line)
                    insights.append("")

                # Raw field stats as fallback
                fields = result.get("fields_stats", {})
                if not observables:
                    for fname, stats in fields.items():
                        insights.append(f"- {fname}: range [{stats['min']:.4g}, {stats['max']:.4g}], mean={stats['mean']:.4g}, shape={stats.get('shape', 'N/A')}")

                # Module-specific causal interpretation
                if module == "navier_stokes":
                    vx = fields.get("velocity_x", {})
                    vy = fields.get("velocity_y", {})
                    pressure = fields.get("pressure", {})
                    # Extract grid shape for physical mapping
                    grid_shape = vx.get("shape") or vy.get("shape") or []
                    ny = grid_shape[0] if len(grid_shape) >= 2 else grid_shape[0] if len(grid_shape) == 1 else 0
                    nx = grid_shape[1] if len(grid_shape) >= 2 else 0

                    if vx and vy:
                        max_vel = max(abs(vx.get("max", 0)), abs(vy.get("max", 0)))
                        insights.append(f"- Peak velocity magnitude: {max_vel:.4g} m/s")
                        # Recirculation detection
                        if vx.get("min", 0) < 0:
                            insights.append(f"- Recirculation detected: velocity_x sign reversal (min={vx['min']:.4g} m/s) indicates vortex formation")
                            if is_industry and ny:
                                insights.append(f"  → Grid has {ny} rows; map row index to physical height as (row/{ny}) × domain_height to locate the recirculation zone.")
                                insights.append(f"  → IMPORTANT: Translate this to a physical location (e.g., 'recirculation extends down to row K of {ny} ≈ X.Xm above floor') and state whether it violates the operating threshold.")
                        else:
                            insights.append("- No recirculation detected: flow is predominantly unidirectional")
                            if is_industry:
                                insights.append("  → Good: uniform flow with no stagnation dead zones detected.")
                    if pressure:
                        p_range = pressure.get("max", 0) - pressure.get("min", 0)
                        insights.append(f"- Pressure field spans {p_range:.4g} Pa")
                        if is_industry:
                            insights.append(f"  → Pressure differential of {p_range:.4g} Pa drives the flow. Compare against pump/fan capacity and allowable pressure drop limits.")
                    insights.append("")

                    if is_industry:
                        insights.append("**Physical mapping:** The analysis domain maps to the real geometry. Convert grid coordinates to physical dimensions (metres) and compare against design thresholds. Identify WHERE problems occur, not just that they exist.")
                        insights.append("**Operational interpretation:** Recirculation zones = potential hot spots, dead zones, or accumulation areas. State the physical consequence for this specific application.")
                    else:
                        insights.append("**Physics interpretation:** The lid-driven cavity develops a primary vortex whose strength depends on the Reynolds number (Re = U*L/nu). Higher Re produces stronger recirculation and secondary corner vortices.")
                        insights.append("**Design feedback:** Increase grid resolution or reduce viscosity to raise Re and resolve finer vortex structure.")
                    insights.append("**Model limits:** 2D incompressible; no turbulence model; lid-driven geometry only.")
                    # Validation
                    insights.extend(_validate_against_benchmarks(module, fields))

                elif module == "drift_diffusion":
                    pot = fields.get("electrostatic_potential", {})
                    n_density = fields.get("electron_density", {})
                    if pot:
                        built_in = abs(pot.get("max", 0) - pot.get("min", 0))
                        insights.append(f"- Built-in potential drop: {built_in:.4g} V")
                        if is_industry:
                            insights.append(f"  → This voltage represents the threshold above which rapid change occurs. Compare against the operational limit for this application.")
                        else:
                            insights.append(f"- The built-in potential arises from the charge equilibrium at the PN junction; higher doping asymmetry increases this voltage.")
                    if n_density:
                        insights.append(f"- Carrier density spans {n_density.get('min', 0):.4g} to {n_density.get('max', 0):.4g}")
                        if is_industry:
                            insights.append(f"  → The ratio of max to min density ({n_density.get('max', 0) / max(n_density.get('min', 1e-30), 1e-30):.2g}×) indicates how sharply the system transitions. A large ratio means narrow operating margin.")
                        else:
                            insights.append("  -- the depletion region is where density drops to near-intrinsic levels.")
                    insights.append("")

                    if is_industry:
                        insights.append("**Operational interpretation:** The voltage at which current spikes represents the onset threshold (thermal runaway, breakdown, etc.). State the margin between operating point and this threshold as a percentage.")
                        insights.append("**Design feedback:** Identify which parameter (temperature, load, etc.) most rapidly erodes the safety margin, and recommend the maximum safe operating point.")
                    else:
                        insights.append("**Physics interpretation:** Forward bias reduces the depletion width and lowers the potential barrier, enabling exponential current increase (Shockley equation). The I-V knee occurs near the built-in potential.")
                        insights.append("**Design feedback:** Doping asymmetry (N_d vs N_a) controls the built-in potential and breakdown characteristics. Verify turn-on voltage meets circuit requirements.")
                    insights.append("**Model limits:** 1D drift-diffusion; no recombination models beyond SRH; no high-injection effects.")
                    insights.extend(_validate_against_benchmarks(module, fields))

                elif module == "sensing":
                    det_prob = fields.get("detection_probability", {})
                    snr = fields.get("snr", fields.get("signal_to_noise", {}))
                    if det_prob:
                        insights.append(f"- Detection probability: peak={det_prob.get('max', 0):.4g}, coverage fraction at >0.5 depends on range and RCS")
                        if is_industry:
                            insights.append("  → Map the detection probability contour to physical distance from the source. The 0.5 probability contour defines the effective boundary of the zone (exclusion zone, coverage zone, etc.).")
                    insights.append("")

                    if is_industry:
                        insights.append("**Physical mapping:** Convert the detection/intensity field to a distance map. State the boundary distance where the threshold is met/exceeded, and compare against the required standoff or coverage distance.")
                        insights.append("**Design feedback:** State whether the current configuration meets the required coverage/exclusion zone, and by what margin. Recommend specific parameter changes if it does not.")
                    else:
                        insights.append("**Physics interpretation:** Detection range follows the radar range equation (R^4 dependence on power, aperture, and RCS). Smaller targets require exponentially more power or larger apertures to detect at the same range.")
                        insights.append("**Design feedback:** To extend detection range by 2x, you need 16x the power or 4x the antenna area. Consider whether increasing RCS sensitivity or adding sensors is more cost-effective.")
                    insights.append("**Model limits:** Free-space propagation; no multipath, clutter, or atmospheric absorption.")

                # Suggested follow-ups for single runs
                insights.append("")
                if is_industry:
                    insights.append("**Suggested follow-ups:** Sweep the critical parameter to find the exact threshold where compliance is met/lost, or run UQ to quantify how much manufacturing/operating variability erodes the safety margin.")
                else:
                    insights.append("**Suggested follow-ups:** Run a parameter sweep to assess sensitivity, or run UQ with +/-10% variation on key parameters to quantify uncertainty.")

            else:
                insights.append(f"- Module {module} execution FAILED: {result.get('error', 'unknown')}")

        elif tool_name == "sweep_parameter":
            sweep_results = result.get("results", [])
            param_values = result.get("values", [])
            if sweep_results:
                n_points = len(sweep_results)
                insights.append(f"- Swept {result.get('param_path')} over {n_points} values")
                if n_points <= 10:
                    insights.append(f"  - Note: {n_points}-point sweep provides limited resolution. Do NOT claim precise optimal ranges from sparse data. Use qualitative trend descriptions.")

                # Observable-based sweep trends
                all_obs_names = set()
                for sr in sweep_results:
                    if sr.get("observables"):
                        all_obs_names.update(sr["observables"].keys())
                if all_obs_names:
                    insights.append("**Observable trends across sweep:**")
                    for obs_name in sorted(all_obs_names):
                        vals_and_params = []
                        for sr in sweep_results:
                            obs_dict = sr.get("observables", {}).get(obs_name, {})
                            v = obs_dict.get("value")
                            if isinstance(v, (int, float)) and sr.get("success"):
                                vals_and_params.append((sr["param_value"], v))
                        if len(vals_and_params) >= 2:
                            ov = [vp[1] for vp in vals_and_params]
                            unit = sweep_results[0].get("observables", {}).get(obs_name, {}).get("unit", "")
                            insights.append(f"- **{obs_name}**: {min(ov):.4g} → {max(ov):.4g} {unit} (range across sweep)")
                    insights.append("")

                for field_name in sweep_results[0].get("fields", {}):
                    means = [r["fields"][field_name]["mean"] for r in sweep_results if r.get("success") and field_name in r.get("fields", {})]
                    if means:
                        mean_min, mean_max = min(means), max(means)
                        insights.append(f"  - {field_name} mean: [{mean_min:.4g} .. {mean_max:.4g}] across sweep")
                        # Sensitivity check
                        mean_avg = sum(means) / len(means) if means else 1
                        if mean_avg != 0 and abs(mean_max - mean_min) / abs(mean_avg) < 0.05:
                            insights.append(f"  - ⚠ LOW SENSITIVITY: {field_name} varies by less than 5% across the sweep range. The metric may be insensitive to this parameter.")
                            insights.append(f"    Consider using a more sensitive diagnostic (e.g., velocity gradients, vorticity, mixing index).")
                        if len(means) >= 3:
                            # Monotonicity check
                            diffs = [means[j+1] - means[j] for j in range(len(means)-1)]
                            if all(d > 0 for d in diffs):
                                insights.append(f"  - {field_name} increases monotonically with {result.get('param_path')}")
                                # Flag if monotonic to boundary
                                if param_values:
                                    insights.append(f"    Note: monotonic trend means the extreme is at the boundary of the sweep range, not necessarily an optimum.")
                            elif all(d < 0 for d in diffs):
                                insights.append(f"  - {field_name} decreases monotonically with {result.get('param_path')}")
                                if param_values:
                                    insights.append(f"    Note: monotonic trend means the extreme is at the boundary of the sweep range, not necessarily an optimum.")
                            else:
                                insights.append(f"  - {field_name} shows non-monotonic behavior -- possible optimum exists in this range")
                insights.append("")

                # ---- Structured sweep data table (CRITICAL for LLM reasoning) ----
                if all_obs_names and sweep_results:
                    insights.append("**Sweep data table (use these exact values for analysis):**")
                    # Build header
                    top_obs = sorted(all_obs_names)[:6]  # limit columns
                    header = f"  | {result.get('param_path', 'param')} | " + " | ".join(top_obs) + " |"
                    insights.append(header)
                    insights.append("  |" + "---|" * (len(top_obs) + 1))
                    for sr in sweep_results:
                        if not sr.get("success"):
                            continue
                        pv = sr.get("param_value", "?")
                        vals = []
                        for oname in top_obs:
                            od = sr.get("observables", {}).get(oname, {})
                            v = od.get("value") if isinstance(od, dict) else None
                            if isinstance(v, (int, float)):
                                vals.append(f"{v:.4g}")
                            elif isinstance(v, bool):
                                vals.append("Y" if v else "N")
                            else:
                                vals.append("-")
                        pv_str = f"{pv:.4g}" if isinstance(pv, (int, float)) else str(pv)
                        insights.append(f"  | {pv_str} | " + " | ".join(vals) + " |")
                    insights.append("")

                    # ---- Curve feature detection ----
                    for obs_name in top_obs:
                        obs_vals = []
                        obs_params = []
                        for sr in sweep_results:
                            if not sr.get("success"):
                                continue
                            od = sr.get("observables", {}).get(obs_name, {})
                            v = od.get("value") if isinstance(od, dict) else None
                            if isinstance(v, (int, float)):
                                obs_vals.append(v)
                                obs_params.append(sr["param_value"])
                        if len(obs_vals) >= 3:
                            # Detect curve shape
                            diffs = [obs_vals[j + 1] - obs_vals[j] for j in range(len(obs_vals) - 1)]
                            all_inc = all(d > 0 for d in diffs)
                            all_dec = all(d < 0 for d in diffs)
                            # Check for exponential growth (ratio of consecutive diffs)
                            is_exponential = False
                            if all_inc and len(diffs) >= 3:
                                ratios = [diffs[j + 1] / diffs[j] if abs(diffs[j]) > 1e-30 else 0
                                          for j in range(len(diffs) - 1)]
                                if all(r > 1.5 for r in ratios):
                                    is_exponential = True
                            # Check for plateau (last 2+ values within 5%)
                            has_plateau = False
                            if len(obs_vals) >= 3:
                                last3 = obs_vals[-3:]
                                avg_last = sum(last3) / len(last3)
                                if avg_last != 0 and all(abs(v - avg_last) / abs(avg_last) < 0.05 for v in last3):
                                    has_plateau = True
                            # Find inflection/knee point
                            knee_idx = None
                            if len(diffs) >= 3:
                                second_diffs = [diffs[j + 1] - diffs[j] for j in range(len(diffs) - 1)]
                                max_curvature = max(range(len(second_diffs)), key=lambda j: abs(second_diffs[j]))
                                if abs(second_diffs[max_curvature]) > abs(obs_vals[-1] - obs_vals[0]) * 0.1:
                                    knee_idx = max_curvature + 1

                            features = []
                            if is_exponential:
                                features.append("exponential growth")
                            elif all_inc:
                                features.append("monotonically increasing")
                            elif all_dec:
                                features.append("monotonically decreasing")
                            else:
                                features.append("non-monotonic")
                            if has_plateau:
                                features.append(f"plateau at ~{obs_vals[-1]:.4g}")
                            if knee_idx is not None:
                                features.append(f"knee/inflection near {result.get('param_path')}={obs_params[knee_idx]:.4g}")

                            if features:
                                insights.append(f"**{obs_name} curve shape:** {', '.join(features)}")
                    insights.append("")

                if is_industry:
                    insights.append("**Operational interpretation:** For each sweep point, state the parameter value AND the corresponding physical outcome. Identify the EXACT parameter value where the compliance threshold is first met or lost.")
                    if param_values and len(param_values) >= 2:
                        insights.append(f"  → Swept values: {[round(v, 6) for v in param_values]}. Map each to its physical meaning for this application.")
                    insights.append("**Recommendation format:** State the safe operating range as: '{parameter} must be between X and Y to meet the requirement, with Z% margin at the recommended setpoint of W.'")
                else:
                    insights.append("**Physics interpretation:** The parameter sweep reveals how the system responds to changes in the control variable. Use the data table above to identify specific transition points, knee voltages, or operating limits.")

        elif tool_name == "optimize_parameter":
            opt_val = result.get('optimal_param_value', 0)
            opt_metric = result.get('optimal_metric', 0)
            insights.append(f"- Optimization: {result.get('param_path')} optimal={opt_val:.4g}, metric={opt_metric:.4g}")
            bounds = result.get("bounds", [])
            at_boundary = False
            if bounds and len(bounds) == 2:
                span = abs(bounds[1] - bounds[0])
                if span > 0 and (abs(opt_val - bounds[0]) < 0.02 * span or abs(opt_val - bounds[1]) < 0.02 * span):
                    at_boundary = True
                    insights.append(f"  - ⚠ BOUNDARY OPTIMUM: optimal value {opt_val:.4g} is at the edge of search range [{bounds[0]:.4g}, {bounds[1]:.4g}].")
                    insights.append(f"    The TRUE optimum likely lies OUTSIDE the tested range. This is NOT a confirmed optimum.")
                    insights.append(f"    You MUST report this as a boundary result, NOT as 'the optimal value.'")
                    insights.append(f"    Recommend extending the search range beyond {bounds[1]:.4g} (or below {bounds[0]:.4g}) to locate the true optimum.")
            insights.append("")
            if is_industry:
                insights.append(f"**Operational interpretation:** The optimal {result.get('param_path')} = {opt_val:.4g} achieves the best performance metric of {opt_metric:.4g}. Translate this to a specific engineering setpoint (e.g., 'set impeller speed to X RPM' or 'use pipe diameter Y mm').")
                insights.append("**Margin analysis:** State how much the parameter can deviate from optimal before performance drops below the acceptable threshold. This defines the operating tolerance band.")
            else:
                insights.append("**Physics interpretation:** The optimal point represents the best trade-off within the search range. If the optimum is at a boundary, the true optimum may lie outside the tested range.")

        elif tool_name == "compare_scenarios":
            comp = result.get("comparison", [])
            if comp:
                insights.append(f"- Compared {len(comp)} scenarios")
                for c in comp:
                    if c.get("success"):
                        field_summary = ", ".join(f"{fn}: mean={fs['mean']:.3g}" for fn, fs in c.get("fields", {}).items())
                        insights.append(f"  - {c['label']}: {field_summary}")
                insights.append("")
                if is_industry:
                    insights.append("**Operational interpretation:** Rank the scenarios from best to worst for the decision at hand. For each scenario, state whether it PASSES or FAILS the requirement, and by what margin. Identify the critical transition point between scenarios where acceptable becomes unacceptable.")
                    insights.append("**Decision format:** 'Scenario A (label) meets the requirement with X% margin. Scenario C (label) fails — the metric exceeds the limit by Y%. The transition occurs between Scenario B and C.'")
                else:
                    insights.append("**Physics interpretation:** Side-by-side comparison reveals which configuration parameters have the strongest influence on the output. Large differences between scenarios indicate high sensitivity to that parameter.")

        elif tool_name == "run_uncertainty_quantification":
            stats = result.get("statistics", {})
            for key, s in stats.items():
                insights.append(f"- UQ {key}: mean={s['mean']:.4g} +/- {s['std']:.4g} (CV={s['cv_percent']:.1f}%), 90% CI=[{s['p5']:.4g}, {s['p95']:.4g}]")
                if s.get('cv_percent', 0) > 20:
                    insights.append(f"  - WARNING: High coefficient of variation ({s['cv_percent']:.1f}%) -- results are highly sensitive to input uncertainty")
            insights.append("")
            if is_industry:
                insights.append("**Risk interpretation:** The 90% confidence interval defines the range of outcomes you should design for. If the UPPER bound (p95) of a harmful metric exceeds the safety threshold, the design has insufficient margin even at the mean.")
                for key, s in stats.items():
                    if s.get('cv_percent', 0) > 10:
                        insights.append(f"  → {key}: CV={s['cv_percent']:.1f}% — this parameter is sensitive to input variation. Tighten the tolerance on the input that drives it, or add design margin to absorb the spread.")
                insights.append("**Decision format:** 'Under worst-case input variation (p95), the metric reaches X, which is [above/below] the limit of Y by Z%. The design [does/does not] have adequate margin.'")
            else:
                insights.append("**Physics interpretation:** Monte Carlo UQ quantifies how input uncertainty propagates to output uncertainty. High CV% fields are the ones that need tighter manufacturing tolerances or better input data.")

    # ---- Cross-tool observable correlation ----
    # When multiple tools ran, correlate their observables to answer coupled questions
    all_obs: Dict[str, Dict[str, Any]] = {}  # module_name -> {obs_name: value}
    for result in tool_results:
        tool_name = result.get("_tool_name", "")
        module = result.get("module_name", "")
        if tool_name == "run_module" and result.get("observables"):
            for obs in result["observables"]:
                all_obs.setdefault(module, {})[obs["name"]] = obs
        elif tool_name == "sweep_parameter" and result.get("results"):
            for sr in result["results"]:
                if sr.get("observables"):
                    for obs_name, obs in sr["observables"].items():
                        all_obs.setdefault(module, {})[obs_name] = obs  # last sweep point
        elif tool_name == "compare_scenarios" and result.get("comparison"):
            for comp in result["comparison"]:
                if comp.get("observables"):
                    for obs_name, obs in comp["observables"].items():
                        all_obs.setdefault(module, {})[obs_name] = obs

    if len(all_obs) >= 2:
        insights.append("")
        insights.append("## Cross-Module Observable Correlation")
        insights.append("Multiple physics modules were executed. Key observables to correlate:")
        for mod, obs_dict in all_obs.items():
            top = sorted(obs_dict.values(), key=lambda o: o.get("rank", 99))[:3]
            for o in top:
                v = o.get("value", "?")
                u = o.get("unit", "")
                m_val = o.get("margin")
                v_str = f"{v:.4g}" if isinstance(v, float) else str(v)
                margin_str = f" (margin={m_val:+.3g})" if isinstance(m_val, (int, float)) else ""
                insights.append(f"- [{mod}] **{o['name']}**: {v_str} {u}{margin_str} — {o.get('description', '')}")
        insights.append("")
        insights.append("**CRITICAL: Cross-reference these observables to answer the coupled question.** "
                        "For example: if aero heat flux exceeds the thermal limit implied by structural margin, "
                        "report the EXACT angle of attack or parameter value where the crossover occurs. "
                        "State: 'At [parameter]=[value], [observable A] from [module 1] exceeds [threshold] "
                        "derived from [observable B] in [module 2], indicating failure onset.'")

    return "\n".join(insights) if insights else "No quantitative insights extracted."


def _build_analysis_rigor_instructions() -> str:
    """Return engineering-rigor guidelines the LLM must follow when interpreting results."""
    return """
## Engineering Analysis Rigor — Mandatory Guidelines

You MUST follow every rule below when writing your analysis. Violations destroy credibility.

### 1. Question the Objective Function
- State explicitly what metric was optimized/measured and whether it truly represents the real-world goal.
- Flag proxy metrics: e.g., "velocity magnitude" ≠ mixing quality; high velocity can increase shear damage.
- If the metric is a proxy, state its limitations and what a better metric would be.

### 2. Never Trust a Boundary Optimum
- If the optimal value sits at the edge of the search range, say so explicitly: "The optimum was NOT found — the objective is still improving at the boundary of the tested range."
- Do NOT present a boundary result as "the optimal value."
- Recommend extending the range or justify why the boundary is physically meaningful.

### 3. Parameter Consistency
- If parameters changed between runs (different lid velocity, different objective, different range), state this clearly and warn that results are NOT directly comparable.
- One study = one consistent set of physics + parameters. If anything changed, treat it as a separate experiment.

### 4. Check Metric Sensitivity
- If the output barely changes across the sweep, say so: "The metric shows low sensitivity to this parameter — the model may not be informative for this variable."
- Suggest more sensitive diagnostics (velocity gradients, vorticity, mixing entropy, scalar dispersion).

### 5. Separate Physics from Numerical Artifacts
- Acknowledge the grid resolution (e.g., 24×24 is coarse) and its implications.
- Note that results have NOT been verified with a grid independence study unless one was performed.
- Flag potential numerical diffusion or convergence issues.

### 6. Do Not Over-Interpret Sparse Data
- An 8-point sweep is useful but limited. Do NOT claim precise ranges (e.g., "0.043–0.050") from sparse points.
- Report: observed trend direction, confidence level (low/moderate/high).
- Use words like "approximately" and "in the vicinity of" for interpolated values.

### 7. Separate Observation from Inference
Structure your analysis clearly:
- **Observation:** What the analysis computed (numbers only).
- **Interpretation:** What the numbers physically mean and why.
- **Limitation:** What the model cannot capture.
- **Recommendation:** What to do next.
Mixing these layers is a credibility loss.

### 8. Include Physical Trade-Offs
- Engineering decisions are never single-metric. For every recommendation, state:
  - Benefits of the recommended setting
  - Risks or downsides (energy cost, shear stress, mixing time, etc.)
  - Unknowns that remain

### 9. Be Explicit About Limitations
Openly state:
- Simplified geometry (e.g., lid-driven cavity ≠ real mixer)
- Proxy metric limitations
- Boundary-driven flow dominance
- Limited parameter space explored
- No turbulence model, 2D assumption, etc.

### 10. Use Conservative Language
- NEVER say "optimal viscosity is X" unless the optimum is interior and well-resolved.
- USE: "Within the tested range…", "Preliminary indication suggests…", "Requires further validation…"
- Match confidence of language to strength of evidence.

### 11. Align Recommendations with Evidence Strength
| Evidence Strength | What You Can Say |
|---|---|
| Weak (sparse data, boundary optimum, proxy metric) | "Suggest further study" |
| Moderate (clear trend, interior optimum, but limited validation) | "Tentative operating range" |
| Strong (validated, grid-independent, domain-relevant metric) | "Recommend operating condition" |

### 12. Think Like a Reviewer
Before finalizing, verify:
- Would someone else reproduce this conclusion from the same data?
- Is every number traceable to the analysis output?
- Are all assumptions stated?
- Could a skeptical engineer poke holes in the logic?
"""


# ---------------------------------------------------------------------------
#  Chitchat Detection & Conversational Response
# ---------------------------------------------------------------------------
_ANALYSIS_KEYWORDS = [
    "simulat", "analys", "evaluat", "run ", "solve", "sweep", "compar", "optimiz", "cavity",
    "junction", "radar", "diode", "flow", "navier", "drift", "sensor",
    "module", "thermal", "channel", "grid", "viscosity", "voltage",
    "reynolds", "pressure", "velocity", "temperature", "mesh", "fem",
    "poisson", "chain", "uncertainty", "monte carlo", "uq",
]


# ---------------------------------------------------------------------------
#  Intent Classification & Conversational Mode
# ---------------------------------------------------------------------------

def _classify_intent(prompt: str, token: str, model: str,
                     history_context: str) -> str:
    """Use the LLM to classify user intent as 'conversation' or 'analysis'.

    Returns 'conversation' or 'analysis'.
    """
    modules = sorted(TRIALITY_MODULES_INFO.keys())
    module_list = ", ".join(modules)

    context_block = ""
    if history_context:
        context_block = f"\nConversation history:\n{history_context}\n"

    classification_prompt = f"""Classify the user's message as either CONVERSATION or ANALYSIS.

CONVERSATION — the user is asking a question, seeking advice, discussing a concept, troubleshooting, or asking how to set something up. They are NOT providing enough concrete parameters to run a simulation. Examples:
- "Why is my diode not turning on properly?"
- "I want to design a diode for low voltage switching — how should I set this up?"
- "What module should I use for thermal analysis?"
- "Can you explain the Shockley equation?"
- "What doping levels are typical for a power diode?"

ANALYSIS — the user is providing specific numerical parameters and wants a simulation run. They have concrete values (doping levels, temperatures, voltages, dimensions, etc.) or are explicitly requesting a computation. Examples:
- "Characterize a silicon PN junction at 300K with Nd=1e17 cm-3"
- "Sweep voltage from 0 to 0.7V for Na=5e16"
- "Nd=1e16, Na=1e15, now check"
- "Run thermal analysis at 500K with 100W heat source"
- "Compare 1e16 vs 1e17 doping"

Available physics modules: {module_list}
{context_block}
User's message: "{prompt}"

Reply with ONLY one word: CONVERSATION or ANALYSIS"""

    result = call_llm(
        "You are an intent classifier. Reply with exactly one word: CONVERSATION or ANALYSIS. Nothing else.",
        classification_prompt,
        token=token,
        model=model,
    )
    if result and "analysis" in result.strip().lower():
        return "analysis"
    return "conversation"


def _build_conversational_response(prompt: str, token: str, model: str,
                                    history_context: str) -> Optional[str]:
    """Use the LLM to generate a conversational response for discussion-type queries."""
    modules = sorted(TRIALITY_MODULES_INFO.keys())
    module_list = ", ".join(modules)

    system = (
        "You are the Triality Agent — a physics reasoning assistant powered by Mentis OS, "
        "built by Genovation Technological Solutions.\n\n"
        "You have the following capabilities:\n"
        f"- **{len(modules)} physics modules:** {module_list}\n"
        "- **6 tools:** run_module, sweep_parameter, optimize_parameter, compare_scenarios, "
        "run_uncertainty_quantification, chain_modules\n\n"
        "Respond naturally to whatever the user says — greetings, questions, design advice, troubleshooting, "
        "concept explanations, or capability questions. Be helpful, specific, and concise.\n"
        "If the question would benefit from a simulation, suggest what analysis they could run and what parameters "
        "they would need to provide.\n\n"
        "Use markdown formatting. Keep your response to 1-4 paragraphs depending on complexity. "
        "Use LaTeX math notation for expressions — e.g. $V_{bi}$, $10^{17}$ cm$^{-3}$."
    )

    user_msg = history_context + prompt if history_context else prompt

    return call_llm(system, user_msg, token=token, model=model)


# ---------------------------------------------------------------------------
#  Tool Execution Helper (shared by both legacy and goal-driven paths)
# ---------------------------------------------------------------------------
# Mapping of tools that support step-by-step progress streaming
_ITER_TOOLS = {
    "run_module": _iter_run_module,
    "sweep_parameter": _iter_sweep_parameter,
    "optimize_parameter": _iter_optimize_parameter,
    "compare_scenarios": _iter_compare_scenarios,
    "run_uncertainty_quantification": _iter_uncertainty_quantification,
}


# ---------------------------------------------------------------------------
#  Agentic Loop — SSE Streaming (Goal-Driven)
# ---------------------------------------------------------------------------
def _run_agent_turn(prompt: str, token: str, model: str, scenario_id: Optional[str],
                    conversation_history: Optional[List[Dict[str, str]]] = None):
    """Generator that yields SSE events for the goal-driven agentic loop.

    Pipeline:
        Phase 0:  Goal Extraction — parse user prompt into structured goal
        Phase 0.5: Analytical Estimation — LLM derives first-principles estimate
        Phase 1:  Planning — LLM generates initial tool_calls (goal-informed)
        Phase 2:  Goal-Driven Execution — iterate until goal is satisfied
        Phase 3:  Summarization — LLM interprets converged results
        Phase 4:  Reflection — optional follow-up for completeness
    """

    turn_id = str(uuid.uuid4())[:8]

    def sse(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    yield sse("turn_start", {"turn_id": turn_id, "prompt": prompt})

    # ------- Build conversation context for LLM -------
    history_context = ""
    if conversation_history:
        recent = conversation_history[-10:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:1500]
            if content.strip():
                history_lines.append(f"[{role}]: {content}")
        if history_lines:
            history_context = "CONVERSATION HISTORY (for context):\n" + "\n".join(history_lines) + "\n\n"

    # ------- Intent classification: conversation vs analysis -------
    if not scenario_id:
        yield sse("phase", {"phase": "thinking", "message": "Thinking..."})
        intent = _classify_intent(prompt, token, model, history_context)
        if intent == "conversation":
            conv_response = _build_conversational_response(prompt, token, model, history_context)
            if conv_response:
                yield sse("chitchat", {"response": conv_response})
                yield sse("turn_complete", {"turn_id": turn_id, "total_tools": 0, "all_succeeded": True})
                return

    # ------- Phase 0: Goal Extraction -------
    yield sse("phase", {"phase": "goal_extraction", "message": "Understanding your goal..."})

    goal = None
    analytical = None

    # Pre-check: if the prompt explicitly says "characterize", "analyze", "show", "describe"
    # without any optimization/threshold language, force characterize mode
    _prompt_lower = prompt.lower()
    _is_explicit_characterize = (
        any(w in _prompt_lower for w in ["characterize", "analyze", "describe", "show", "visualize", "plot"])
        and not any(w in _prompt_lower for w in ["optimize", "maximize", "minimize", "find where",
                                                   "find the", "at what", "threshold", "exceeds"])
    )

    if _is_explicit_characterize:
        goal = AnalysisGoal(goal_type=GoalType.CHARACTERIZE, metric="", raw_prompt=prompt)
        logger.info("Explicit characterize prompt detected — skipping goal extraction")
    else:
        # Strategy 1: Try scenario-based goal extraction (if scenario provided)
        if scenario_id:
            for s in SCENARIOS:
                if s["id"] == scenario_id:
                    scenario_goal = extract_goal_from_scenario(s)
                    if scenario_goal and scenario_goal.goal_type != GoalType.CHARACTERIZE:
                        goal = scenario_goal
                        logger.info("Goal from scenario: type=%s, metric=%s", goal.goal_type.value, goal.metric)
                    break

        # Strategy 2: Try LLM-based goal extraction
        if not goal or goal.goal_type == GoalType.CHARACTERIZE:
            goal_prompt = GOAL_EXTRACTION_PROMPT.format(prompt=prompt)
            goal_output = call_llm(
                "You are a physics problem analyzer. Extract the user's goal into structured JSON. Be precise about units (use SI).",
                goal_prompt,
                token=token,
                model=model,
            )
            goal_json = parse_llm_json(goal_output) if goal_output else None

            if goal_json and goal_json.get("has_clear_goal"):
                goal = extract_goal_from_llm_response(goal_json, prompt)
                if goal:
                    logger.info("Goal extracted (LLM): %s (type=%s, metric=%s)", goal.metric, goal.goal_type.value, goal.threshold)
        elif not goal:
            # Strategy 3: Try heuristic fallback
            goal = extract_goal_heuristic(prompt)
            if goal:
                logger.info("Goal extracted (heuristic): type=%s", goal.goal_type.value)

    # Default to characterize if no clear goal
    if not goal:
        goal = AnalysisGoal(goal_type=GoalType.CHARACTERIZE, metric="", raw_prompt=prompt)

    is_goal_driven = goal.goal_type != GoalType.CHARACTERIZE

    if is_goal_driven:
        yield sse("goal_extracted", {
            "goal": goal.to_dict(),
            "message": f"Goal: {goal.goal_type.value} — {goal.metric}",
        })

        # ------- Phase 0.5: Analytical Pre-Flight -------
        if goal.goal_type == GoalType.FIND_THRESHOLD and goal.threshold is not None:
            yield sse("phase", {"phase": "analytical_estimation", "message": "Computing analytical estimate..."})

            analytical_prompt = ANALYTICAL_ESTIMATION_PROMPT.format(
                prompt=prompt,
                module_name=goal.module_name or "auto",
                search_variable=goal.search_variable or "unknown",
                metric=goal.metric,
                operator=goal.operator or ">",
                threshold=goal.threshold,
                unit=goal.unit or "",
            )
            analytical_output = call_llm(
                "You are a physics expert. Estimate the answer analytically using first principles. Be quantitative.",
                analytical_prompt,
                token=token,
                model=model,
            )
            analytical_json = parse_llm_json(analytical_output) if analytical_output else None

            if analytical_json:
                analytical = parse_analytical_estimate(analytical_json)
                # Use analytical bounds if goal has none
                if analytical.suggested_bounds and not goal.search_bounds:
                    goal.search_bounds = analytical.suggested_bounds

                yield sse("analytical_estimate", {
                    "estimate": analytical.to_dict(),
                    "message": f"Analytical estimate: {analytical.estimate:.4g} {analytical.unit or ''}" if analytical.estimate else "No analytical estimate available",
                })

    # Build system prompt early — needed by reflection phase even if decomposition path runs
    system_prompt = _build_system_prompt(goal=goal if is_goal_driven else None, analytical=analytical)

    # ------- Phase 1: Task Decomposition (multi-step LLM) -------
    yield sse("phase", {"phase": "planning", "message": "Decomposing into analysis tasks..."})

    try:
        from triality_app.task_decomposer import (
            decompose_prompt, configure_task, build_task_result_summary,
            build_module_info_for_task, CONSOLIDATE_PROMPT, _sanitize_args,
        )
    except ImportError:
        from task_decomposer import (  # type: ignore[no-redef]
            decompose_prompt, configure_task, build_task_result_summary,
            build_module_info_for_task, CONSOLIDATE_PROMPT, _sanitize_args,
        )

    full_prompt = history_context + prompt if history_context else prompt

    # First: try to decompose the prompt into ordered tasks
    decomposition = decompose_prompt(
        full_prompt, list(available_runtime_modules()), call_llm, token, model,
    )

    # Fallback: if decomposition fails, use single-call planning
    if not decomposition or not decomposition.get("tasks"):
        # Single-call fallback (original behavior)
        system_prompt = _build_system_prompt(goal=goal if is_goal_driven else None, analytical=analytical)
        llm_output = call_llm(system_prompt, full_prompt, token=token, model=model)
        if not llm_output:
            yield sse("error", {"error": "LLM call failed. Please check your Replicate API key and try again."})
            yield sse("turn_complete", {"turn_id": turn_id, "total_tools": 0, "all_succeeded": False})
            return
        plan = parse_llm_json(llm_output)
        if not plan:
            yield sse("chitchat", {"response": llm_output.strip()})
            yield sse("turn_complete", {"turn_id": turn_id, "total_tools": 0, "all_succeeded": True})
            return
        if not plan.get("tool_calls"):
            thinking = plan.get("thinking", "")
            response = plan.get("summary_request", thinking) or llm_output.strip()
            yield sse("chitchat", {"response": response})
            yield sse("turn_complete", {"turn_id": turn_id, "total_tools": 0, "all_succeeded": True})
            return
        decomposition = {
            "tasks": [
                {"step": i + 1, "description": tc.get("tool", ""), "tool": tc.get("tool", ""),
                 "purpose": "", "prebuilt_args": tc.get("args", {})}
                for i, tc in enumerate(plan.get("tool_calls", []))
            ],
            "final_question": prompt,
        }
        yield sse("thinking", {
            "thinking": plan.get("thinking", ""),
            "plan": plan.get("plan", []),
            "source": "llm",
            "goal_driven": is_goal_driven,
        })
    else:
        # Show decomposed plan
        tasks = decomposition["tasks"]
        yield sse("thinking", {
            "thinking": f"Decomposed into {len(tasks)} analysis tasks",
            "plan": [f"Step {t['step']}: {t['description']}" for t in tasks],
            "source": "llm",
            "goal_driven": is_goal_driven,
        })

    tasks = decomposition.get("tasks", [])

    # ------- Phase 2: Per-Task Execution -------
    all_results = []
    convergence_summary = ""
    final_eval = None
    task_summaries: List[str] = []
    module_info_str = build_module_info_for_task("", TRIALITY_MODULES_INFO, list(available_runtime_modules()))

    yield sse("phase", {"phase": "executing", "message": f"Running {len(tasks)} analysis task(s)..."})

    # Track the last successful config so subsequent tasks can reuse it
    _last_good_config: Optional[Dict] = None
    _last_good_module: Optional[str] = None

    # Valid tool names — skip any hallucinated tools
    _VALID_TOOLS = set(TOOL_REGISTRY.keys())

    for task in tasks:
        task_step = task.get("step", len(all_results) + 1)
        tool_name = task.get("tool", "run_module")
        task_desc = task.get("description", tool_name)

        # Skip hallucinated tools (visualize_results, plot_fields, etc.)
        if tool_name not in _VALID_TOOLS:
            logger.warning("Skipping hallucinated tool '%s' in task %s", tool_name, task_step)
            task_summaries.append(f"Step {task_step}: Skipped ('{tool_name}' is not a valid tool)")
            continue

        # Configure this task's tool arguments
        if task.get("prebuilt_args"):
            # From single-call fallback — already have args
            args = _sanitize_args(tool_name, task["prebuilt_args"])
        else:
            # Per-task LLM call — focused context with prior results
            prior_ctx = "\n\n".join(task_summaries) if task_summaries else ""
            args = configure_task(
                task, prompt, module_info_str, prior_ctx,
                call_llm, token, model, parse_llm_json,
            )
            if not args:
                args = {"error": f"Failed to configure task: {task_desc}"}
            else:
                args = _sanitize_args(tool_name, args)

        # Ensure config is present for tools that need it (use last good config)
        if tool_name in ("sweep_parameter", "optimize_parameter", "run_module"):
            if not args.get("config") and _last_good_config:
                args["config"] = copy.deepcopy(_last_good_config)
            if not args.get("module_name") and _last_good_module:
                args["module_name"] = _last_good_module

        if args.get("error"):
            result = {"error": args["error"], "_tool_name": tool_name, "_elapsed_s": 0, "_index": len(all_results)}
            all_results.append(result)
            task_summaries.append(f"Step {task_step}: FAILED — {args['error']}")
            continue

        idx = len(all_results)
        yield sse("tool_start", {"index": idx, "tool": tool_name, "args": args})

        started = time.perf_counter()

        iter_fn = _ITER_TOOLS.get(tool_name)
        if iter_fn:
            result = None
            try:
                for item in iter_fn(**args):
                    if isinstance(item, dict) and item.get("_progress"):
                        yield sse("tool_progress", {
                            "index": idx,
                            "tool": tool_name,
                            "step": item.get("step", 0),
                            "total": item.get("total", 0),
                            "detail": {k: v for k, v in item.items() if k != "_progress"},
                        })
                    else:
                        result = item
            except Exception as exc:
                result = {"error": str(exc)}
            if result is None:
                result = {"error": "Tool produced no result."}
        else:
            result = execute_tool_call(tool_name, args)

        elapsed = time.perf_counter() - started
        result["_tool_name"] = tool_name
        result["_elapsed_s"] = elapsed
        result["_index"] = idx

        all_results.append(result)
        yield sse("tool_result", {
            "index": idx,
            "tool": tool_name,
            "elapsed_s": round(elapsed, 3),
            "success": not result.get("error"),
            "result": result,
        })

        # Track last good config so subsequent tasks can reuse it
        if not result.get("error"):
            if args.get("config"):
                _last_good_config = copy.deepcopy(args["config"])
            if args.get("module_name"):
                _last_good_module = args["module_name"]

        # Build summary for next task's context
        task_summaries.append(build_task_result_summary(task, result))

        # Capture module_name/config for potential goal-driven follow-up
        if not goal.module_name and args.get("module_name"):
            goal.module_name = args["module_name"]
        if not goal.config and args.get("config"):
            goal.config = args["config"]

    # ------- Post-Execution: Convergence Loop with plateau detection -------
    # Only chase findings when the original prompt implies a goal (optimize, find threshold)
    # NOT for characterize/analyze/sweep requests where the user just wants to see results
    _should_chase = is_goal_driven  # Only chase if we identified a clear goal earlier
    initial_findings = detect_unresolved_findings(all_results) if _should_chase else []
    if initial_findings:
        finding = initial_findings[0]
        chase_goal = build_goal_from_finding(finding)

        if (chase_goal.goal_type != GoalType.CHARACTERIZE
                and chase_goal.search_variable and chase_goal.module_name):

            if not chase_goal.config:
                chase_goal.config = goal.config or _last_good_config

            # Initialize search state for proper convergence tracking
            search_state = SearchState(
                current_bounds=finding.suggested_bounds or chase_goal.search_bounds or (0, 1),
                objective="maximize" if chase_goal.goal_type == GoalType.MAXIMIZE else "minimize",
            )

            MAX_CHASE_ITERATIONS = 4
            for chase_iter in range(MAX_CHASE_ITERATIONS):
                yield sse("convergence_adapting", {
                    "iteration": chase_iter + 1,
                    "action": "search",
                    "details": f"Searching [{search_state.current_bounds[0]:.4g}, {search_state.current_bounds[1]:.4g}]",
                    "next_bounds": list(search_state.current_bounds),
                })

                # Build and execute the optimizer/sweep for this iteration
                tool_call = {
                    "tool": "optimize_parameter",
                    "args": {
                        "module_name": chase_goal.module_name,
                        "config": chase_goal.config or {},
                        "param_path": chase_goal.search_variable,
                        "bounds": list(search_state.current_bounds),
                        "objective_field": chase_goal.metric,
                        "objective": search_state.objective,
                        "n_evals": max(5, 10 - chase_iter),
                    },
                }

                extra_tool = tool_call["tool"]
                extra_args = tool_call["args"]
                extra_idx = len(all_results)

                yield sse("tool_start", {"index": extra_idx, "tool": extra_tool, "args": extra_args})

                started = time.perf_counter()
                iter_fn = _ITER_TOOLS.get(extra_tool)
                if iter_fn:
                    result = None
                    try:
                        for item in iter_fn(**extra_args):
                            if isinstance(item, dict) and item.get("_progress"):
                                yield sse("tool_progress", {
                                    "index": extra_idx, "tool": extra_tool,
                                    "step": item.get("step", 0), "total": item.get("total", 0),
                                    "detail": {k: v for k, v in item.items() if k != "_progress"},
                                })
                            else:
                                result = item
                    except Exception as exc:
                        result = {"error": str(exc)}
                    if result is None:
                        result = {"error": "Tool produced no result."}
                else:
                    result = execute_tool_call(extra_tool, extra_args)

                elapsed = time.perf_counter() - started
                result["_tool_name"] = extra_tool
                result["_elapsed_s"] = elapsed
                result["_index"] = extra_idx
                all_results.append(result)
                yield sse("tool_result", {
                    "index": extra_idx, "tool": extra_tool,
                    "elapsed_s": round(elapsed, 3),
                    "success": not result.get("error"),
                    "result": result,
                })

                # Extract best_x and best_y from optimizer result
                best_x = result.get("optimal_param_value", 0)
                best_y = result.get("optimal_metric", 0)

                # Run convergence check
                conv_result = check_convergence(search_state, best_x, best_y)

                yield sse("convergence_step", {
                    "iteration": chase_iter + 1,
                    "max_iterations": MAX_CHASE_ITERATIONS,
                    "evaluation": {
                        "satisfied": conv_result["converged"],
                        "answer": best_x,
                        "closest_value": best_y,
                        "action": conv_result["status"],
                        "details": conv_result["details"],
                    },
                    "total_solver_runs": len(all_results),
                })

                if conv_result["converged"]:
                    # Generate deterministic answer
                    confidence = compute_confidence(
                        converged=True,
                        boundary_plateau=(conv_result["status"] == "converged_boundary_plateau"),
                        metric_consistent=True,
                    )
                    convergence_summary = commit_answer_optimum(
                        param_name=chase_goal.search_variable,
                        best_x=best_x,
                        objective_name=chase_goal.metric,
                        best_y=best_y,
                        status=conv_result["status"],
                        confidence=confidence,
                        caveats="32x32 grid, 2D assumption" if "nx" in str(chase_goal.config) else None,
                    )
                    yield sse("goal_satisfied", {
                        "answer": best_x,
                        "answer_unit": chase_goal.unit,
                        "accuracy": conv_result["status"],
                        "iterations_used": chase_iter + 1,
                        "total_solver_runs": len(all_results),
                        "details": conv_result["details"],
                    })
                    # Set final_eval so the summary phase picks it up
                    final_eval = GoalEvaluation(
                        satisfied=True,
                        answer=best_x,
                        answer_unit=chase_goal.unit,
                        accuracy=conv_result["status"],
                        closest_value=best_y,
                        action=ConvergenceAction.REPORT_ANSWER,
                        details=conv_result["details"],
                    )
                    break

                elif conv_result["status"] == "expand":
                    # Expand bounds for next iteration
                    boundary_side = conv_result.get("boundary_side", "upper")
                    factor = 1.5 if chase_iter == 0 else 1.3
                    search_state.current_bounds = expand_bounds(
                        search_state.current_bounds, boundary_side, factor,
                    )
                else:
                    # No expansion possible, stop
                    break

    # ------- Phase 2.5: Physics Truth Layer -------
    try:
        from triality_app.physics_truth import validate_results as _validate_physics
    except ImportError:
        from physics_truth import validate_results as _validate_physics  # type: ignore[no-redef]

    truth_report = _validate_physics(all_results, module_name=goal.module_name, config=goal.config or _last_good_config)
    physics_truth_context = truth_report.to_llm_context()

    if truth_report.violations:
        yield sse("phase", {"phase": "validating", "message": f"Physics validation: {len(truth_report.violations)} issue(s) found"})

    # ------- Phase 2.75: Semantic Physics Layer -------
    try:
        from triality_app.semantic_physics import full_semantic_analysis as _semantic_analysis
    except ImportError:
        from semantic_physics import full_semantic_analysis as _semantic_analysis  # type: ignore[no-redef]

    _spl_understanding, _spl_context = _semantic_analysis(
        all_results, module_name=goal.module_name, config=goal.config or _last_good_config
    )

    if _spl_understanding.rejected_concepts or _spl_understanding.corrections:
        yield sse("phase", {"phase": "interpreting", "message": f"SPL: {_spl_understanding.regime} regime, {_spl_understanding.behavior} behavior"})

    # ------- Phase 3: Summarization -------
    yield sse("phase", {"phase": "summarizing", "message": "Interpreting results..."})

    scenario_ctx = _get_scenario_context(scenario_id)
    programmatic_insights = _build_programmatic_insights(all_results, scenario=scenario_ctx)
    industry_framing = _build_industry_framing(scenario_ctx)

    # Inject SPL semantic context into insights (before physics truth)
    if _spl_context:
        programmatic_insights = _spl_context + "\n\n" + programmatic_insights

    # Inject physics truth corrections into insights
    if physics_truth_context:
        programmatic_insights = physics_truth_context + "\n\n" + programmatic_insights

    # Inject convergence summary into insights if goal-driven
    if convergence_summary:
        programmatic_insights = convergence_summary + "\n\n" + programmatic_insights

    # Inject final answer context for goal-driven runs
    goal_answer_context = ""
    if final_eval and final_eval.satisfied and final_eval.answer is not None:
        unit = final_eval.answer_unit or goal.unit or ""
        goal_answer_context = f"""
## GOAL CONVERGENCE RESULT

The goal-driven engine has CONVERGED on an answer:
- **Answer: {final_eval.answer:.6g} {unit}**
- **Accuracy: {('±' + f'{final_eval.accuracy_pct:.2f}%') if final_eval.accuracy_pct is not None else 'practical optimum'}** ({final_eval.accuracy or 'converged'})
- **Goal: {goal.goal_type.value} — {goal.metric} {goal.operator or ''} {goal.threshold or ''} {goal.unit or ''}**

IMPORTANT: Lead your summary with this answer. The user asked a specific question and the convergence engine found the specific answer.
Do NOT say "further study needed" or "extend the range" — the answer has been found.
"""
    elif final_eval and not final_eval.satisfied:
        goal_answer_context = f"""
## GOAL CONVERGENCE RESULT

The goal-driven engine did NOT fully converge:
- **Best estimate so far:** {final_eval.answer or final_eval.closest_value or 'N/A'}
- **Details:** {final_eval.details}

Report what was found and clearly state what prevented full convergence.
"""

    # Build full results JSON
    results_json = json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_") and k != "state"} for r in all_results],
        indent=2, default=str,
    )[:16000]  # Increased limit for goal-driven runs with multiple iterations

    analysis_rigor = _build_analysis_rigor_instructions()

    summary_prompt = f"""The user asked: "{prompt}"

{goal_answer_context}

The following tool calls were executed and produced these results:

{results_json}

Programmatic insights:
{programmatic_insights}
{industry_framing}
{analysis_rigor}

STRICT RULES:
- ONLY report numbers that appear VERBATIM in the results above. Every number you cite MUST be traceable to the JSON or programmatic insights.
- Do NOT invent, estimate, or hallucinate any numerical values.
- Report what was computed, key findings, and physical interpretation.
- If a tool returned an error, report the error.
- Use markdown formatting with headers and bullet points.
- Use LaTeX math notation for all units, exponents, and expressions — e.g. $10^{{17}}$ cm$^{{-3}}$, $1.57 \\times 10^{{5}}$ A/cm$^{{2}}$, $V_{{bi}} = 0.79$ V. This is required for proper rendering.
- Be concise but thorough. Focus on engineering significance.
- If a goal convergence result is provided above, LEAD with that answer prominently.
- If an optimum is at the boundary of the search range, explicitly flag it as a boundary result, NOT a true optimum.
- If the metric barely changes across a sweep, flag low sensitivity.
- State ALL limitations of the analysis setup (grid resolution, 2D assumption, proxy metrics, etc.).
- Use conservative language proportional to evidence strength. Do NOT overstate conclusions.
- Structure your response with these sections:
  1. **Answer** — ALWAYS lead with a direct answer to the user's question. State the key numerical results with units prominently. If the user asked a specific question, answer it first in 1-2 sentences before giving details.
  2. **Key Results** — list the important computed values (numbers from results only, with units)
  3. **Physics Interpretation** — why the results look this way (use the programmatic insights above)
  4. **Limitations & Caveats** — grid resolution, proxy metrics, boundary effects, parameter space coverage, model assumptions. Keep this to 2-3 bullet points, not an exhaustive list.
  5. **{"Decision & Recommended Action" if scenario_ctx else "Design Recommendations"}** — {"directly answer the decision question with a YES/NO/CONDITIONAL recommendation backed by analysis numbers. State evidence strength (weak/moderate/strong) and physical trade-offs." if scenario_ctx else "actionable next steps based on the findings. Keep this brief — 1-2 concrete suggestions."}
"""
    system_role = (
        "You are a senior industry physics analyst who translates analysis results into engineering decisions. "
        "You think like a critical reviewer: question objective functions, flag boundary optima, state limitations, "
        "separate observations from inferences, and use conservative language proportional to evidence strength. "
        "Map grid positions to physical locations, compare results against compliance thresholds, include physical trade-offs, "
        "and give clear go/no-go recommendations with stated confidence level. Never invent numbers. "
        "Every number you report MUST appear in the analysis results provided to you."
        if scenario_ctx else
        "You are a senior lead engineer who solves problems completely. When a convergence result gives a specific answer, "
        "LEAD with that answer — do not bury it. You think like a critical reviewer: question objective functions, "
        "flag boundary optima, state limitations, separate observations from inferences, and use conservative language "
        "proportional to evidence strength. Include physical trade-offs and acknowledge what the analysis cannot capture. "
        "Never invent numbers. Every number you report MUST appear in the analysis results provided to you."
    )
    summary = call_llm(
        system_role,
        summary_prompt,
        token=token,
        model=model,
    )
    if not summary:
        yield sse("error", {"error": "LLM summary call failed. Results were computed but could not be interpreted. Check your Replicate API key."})
        yield sse("turn_complete", {"turn_id": turn_id, "total_tools": len(all_results), "all_succeeded": False})
        return

    yield sse("summary", {"summary": summary, "insights": programmatic_insights})

    # ------- Phase 4: Reflection (optional, skipped if goal already converged) -------
    if not (final_eval and final_eval.satisfied):
        scenario_category = ""
        if scenario_ctx:
            scenario_category = scenario_ctx.get("category", "")

        max_reflections = 3 if scenario_category == "advanced" else MAX_REFLECTION_STEPS
        for reflection_step in range(max_reflections):
            urgency = ""
            if scenario_category == "advanced":
                urgency = """
IMPORTANT: This is an ADVANCED multi-physics scenario. You should STRONGLY consider follow-up actions:
- If results are near a safety threshold or margin, run UQ to quantify uncertainty
- If a sweep was performed, suggest a comparison of key configurations
- If multiple modules were used, suggest cross-referencing their results
- Default to "complete": false unless you are confident the analysis is truly comprehensive
"""
            elif scenario_category == "industry":
                urgency = """
NOTE: This is an INDUSTRY scenario with real business impact. Consider:
- Running UQ on the parameter closest to a safety/compliance threshold
- Suggesting a comparison scenario if only one configuration was tested
"""

            reflect_prompt = f"""You just completed an analysis. Results summary:
{summary[:2000]}

Programmatic insights:
{programmatic_insights}
{urgency}
Is the analysis complete, or would ONE extra tool call significantly improve the answer?
Respond with JSON: {{"complete": true/false, "reason": "...", "extra_tool_call": {{"tool": "...", "args": {{...}}}} }}
Only suggest a follow-up if it adds significant value (e.g., UQ near safety limits, sweep for sensitivity, comparison for design alternatives).
"""
            reflect_output = call_llm(system_prompt, reflect_prompt, token=token, model=model)
            reflect = parse_llm_json(reflect_output) if reflect_output else None

            if not reflect or reflect.get("complete", True):
                break

            extra_tc = reflect.get("extra_tool_call")
            if not extra_tc or not extra_tc.get("tool"):
                break

            yield sse("reflection", {"step": reflection_step, "reason": reflect.get("reason", "")})

            extra_tool_name = extra_tc["tool"]
            extra_args = extra_tc.get("args", {})
            extra_idx = len(all_results)
            yield sse("tool_start", {"index": extra_idx, "tool": extra_tool_name, "args": extra_args})

            started = time.perf_counter()

            iter_fn = _ITER_TOOLS.get(extra_tool_name)
            if iter_fn:
                result = None
                try:
                    for item in iter_fn(**extra_args):
                        if isinstance(item, dict) and item.get("_progress"):
                            yield sse("tool_progress", {
                                "index": extra_idx,
                                "tool": extra_tool_name,
                                "step": item.get("step", 0),
                                "total": item.get("total", 0),
                                "detail": {k: v for k, v in item.items() if k not in ("_progress",)},
                            })
                        else:
                            result = item
                except Exception as exc:
                    result = {"error": str(exc)}
                if result is None:
                    result = {"error": "Tool produced no result."}
            else:
                result = execute_tool_call(extra_tool_name, extra_args)

            elapsed = time.perf_counter() - started
            result["_tool_name"] = extra_tool_name
            result["_elapsed_s"] = elapsed

            all_results.append(result)
            yield sse("tool_result", {
                "index": extra_idx,
                "tool": extra_tool_name,
                "elapsed_s": round(elapsed, 3),
                "success": not result.get("error"),
                "result": result,
            })

            # Generate addendum with LLM
            extra_insights = _build_programmatic_insights([result])
            programmatic_insights += "\n" + extra_insights

            addendum_prompt = f"""Reflection step {reflection_step + 1}: an additional analysis was run.
Original summary: {summary[:2000]}
Extra tool: {extra_tc['tool']}, result: {json.dumps({k: v for k, v in result.items() if not k.startswith("_") and k != "state"}, default=str)[:2000]}
New deterministic insights: {extra_insights[:1500]}
Write a short (3-5 sentence) addendum that updates or refines the original summary with the new data.
If this was a UQ run, report the confidence interval. If a sweep, note the sensitivity.
If a comparison, highlight the key differences.
STRICT: only report numbers from the result JSON above. No hallucination."""
            addendum = call_llm(
                "You are a senior physics analyst with rigorous engineering judgment. "
                "Summarize additional results accurately. Flag boundary optima, state limitations, "
                "use conservative language, and never invent numbers.",
                addendum_prompt,
                token=token,
                model=model,
            )
            if not addendum:
                addendum = extra_insights
            summary = summary + f"\n\n---\n**Agent reflection {reflection_step + 1}:**\n" + addendum
            yield sse("reflection_addendum", {"addendum": addendum})

    # ------- Phase 5: Final Consolidation -------
    # One final LLM call that sees EVERYTHING: original prompt, all results,
    # all insights, all reflections — and produces the definitive conclusion.
    if len(all_results) >= 2:
        yield sse("phase", {"phase": "consolidating", "message": "Producing final answer..."})

        # Rebuild insights with ALL results (including reflection tools)
        final_insights = _build_programmatic_insights(all_results, scenario=scenario_ctx)
        if convergence_summary:
            final_insights = convergence_summary + "\n\n" + final_insights

        final_results_json = json.dumps(
            [{k: v for k, v in r.items() if not k.startswith("_") and k != "state"} for r in all_results],
            indent=2, default=str,
        )[:20000]

        # Include physics truth corrections and SPL in consolidation
        _ptl_context = physics_truth_context if physics_truth_context else ""
        _spl_consolidation = _spl_context if _spl_context else ""

        consolidation_prompt = f"""You are a senior lead engineer delivering a final conclusion.

The user asked: "{prompt}"

ALL analysis steps have been completed. Here are the complete results:

{final_results_json}

Complete programmatic insights:
{final_insights}

{_spl_consolidation}

{_ptl_context}

Previous summary (for context, may be incomplete):
{summary[:3000]}

{goal_answer_context}
{industry_framing}

YOUR JOB: Write the DEFINITIVE final answer. This replaces everything above.

RULES:
1. LEAD with a direct answer to the user's question in the FIRST sentence.
   - For characterization: state the key device/system parameters discovered
   - For threshold finding: state the exact threshold value
   - For optimization: state the optimal parameter and its value
   - For comparison: state the winner and by how much
2. Include a concise data summary (key numbers from results, with units)
3. State confidence level (high/moderate/low) based on convergence and validation
4. State limitations in ONE sentence, not a bullet list
5. Do NOT say "further study needed" unless the analysis genuinely failed to answer the question
6. Do NOT repeat the methodology — the user can see the tool blocks above
7. Every number MUST come from the results JSON or the Physics Validation corrections above — never invent values
8. If Physics Validation provides CORRECTED VALUES, use those instead of solver output and state they were recomputed
9. Keep it to 2-3 short paragraphs maximum
10. Use LaTeX math notation for all units, exponents, and expressions — e.g. $10^{{17}}$ cm$^{{-3}}$, $1.57 \\times 10^{{5}}$ A/cm$^{{2}}$, $V_{{bi}} = 0.79$ V

Write the final answer now:"""

        final_answer = call_llm(
            "You are a decisive senior engineer. Give the final answer — concise, specific, numbers-first. "
            "Never hedge when the data supports a conclusion. Never say 'further study needed' when the analysis answered the question.",
            consolidation_prompt,
            token=token,
            model=model,
        )
        if final_answer:
            # Replace the summary with the consolidated final answer
            summary = final_answer
            yield sse("summary", {"summary": f"## Final Answer\n\n{final_answer}", "insights": final_insights})

    yield sse("turn_complete", {
        "turn_id": turn_id,
        "total_tools": len(all_results),
        "all_succeeded": all(not r.get("error") for r in all_results),
        "goal_driven": is_goal_driven,
        "goal_satisfied": final_eval.satisfied if final_eval else None,
        "answer": final_eval.answer if final_eval else None,
    })


def _build_deterministic_summary(
    prompt: str,
    plan: Dict,
    results: List[Dict],
    insights: str,
) -> str:
    """Build a structured summary without LLM."""
    lines = ["## Simulation Results\n"]

    for result in results:
        tool_name = result.get("_tool_name", "unknown")
        elapsed = result.get("_elapsed_s", 0)

        if tool_name == "run_module":
            module = result.get("module_name", "unknown")
            success = result.get("success", False)
            status = "completed successfully" if success else "failed"
            lines.append(f"### {module} — {status} ({elapsed:.3f}s)\n")
            if result.get("error"):
                lines.append(f"**Error:** {result['error']}\n")
            else:
                fields = result.get("fields_stats", {})
                if fields:
                    lines.append("| Field | Min | Max | Mean |")
                    lines.append("|-------|-----|-----|------|")
                    for fname, stats in fields.items():
                        lines.append(f"| {fname} | {stats['min']:.4g} | {stats['max']:.4g} | {stats['mean']:.4g} |")
                    lines.append("")

        elif tool_name == "sweep_parameter":
            lines.append(f"### Parameter Sweep — {result.get('param_path', '')} ({elapsed:.3f}s)\n")
            sweep_results = result.get("results", [])
            if sweep_results:
                lines.append(f"Swept over {len(sweep_results)} values: {result.get('values', [])}\n")
                # Build comparison table for first field
                first_fields = sweep_results[0].get("fields", {})
                if first_fields:
                    fname = list(first_fields.keys())[0]
                    lines.append(f"| {result.get('param_path', 'param')} | {fname} min | {fname} max | {fname} mean |")
                    lines.append("|------|------|------|------|")
                    for sr in sweep_results:
                        if sr.get("success") and fname in sr.get("fields", {}):
                            fs = sr["fields"][fname]
                            lines.append(f"| {sr['param_value']:.4g} | {fs['min']:.4g} | {fs['max']:.4g} | {fs['mean']:.4g} |")
                    lines.append("")

        elif tool_name == "optimize_parameter":
            lines.append(f"### Optimization — {result.get('param_path', '')} ({elapsed:.3f}s)\n")
            lines.append(f"- **Optimal value:** {result.get('optimal_param_value', 'N/A'):.6g}")
            lines.append(f"- **Optimal metric ({result.get('objective_field', '')}):** {result.get('optimal_metric', 'N/A'):.6g}")
            lines.append(f"- **Objective:** {result.get('objective', 'maximize')}")
            lines.append(f"- **Search bounds:** {result.get('bounds', [])}")
            lines.append(f"- **Evaluations:** {len(result.get('evaluations', []))}\n")

        elif tool_name == "compare_scenarios":
            lines.append(f"### Scenario Comparison ({elapsed:.3f}s)\n")
            comp = result.get("comparison", [])
            if comp:
                all_fields = set()
                for c in comp:
                    all_fields.update(c.get("fields", {}).keys())
                if all_fields:
                    fname = list(all_fields)[0]
                    lines.append(f"| Scenario | {fname} min | {fname} max | {fname} mean |")
                    lines.append("|----------|------|------|------|")
                    for c in comp:
                        if c.get("success") and fname in c.get("fields", {}):
                            fs = c["fields"][fname]
                            lines.append(f"| {c['label']} | {fs['min']:.4g} | {fs['max']:.4g} | {fs['mean']:.4g} |")
                    lines.append("")

        elif tool_name == "run_uncertainty_quantification":
            lines.append(f"### Uncertainty Quantification ({elapsed:.3f}s)\n")
            stats = result.get("statistics", {})
            if stats:
                lines.append("| Metric | Mean | Std | CV% | 90% CI |")
                lines.append("|--------|------|-----|-----|--------|")
                for key, s in stats.items():
                    lines.append(f"| {key} | {s['mean']:.4g} | {s['std']:.4g} | {s['cv_percent']:.1f}% | [{s['p5']:.4g}, {s['p95']:.4g}] |")
                lines.append("")

        elif tool_name == "list_modules":
            lines.append("### Available Modules\n")
            for m, info in result.get("modules", {}).items():
                lines.append(f"- **{m}**: {info.get('description', '')}")
            lines.append("")

        elif tool_name == "describe_module":
            lines.append(f"### Module: {result.get('module_name', '')}\n")
            lines.append(f"- **Domain:** {result.get('domain', '')}")
            lines.append(f"- **Description:** {result.get('description', '')}")
            lines.append(f"- **Config keys:** {json.dumps(result.get('config_keys', {}))}")
            lines.append("")

    # Extract structured sections from programmatic insights
    if insights and insights != "No quantitative insights extracted.":
        # Split insights into sections
        interpretation_lines = []
        validation_lines = []
        recommendation_lines = []
        other_lines = []

        for line in insights.split("\n"):
            if line.startswith("**Physics interpretation:**"):
                interpretation_lines.append(line.replace("**Physics interpretation:** ", "- "))
            elif line.startswith("**Design feedback:**"):
                recommendation_lines.append(line.replace("**Design feedback:** ", "- "))
            elif line.startswith("**Model limits:**"):
                recommendation_lines.append(line.replace("**Model limits:** ", "- *Model limits:* "))
            elif line.startswith("**Suggested follow-ups:**"):
                recommendation_lines.append(line.replace("**Suggested follow-ups:** ", "- *Next steps:* "))
            elif "Validation against references" in line or "PASS" in line or "MARGIN" in line:
                validation_lines.append(line)
            elif line.strip():
                other_lines.append(line)

        if other_lines:
            lines.append("### Key Findings\n")
            lines.append("\n".join(other_lines))
            lines.append("")

        if interpretation_lines:
            lines.append("### Physics Interpretation\n")
            lines.append("\n".join(interpretation_lines))
            lines.append("")

        if validation_lines:
            lines.append("### Validation\n")
            lines.append("\n".join(validation_lines))
            lines.append("")

        if recommendation_lines:
            lines.append("### Design Recommendations\n")
            lines.append("\n".join(recommendation_lines))
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Triality Agentic Simulator",
    version="2.0.0",
    description="Agentic physics reasoning engine with LLM planning, tool execution, and streaming results.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
# Keep legacy /static mount for backward compatibility
_legacy_static = BASE_DIR / "static"
if _legacy_static.is_dir():
    app.mount("/static", StaticFiles(directory=_legacy_static), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "triality_modules": available_runtime_modules(),
        "triality_templates": available_runtime_templates(),
    }


@app.get("/api/catalog")
def catalog() -> Dict[str, Any]:
    return {
        "scenarios": SCENARIOS,
        "modules": {
            m: {
                "description": TRIALITY_MODULES_INFO.get(m, {}).get("description", ""),
                "domain": TRIALITY_MODULES_INFO.get(m, {}).get("domain", ""),
                "config_keys": TRIALITY_MODULES_INFO.get(m, {}).get("config_keys", {}),
                "defaults": TRIALITY_MODULES_INFO.get(m, {}).get("defaults", {}),
            }
            for m in available_runtime_modules()
        },
        "templates": available_runtime_templates(),
        "tools": list(TOOL_REGISTRY.keys()),
        "capabilities": [
            {"name": "Agentic Planning", "status": "Full", "icon": "brain", "description": "LLM-powered planning. Understands your engineering problem, selects the right physics module, and configures analyses."},
            {"name": "Physics Execution", "status": "Full", "icon": "atom", "description": "8 tool types: run, sweep, optimize, compare, UQ, chain, describe, list."},
            {"name": "Streaming Results", "status": "Full", "icon": "radio", "description": "Server-Sent Events stream thinking, tool calls, and results in real time."},
            {"name": "Autonomous Reflection", "status": "Full", "icon": "sparkles", "description": "Agent reflects on results and autonomously runs follow-up analysis when valuable."},
        ],
        "default_model": os.getenv("TRIALITY_LLM_MODEL", DEFAULT_MODEL),
    }


@app.post("/api/agent")
async def agent_endpoint(request: AgentRequest):
    """SSE streaming endpoint for the agentic loop."""
    token = request.replicate_api_token.strip()
    if not token and not DEFAULT_REPLICATE_TOKEN:
        return JSONResponse(status_code=422, content={"detail": "replicate_api_token is required — LLM is mandatory for the agent."})

    # If scenario_id provided, use its prompt
    prompt = request.prompt
    if request.scenario_id:
        for s in SCENARIOS:
            if s["id"] == request.scenario_id:
                prompt = prompt or s["prompt"]
                break

    _SENTINEL = object()

    def _next_or_sentinel(gen):
        """Wrapper around next() that returns a sentinel instead of raising
        StopIteration, which cannot propagate through asyncio futures."""
        try:
            return next(gen)
        except StopIteration:
            return _SENTINEL

    async def event_stream():
        try:
            loop = asyncio.get_event_loop()
            gen = _run_agent_turn(prompt, token, request.llm_model, request.scenario_id,
                                     conversation_history=request.conversation_history)
            while True:
                # Run the synchronous generator's next() in a thread so the
                # event loop stays free to flush each SSE chunk to the client.
                event = await loop.run_in_executor(None, _next_or_sentinel, gen)
                if event is _SENTINEL:
                    break
                yield event
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Keep legacy endpoint for backward compatibility
@app.post("/api/simulations")
def run_simulation(request: AnalysisRequest) -> Dict[str, Any]:
    """Legacy synchronous endpoint."""
    plan = _heuristic_plan(request.prompt, request.scenario_id)
    results = []
    started = time.perf_counter()
    for tc in plan.get("tool_calls", []):
        result = execute_tool_call(tc["tool"], tc.get("args", {}))
        result["_tool_name"] = tc["tool"]
        results.append(result)
    return {
        "success": all(not r.get("error") for r in results),
        "prompt": request.prompt,
        "plan": plan,
        "results": results,
        "elapsed_time_s": time.perf_counter() - started,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("triality_app.main:app", host="0.0.0.0", port=8510, reload=False)
