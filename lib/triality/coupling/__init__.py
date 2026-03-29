"""
Multi-Physics Coupling Modules

Tier A (platform-critical):
    thermal_structural      -- Thermal → Structural (thermo-mechanical)
    aero_structural         -- Aero loads → Structural (aeroelasticity)
    cfd_cht                 -- CFD → Conjugate Heat Transfer
    neutronics_thermal_hydraulics -- Neutronics ↔ Thermal-Hydraulics

Tier B (deep-chain couplings):
    cfd_reacting_flows      -- CFD turbulence → Reacting flows
    combustion_radiation    -- Combustion → Soot → Radiation transport
"""
