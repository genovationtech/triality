"""Modal analysis for structural dynamics."""
import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class ModeShape:
    frequency_Hz: float
    mode_vector: np.ndarray
    damping_ratio: float = 0.05

@dataclass
class StructuralModel:
    mass_matrix: np.ndarray  # M
    stiffness_matrix: np.ndarray  # K
    damping_matrix: np.ndarray = None  # C (optional)

class ModalSolver:
    """Solve eigenvalue problem for natural frequencies and mode shapes."""
    
    def __init__(self, model: StructuralModel):
        self.model = model
        self.modes: List[ModeShape] = []
    
    def solve_eigen(self, n_modes: int = None) -> List[ModeShape]:
        """
        Solve eigenvalue problem: (K - ω²·M)·φ = 0
        
        Returns:
            List of ModeShape objects
        """
        M = self.model.mass_matrix
        K = self.model.stiffness_matrix
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(K, M)
        
        # Natural frequencies: ω = √λ
        omega = np.sqrt(np.abs(eigenvalues))
        freq_Hz = omega / (2 * np.pi)
        
        # Sort by frequency
        idx = np.argsort(freq_Hz)
        freq_Hz = freq_Hz[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Extract requested modes
        if n_modes is None:
            n_modes = len(freq_Hz)
        
        self.modes = []
        for i in range(min(n_modes, len(freq_Hz))):
            mode = ModeShape(
                frequency_Hz=freq_Hz[i],
                mode_vector=eigenvectors[:, i],
                damping_ratio=0.05  # Default 5%
            )
            self.modes.append(mode)
        
        return self.modes
    
    def modal_mass(self, mode_idx: int) -> float:
        """Calculate modal mass: M* = φᵀ·M·φ"""
        phi = self.modes[mode_idx].mode_vector
        M = self.model.mass_matrix
        return phi.T @ M @ phi
    
    def participation_factor(self, mode_idx: int, direction: np.ndarray) -> float:
        """
        Calculate modal participation factor.
        
        Γ = φᵀ·M·r / (φᵀ·M·φ)
        
        where r is influence vector (direction of excitation)
        """
        phi = self.modes[mode_idx].mode_vector
        M = self.model.mass_matrix
        
        numerator = phi.T @ M @ direction
        denominator = phi.T @ M @ phi
        
        return numerator / denominator if denominator != 0 else 0.0
