"""Basic 1D Poisson equation example"""

from triality import *
import numpy as np

# Problem: u''(x) = -1 on [0,1], u(0) = u(1) = 0
# Exact solution: u(x) = x(1-x)/2

u = Field("u")
eq = laplacian(u) == -1
domain = Interval(0, 1)
bc = {'left': 0, 'right': 0}

# Solve
sol = solve(eq, domain, bc=bc, resolution=100)

# Check error
x = sol.grid
u_exact = x * (1 - x) / 2
error = np.abs(sol.u - u_exact)
print(f"\nMax error: {np.max(error):.2e}")
print(f"L2 error:  {np.sqrt(np.mean(error**2)):.2e}")

# Evaluate at specific points
print(f"\nu(0.25) = {sol(0.25):.6f}  (exact: 0.09375)")
print(f"u(0.50) = {sol(0.5):.6f}  (exact: 0.125000)")
print(f"u(0.75) = {sol(0.75):.6f}  (exact: 0.09375)")

# Plot
sol.plot()
