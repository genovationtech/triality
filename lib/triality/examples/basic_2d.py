"""Basic 2D Poisson equation example"""

from triality import *

# Problem: ∇²u = -2 on unit square, u = 0 on boundary

u = Field("u")
eq = laplacian(u) == -2
domain = Square(1.0)
bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

# Solve
sol = solve(eq, domain, bc=bc, resolution=50)

# Evaluate at center
u_center = sol(0.5, 0.5)
print(f"\nSolution at center: u(0.5, 0.5) = {u_center:.6f}")

# Plot
sol.plot()
