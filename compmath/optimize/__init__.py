"""
Optimize Submodule
==================

The `optimize` submodule provides robust numerical methods for solving nonlinear equations,
solving tridiagonal systems, and automatically locating intervals where a function changes sign.
These tools are fundamental in scientific computing and numerical analysis, where analytical solutions
are difficult or impossible to obtain.

This submodule includes classical root-finding algorithms and linear system solvers, such as:
- Secant method (Secant-like, fixed left point)
- Newton’s method (based on tangents and symbolic derivatives)
- Tridiagonal matrix algorithm (Thomas algorithm)
- Interval detection for function sign change

Each method is implemented with support for arbitrary precision using Python’s `decimal.Decimal`,
and allows flexible control over stopping criteria and iteration limits.

Features:
---------
- **Secant Method**:
  - Iterative method for solving nonlinear equations.
  - Uses a fixed endpoint and secant approximation to find roots.
  - Suitable for monotonic functions with known sign change.

- **Newton's Method**:
  - Requires first derivative of the function.
  - Fast convergence near simple roots.
  - Utilizes symbolic differentiation with `sympy`.

- **Tridiagonal Matrix Algorithm**:
  - Efficient solver for systems with tridiagonal matrices.
  - Common in numerical solutions of differential equations and spline interpolation.

- **Sign Change Interval Finder**:
  - Automatically detects subintervals where the function changes sign.
  - Useful for initializing root-finding methods.
  - Supports both `float` and `Decimal` precision.

Modules:
--------
- `fsolve`: Contains implementations of the secant and Newton's methods.
- `sign_change_finder`: Provides automatic interval detection for root isolation.
- `tridiagonal_alg`: Implements the Thomas algorithm for tridiagonal systems.

Usage:
------
Each method can be applied to symbolic or numeric functions and supports customizable precision
and convergence settings. The module is suitable for use in educational, scientific, and engineering contexts.

Example:
--------
```python
from sympy import symbols
from optimize import secant_method, newton_method, find_sign_change_interval

x = symbols('x')
f = lambda x: x**3 - x - 1

# Find interval with sign change
a, b = find_sign_change_interval(f, search_range=(1, 2))

# Use secant method
root_secant = secant_method(f, a, b)
print("Root (secant method):", root_secant)

# Use Newton's method
root_newton = newton_method(f, x0=1.5)
print("Root (Newton's method):", root_newton)
"""

from .fsolve import *
from .sign_change_finder import *
from .tridiagonal_alg import *
