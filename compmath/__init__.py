"""CompMath (Computational Mathematics) - Library for complex mathematical calculations

This library provides tools to perform calculations in the field
computing mathematics, including numerical methods, linear algebra,
Interpolation, integration and solution of differential equations.

Basic possibilities:
- Numerical methods for solving equations
- Linear algebra (matrices, vectors, operations)
- Interpolation and approximation of functions
- Numerical integration and differentiation
- Solution of ordinary differential equations (ode)

Modules:
- `math_errors`: a module for calculation errors.
- `interpolation`: a module for interpolation methods.
"""

import importlib as _importlib


MODULES = ["math_errors", "interpolation"]


def __getattr__(name):
    if name in MODULES:
        return _importlib.import_module(f"compmath.{name}")
    else:
        raise AttributeError(f"Module 'compmath' has no attribute '{name}'")
