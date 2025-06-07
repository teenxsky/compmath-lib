"""
compmath Package (Computational Mathematics)
===========================================

The `compmath` package is a modular library for computational mathematics,
offering reliable and high-precision tools for solving fundamental numerical problems.
It is structured into submodules, each targeting a core area of numerical analysis:
error analysis, interpolation, numerical integration, and equation solving.

Whether you're a student, educator, or researcher, `compmath` aims to provide
transparent and well-structured implementations of classical methods with an emphasis
on correctness, precision, and educational clarity.

Submodules:
-----------
- **math_errors**:
  - Tools for error estimation, number rounding, digit analysis, condition number computation,
    and arithmetic with approximate numbers.

- **interpolation**:
  - Methods for polynomial interpolation on discrete data points, including Lagrange,
    Newton (divided and difference-based), Gauss, Stirling, and Bessel interpolation.

- **integration**:
  - Classical and advanced quadrature rules for definite integration of sampled data,
    such as Simpson’s rule, Weddle’s rule, and general Newton–Cotes formulas.

- **optimize**:
  - Methods for solving nonlinear equations and tridiagonal linear systems:
    - Chord method
    - Newton's method
    - Tridiagonal matrix solver (Thomas algorithm)
    - Automatic detection of sign-changing intervals

- **utils**:
  - Utility functions for common numerical operations, such as:
    - Type casting between floats and Decimals
    - Symbolic to numeric function conversion
    - Root formatting and tabulation

Design Features:
----------------
- High-precision arithmetic with `Decimal` to minimize floating-point errors.
- Support for both symbolic (via `SymPy`) and numerical computations.
- Modular structure allows for extensibility and integration into larger systems.

Usage Example:
--------------
```python
from compmath.interpolation import lagrange
from compmath.integration import simpson
from compmath.math_errors import absolute_error
from compmath.optimize import chord_method
from compmath.utils import to_decimal

# Interpolate a function value using Lagrange
x = 1.5
xp = [1.0, 2.0, 3.0]
yp = [2.0, 4.0, 6.0]
y_interp = lagrange(x, xp, yp)
print(f'Interpolated value at x={x}: {y_interp}')

# Integrate a function using Simpson's rule
xp = [0, 0.5, 1.0]
yp = [1, 1.25, 1.5]
area = simpson.quad(xp, yp)
print(f'Approximate integral: {area}')

# Solve equation using chord method
from sympy import symbols
x_sym = symbols('x')
f = x_sym**3 - x - 1
root = chord_method(f, a=1.0, b=2.0)
print(f'Root: {root}')

# Compute absolute error
approx = 3.14159
exact = 3.14
err = absolute_error(approx, exact)
print(f'Absolute error: {err}')
"""

import importlib as _importlib

MODULES = ['math_errors', 'interpolation', 'integration', 'optimize', 'utils']


def __getattr__(name):
    if name in MODULES:
        return _importlib.import_module(f'compmath.{name}')
    else:
        raise AttributeError(f"Module 'compmath' has no attribute '{name}'")
