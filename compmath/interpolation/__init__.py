"""
Interpolation Submodule
=======================

The `interpolation` submodule provides a collection of methods and utilities for performing interpolation
on discrete data points. Interpolation is a fundamental technique in numerical analysis that estimates
the value of a function at a given point based on known data points.

This submodule includes implementations of various interpolation methods, such as:
- Lagrange interpolation
- Newton interpolation (divided, forward, and backward differences)
- Gauss interpolation (forward and backward)
- Stirling and Bessel interpolation for equally spaced nodes
- Cubic Hermite spline interpolation

Each method is designed to handle specific use cases, such as unevenly spaced nodes, equally spaced nodes,
or interpolation near the center of the data range.

Features:
---------
- **Lagrange Interpolation**:
  - Computes the interpolating polynomial using the Lagrange formula.
  - Supports symbolic differentiation for estimating derivatives of the interpolating polynomial.
  - Includes error estimation for the interpolation.

- **Newton Interpolation**:
  - Divided differences for unevenly spaced nodes.
  - Forward and backward differences for equally spaced nodes.
  - Efficient computation of the interpolating polynomial.

- **Gauss Interpolation**:
  - Forward and backward methods for equally spaced nodes.
  - Suitable for interpolation near the center of the data range.

- **Stirling and Bessel Interpolation**:
  - Stirling interpolation for odd numbers of intervals (even number of points).
  - Bessel interpolation for even numbers of intervals (odd number of points).
  - Designed for equally spaced nodes.

- **Cubic Hermite Spline Interpolation**:
  - Implemented via the `HSpline` class in the `spline` module.
  - Constructs a smooth spline with continuous first derivatives.
  - Supports evaluation of spline values, derivatives (1st to 3rd order), and definite integrals.
  - Precision-friendly: works with `Decimal` for high-accuracy computations.

- **Finite Difference Tables**:
  - Utility functions for generating finite difference tables used in Newton, Gauss, Stirling, and Bessel methods.

Modules:
--------
- `lagrange_f`: Contains functions for Lagrange interpolation and its derivatives.
- `newton_f`: Implements Newton interpolation methods (divided, forward, and backward differences).
- `gauss_f`: Provides Gauss forward and backward interpolation methods.
- `odd_even_f`: Implements Stirling and Bessel interpolation methods for equally spaced nodes.
- `difftabs`: Utility functions for generating finite difference tables.
- `spline`: Contains the `HSpline` class for cubic Hermite spline interpolation.

Usage:
------
The submodule is designed to be flexible and efficient, supporting both symbolic and numerical computations.
It uses `Decimal` for high-precision arithmetic and `SymPy` for symbolic differentiation where needed.

Example:
--------
```python
import numpy as np
from compmath.interpolation import lagrange, newton, gauss, difftabs, HSpline

# Define known data points
xp = np.array([0, 1, 2, 3], dtype=float)
yp = np.array([1, 2, 0, 5], dtype=float)

# Lagrange interpolation
x = 1.5
y_lagrange = lagrange(x, xp, yp)
print(f'Lagrange interpolation at x={x}: {y_lagrange}')

# Newton interpolation (divided differences)
y_newton = newton.poly(x, xp, yp)
print(f'Newton interpolation at x={x}: {y_newton}')

# Gauss forward interpolation
y_gauss = gauss.fwd(x, xp, yp)
print(f'Gauss forward interpolation at x={x}: {y_gauss}')

# Finite difference table
fd_table = difftabs.fin(yp)
print('Finite difference table:')
print(fd_table)

# Cubic Hermite Spline interpolation
spline = hspline(xp, yp)
y_spline = spline.interpolate(x)
print(f'Cubic spline interpolation at x={x}: {y_spline}')"""

from .difftabs import *
from .gauss_f import *
from .lagrange_f import *
from .newton_f import *
from .odd_even_f import *
from .splain import *
