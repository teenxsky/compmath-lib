"""
Integration Submodule
=====================

The `integration` submodule provides a suite of numerical integration methods (quadrature rules)
for approximating the definite integral of a function based on discrete data points.
These methods are essential in scientific computing where analytical integration is
infeasible or impossible.

This submodule includes classical Newton–Cotes formulas and specialized rules such as:
- Rectangle rule (left and right)
- Midpoint rule
- Trapezoidal rule
- Simpson's rule (1/3 and 3/8 variants)
- Newton–Cotes general formula (with arbitrary coefficients)
- Weddle’s rule

Each method is implemented with high-precision arithmetic using the `Decimal` class from Python’s `decimal` module,
and optionally supports returning results in either `Decimal` or NumPy’s `float64` format.

Features:
---------
- **Rectangle Rule**:
  - Left and right variants.
  - Suitable for rough integration estimates with piecewise constant approximation.

- **Midpoint Rule**:
  - Uses the value of the function at the midpoint of each subinterval.
  - Better accuracy than rectangle methods for smooth functions.

- **Trapezoidal Rule**:
  - Approximates the area under the curve using trapezoids.
  - Simple and effective for linear or smooth functions.

- **Simpson’s Rules**:
  - 1/3 rule (requires even number of intervals).
  - 3/8 rule (requires number of intervals divisible by 3).
  - High accuracy for smooth functions due to quadratic or cubic polynomial approximation.

- **Newton–Cotes General Rule**:
  - Dynamically computes weights for any number of nodes.
  - Flexible integration method for arbitrary small node sets.

- **Weddle’s Rule**:
  - Specialized Newton–Cotes rule of degree 6.
  - Requires the number of intervals to be divisible by 6.
  - Very accurate for well-behaved functions.

- **Gaussian Quadrature**:
  - Implements Gaussian quadrature for high-precision integration.
  - Uses Chebyshev nodes for optimal polynomial approximation.

Modules:
--------
- `basic`: Contains rectangle, midpoint, and trapezoidal rules.
- `simpson`: Implements Simpson’s 1/3 and 3/8 rules.
- `newton_cotes`: Defines a general Newton–Cotes integration method.
- `weddle`: Implements Weddle’s rule for degree-6 interpolation.
- `gauss`: Implements Gaussian quadrature for high-precision integration.

Usage:
------
Each function expects arrays of x and y values representing sampled data points
and returns the numerical approximation of the integral over the specified interval.

Example:
--------
```python
import numpy as np
from compmath.integration import rectangle, trapezoid, simpson, weddles

# Define data points
xp = np.array([0, 0.5, 1.0])
yp = np.array([1, 1.25, 1.5])  # Example f(x) = 1 + 0.5x

# Apply different integration methods
print('Rectangle rule:', rectangle(xp, yp, method='left'))
print('Trapezoidal rule:', trapezoid(xp, yp))
print("Simpson's 1/3 rule:", simpson.quad(xp, yp))
print(
    'Weddle’s rule (requires 7 points):',
    weddles(np.linspace(0, 6, 7), np.sin(np.linspace(0, 6, 7))),
)
print('Gaussian quadrature:', gauss(xp, yp))
"""

from .basic import *
from .gauss import *
from .newton_cotes import *
from .simpson import *
from .weddle import *
