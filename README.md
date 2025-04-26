# CompMath - Computational Mathematics Library

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

CompMath is a Python library for complex mathematical calculations in the field of computational mathematics. It provides robust implementations of numerical methods, linear algebra operations, function interpolation, numerical integration/differentiation, and differential equation solvers.

## Features

### Core Functionality

- **Numerical Methods**: Advanced algorithms for solving equations
- **Linear Algebra**: Matrix and vector operations
- **Interpolation & Approximation**: Multiple methods for function approximation
- **Numerical Calculus**: Integration and differentiation techniques
- **ODE Solvers**: Methods for ordinary differential equations

### Precision and Error Handling

- High-precision arithmetic using `Decimal`
- Comprehensive error analysis tools
- Condition number computation
- Significant digit management

## Installation

```bash
pip install compmath
```

## Modules

### 1. Interpolation

Powerful interpolation methods with error estimation:

```python
from compmath.interpolation import lagrange, newton, gauss

# Lagrange interpolation
x_points = [0, 1, 2, 3]
y_points = [1, 2, 0, 5]
interp_value = lagrange(1.5, x_points, y_points)
```

**Available Methods**:

- Lagrange interpolation
- Newton interpolation (divided/forward/backward differences)
- Gauss interpolation (forward/backward)
- Stirling and Bessel interpolation
- Finite difference tables

### 2. Math Errors

Comprehensive error analysis toolkit:

```python
from compmath.math_errors import absolute_error, round_to

# Error calculation
abs_err = absolute_error(3.14159, 3.14)

# Precision rounding
rounded = round_to(3.14159).sd(3)  # 3 significant digits
```

**Features**:

- Absolute/relative error computation
- Significant digit analysis
- Condition number calculation
- Approximate number arithmetic

### 3. Utilities

Essential helper functions:

```python
from compmath.utils import to_decimal, factorial

# High-precision conversion
decimal_val = to_decimal(3.14159)

# Factorial computation
fact_10 = factorial(10)
```

**Capabilities**:

- Type conversion to `Decimal`
- Mathematical utilities
- Array processing helpers

## Examples

### Interpolation Example

```python
import numpy as np
from compmath.interpolation import lagrange_deriv

# Compute derivative of interpolated function
xp = np.array([0, 1, 2, 3])
yp = np.array([1, 2, 0, 5])
derivative = lagrange_deriv(1.5, 1, xp, yp)  # 1st derivative at x=1.5
```

### Error Analysis Example

```python
from compmath.math_errors import cond_nums

# Compute condition number
def f(x):
    return x**2

condition = cond_nums(f, x=2).rel()  # Relative condition number
```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Key Features:

1. **Professional Structure**: Clear sections for features, installation, usage examples, and documentation
2. **Code Examples**: Ready-to-run examples for each major module
3. **Badges**: Visual indicators for Python version and license
4. **Comprehensive Coverage**: Incorporates all functionality from your docstrings
5. **Modern Formatting**: Clean Markdown with proper code blocks

This README effectively communicates your library's capabilities while maintaining a professional appearance that would appeal to both academic and industrial users. The examples demonstrate practical usage of the key features mentioned in your docstrings.
