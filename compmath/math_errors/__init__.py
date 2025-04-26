"""
Math Errors Submodule
======================

The `math_errors` submodule provides tools and utilities for analyzing and managing numerical errors in
computational mathematics. This includes handling absolute and relative errors, rounding numbers to
specific digits, and analyzing significant, valid, and doubtful digits. Additionally, it provides
methods for calculating condition numbers and working with approximate numbers.

This submodule is essential for ensuring the accuracy and reliability of numerical computations,
especially in fields like numerical analysis, scientific computing, and engineering.

Features:
---------
- **Error Calculations**:
  - Compute absolute and relative errors using various methods.
  - Support for error propagation in arithmetic operations.

- **Rounding**:
  - Round numbers to a specified number of significant, valid, or doubtful digits.
  - High-precision rounding using the `Decimal` type.

- **Digit Analysis**:
  - Analyze significant, valid, and doubtful digits of a number.
  - Determine the precision of a number based on its absolute error.

- **Condition Numbers**:
  - Compute absolute and relative condition numbers for functions.
  - Measure the sensitivity of a function's output to changes in its input.

- **Approximate Numbers**:
  - Represent numbers with associated absolute and relative errors.
  - Perform arithmetic operations with proper error propagation.

Modules:
--------
- `round`: Provides utilities for rounding numbers to significant, valid, or doubtful digits.
- `errors`: Contains functions for calculating absolute and relative errors.
- `digits_analysis`: Analyzes significant, valid, and doubtful digits of a number.
- `cond_nums`: Computes absolute and relative condition numbers for functions.
- `approx_num`: Represents approximate numbers with error bounds and supports arithmetic operations.

Usage:
------
The submodule is designed to be flexible and efficient, supporting both high-precision arithmetic
(using `Decimal`) and standard floating-point operations. It is particularly useful for applications
where numerical accuracy is critical.

Example:
--------
```python
from compmath.math_errors import absolute_error, relative_error, round_to, digits_analysis, cond_nums, ApproxNum

# Example 1: Calculate absolute and relative errors
value = 3.14159
exact_value = 3.14
abs_err = absolute_error(value, exact_value)
rel_err = relative_error(value, exact_value)
print(f"Absolute Error: {abs_err}, Relative Error: {rel_err}")

# Example 2: Round to significant digits
rounded_value = round_to(3.14159).sd(num_digits=3)
print(f"Rounded to 3 significant digits: {rounded_value}")

# Example 3: Analyze digits
analysis = digits_analysis(3.14159)
significant_digits = analysis.sd()
print(f"Significant Digits: {significant_digits}")

# Example 4: Compute condition numbers
def f(x):
    return x**2

condition = cond_nums(f, x=2)
abs_cond = condition.abs()
rel_cond = condition.rel()
print(f"Absolute Condition Number: {abs_cond}, Relative Condition Number: {rel_cond}")

# Example 5: Work with approximate numbers
approx = ApproxNum(3.14159, abs_err=0.001)
print(f"Approximate Number: {approx}")
"""

from .round import *
from .errors import *
from .cond_nums import *
from .approx_num import *
from .digits_analysis import *
