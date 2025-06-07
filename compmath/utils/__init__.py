"""
Utils Submodule
================

The `utils` submodule provides utility functions and tools that are commonly used across the library.
These utilities include mathematical operations, type conversions, and helper functions to simplify
and standardize computations in numerical analysis and computational mathematics.

This submodule is designed to be lightweight and efficient, offering reusable components that can be
leveraged by other submodules such as `interpolation` and `math_errors`.

Features:
---------
- **Type Conversion**:
  - Convert various data types (e.g., `float`, `int`, `str`, `list`, `np.ndarray`) to `Decimal` for
    high-precision arithmetic.

- **Mathematical Utilities**:
  - Compute the factorial of an integer using efficient algorithms.
  - Provide helper functions for numerical computations.

Modules:
--------
- `calc`: Contains mathematical utility functions, such as factorial computation.
- `tools`: Provides type conversion utilities, such as converting numbers or arrays to `Decimal`.

Usage:
------
The submodule is designed to be flexible and efficient, supporting both high-precision arithmetic
(using `Decimal`) and standard numerical operations. It is particularly useful for applications
where precision and consistency are critical.

Example:
--------
```python
from compmath.utils import to_decimal, factorial

# Example 1: Convert a number to Decimal
value = 3.14159
decimal_value = to_decimal(value)
print(f'Decimal Value: {decimal_value}')

# Example 2: Convert a list of numbers to Decimal
values = [1.1, 2.2, 3.3]
decimal_values = to_decimal(values)
print(f'Decimal Values: {decimal_values}')

# Example 3: Compute the factorial of an integer
n = 5
fact = factorial(n)
print(f'Factorial of {n}: {fact}')
"""

from .calc import *
from .tools import *
