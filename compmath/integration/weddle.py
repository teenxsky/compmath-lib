from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np

from ..utils import to_decimal

getcontext().prec = 20
__all__ = ['weddles']


def weddles(
    xp: np.ndarray,
    yp: np.ndarray,
    return_type: Literal['Decimal', 'float'] = 'float',
) -> Union[Decimal, np.float64]:
    """
    Approximate the integral of a function using Weddle's rule.

    Weddle's rule is a higher-order Newton–Cotes numerical integration formula
    that requires the number of intervals to be divisible by 6. It approximates
    the integral over each group of 6 subintervals (7 nodes) with a weighted sum
    of the function values.

    For each segment of 7 points [x₀, x₁, ..., x₆], the formula is:
        ∫ f(x) dx ≈ (3h / 10) * [f₀ + 5f₁ + f₂ + 6f₃ + f₄ + 5f₅ + f₆],
    where h = (x₆ - x₀) / 6 and fᵢ = f(xᵢ)

    Args:
        xp (np.ndarray): A 1D array of x-values. Length must be 6k + 1.
        yp (np.ndarray): A 1D array of corresponding y-values f(x) at each xp.
        return_type (Literal["Decimal", "float"], optional): Specifies the return type.
            Use "Decimal" for high-precision output or "float" for NumPy float. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: Approximate value of the definite integral using Weddle's rule.

    Raises:
        ValueError: If the number of intervals (len(xp) - 1) is not divisible by 6.
    """
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    n = len(xp) - 1
    if n % 6 != 0:
        raise ValueError(
            "Number of intervals must be divisible by 6 for Weddle's rule"
        )

    res = to_decimal(0)
    for i in range(0, n, 6):
        h = (xp[i + 6] - xp[i]) / 6
        coeffs = [1, 5, 1, 6, 1, 5, 1]
        segment = [yp[i + j] for j in range(7)]
        res += (3 * h / 10) * sum(c * y for c, y in zip(coeffs, segment))

    return np.float64(res) if return_type == 'float' else to_decimal(res)
