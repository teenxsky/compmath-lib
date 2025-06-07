from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np

from ..utils import to_decimal

getcontext().prec = 20
__all__ = ['gauss']


def gauss(
    xp: np.ndarray,
    yp: np.ndarray,
    weights: np.ndarray = None,
    a: Union[float, Decimal] = -1,
    b: Union[float, Decimal] = 1,
    return_type: Literal['Decimal', 'float'] = 'float',
) -> Union[Decimal, float]:
    """
    Approximate the integral using Gauss–Legendre quadrature with provided x and y values.

    Args:
        xp (np.ndarray): Points (nodes) where the function is evaluated (usually in [-1, 1]).
        yp (np.ndarray): Function values at each node.
        weights (np.ndarray, optional): Optional Gauss weights for the nodes. If None, computed automatically.
        a (Union[float, Decimal], optional): Lower bound of integration. Default is -1.
        b (Union[float, Decimal], optional): Upper bound of integration. Default is 1.
        return_type (Literal["Decimal", "float"], optional): Result type. Defaults to "float".

    Returns:
        Union[Decimal, float]: Approximate integral of the function on [a, b].
    """
    xp = to_decimal(xp)
    yp = to_decimal(yp)
    a, b = Decimal(str(a)), Decimal(str(b))

    n = len(xp)

    if weights is None:
        _, w = np.polynomial.legendre.leggauss(n)
        weights = to_decimal(w)

    else:
        weights = to_decimal(weights)

    half = (b - a) / 2
    integral = sum(weights[i] * yp[i] for i in range(n))
    result = half * integral

    return result if return_type == 'Decimal' else float(result)
