from decimal import Decimal
from typing import Callable, Literal, Union

import numpy as np

from ..utils import to_decimal

__all__ = ['find_sign_change_interval']


def find_sign_change_interval(
    f: Callable[[Union[np.float64, np.ndarray]], Union[np.float64, np.ndarray]],
    search_range: tuple[
        Union[Decimal, float, int, str], Union[Decimal, float, int, str]
    ],
    step: np.float64 = 0.01,
    return_type: Literal['Decimal', 'float'] = 'float',
) -> tuple[Union[np.float64, Decimal], Union[np.float64, Decimal]]:
    """
    Finds an interval [a, b] where f(x) changes sign (i.e., f(a)*f(b) < 0).

    Args:
        f (Callable): Regular function using numpy methods.
        search_range (tuple): Search interval (start, end).
        step (np.float64): Iteration step.
        return_type (str): Return type: 'float' or 'Decimal'.

    Returns:
        tuple: Interval (a, b) where f changes sign.

    Raises:
        ValueError: If the sign doesn't change on any subinterval.
    """
    a, b = np.float64(search_range[0]), np.float64(search_range[1])
    x_vals = np.arange(a, b, step)

    for i in range(len(x_vals) - 1):
        x0, x1 = x_vals[i], x_vals[i + 1]
        y0, y1 = f(x0), f(x1)
        if y0 * y1 < 0:
            if return_type == 'Decimal':
                return to_decimal(x0), to_decimal(x1)
            return np.float64(x0), np.float64(x1)

    raise ValueError('No interval with a sign change found within the specified range.')
