import numpy as np
from ..utils import to_decimal
from .difftabs import difftabs
from typing import Union, Literal
from decimal import Decimal, getcontext


getcontext().prec = 20
__all__ = ["stirling", "bessel"]


def stirling(
    x: Union[Decimal, float, int, str],
    xp: np.ndarray,
    yp: np.ndarray,
    fd: np.ndarray = None,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Stirling interpolation for equally spaced nodes.
    Suitable when the interpolation point is near the center
    of the table and the number of points is odd.

    Args:
        x (Union[Decimal, float, int, str]): Interpolation point.
        xp (np.ndarray): Equally spaced x-values.
        yp (np.ndarray): Corresponding y-values.
        fd (np.ndarray, optional): Precomputed finite difference table.
        return_type (Literal["Decimal", "float"], optional): Return type. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: Interpolated value at x.

    Raises:
        ValueError: If xp and yp lengths mismatch, fd is invalid,
                    or number of intervals is not odd.
    """
    x = to_decimal(x)
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    if len(xp) != len(yp):
        raise ValueError("xp and yp must be of equal length.")
    if fd is None:
        fd = difftabs.fin(yp)
    elif len(fd) != len(yp):
        raise ValueError("fd must have the same length as yp.")

    n = len(xp) - 1
    if (n + 1) % 2 == 0:
        raise ValueError("Stirling method requires an even number of points (odd n).")

    m = n // 2
    h = xp[1] - xp[0]
    t = (x - xp[m]) / h

    val = fd[0][m]
    fact = to_decimal(1)
    t_prod = t

    for k in range(1, n + 1):
        fact *= k
        i = m - (k // 2)

        if k % 2 == 1:
            dk = (fd[k][i - 1] + fd[k][i]) / 2
            t_prod *= t**2 - (k // 2) ** 2 if k > 1 else 1
        else:
            dk = fd[k][i]
            t_prod *= t

        val += (t_prod * dk) / fact

    return val if return_type == "Decimal" else np.float64(val)


def bessel(
    x: Union[Decimal, float, int, str],
    xp: np.ndarray,
    yp: np.ndarray,
    fd: np.ndarray = None,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Bessel interpolation for equally spaced nodes.
    Recommended when the interpolation point lies halfway between
    two central data points (especially with even number of intervals).

    Args:
        x (Union[Decimal, float, int, str]): Interpolation point.
        xp (np.ndarray): Equally spaced x-values.
        yp (np.ndarray): Corresponding y-values.
        fd (np.ndarray, optional): Precomputed finite difference table.
        return_type (Literal["Decimal", "float"], optional): Return type. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: Interpolated value at x.

    Raises:
        ValueError: If input data is invalid or n is not odd.
    """
    x = to_decimal(x)
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    n = len(xp) - 1
    if (n + 1) % 2 == 1:
        raise ValueError("Bessel method requires an even number of points (odd n).")

    if len(xp) != len(yp):
        raise ValueError("xp and yp must be of equal length.")
    if fd is None:
        fd = difftabs.fin(yp)
    elif len(fd) != len(yp):
        raise ValueError("fd must have the same length as yp.")

    m = n // 2
    h = xp[1] - xp[0]
    t = (x - xp[m]) / h

    val = (fd[0][m] + fd[0][m + 1]) / 2
    fact = to_decimal(1)
    t_prod = to_decimal(1)

    for k in range(1, n + 1):
        fact *= k
        i = m - (k // 2)

        if k % 2 == 1:
            dk = fd[k][i]
            p = t_prod * (t - Decimal("0.5")) * dk
        else:
            dk = (fd[k][i] + fd[k][i + 1]) / 2
            t_prod = to_decimal(1)
            for j in range(1, k // 2):
                t_prod *= t**2 - j**2
            t_prod *= t * (t - k // 2)
            p = t_prod * dk

        val += p / fact

    return val if return_type == "Decimal" else np.float64(val)
