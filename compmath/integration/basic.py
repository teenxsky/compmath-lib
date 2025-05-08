import numpy as np
from ..utils import to_decimal
from typing import Union, Literal
from decimal import Decimal, getcontext


getcontext().prec = 20
__all__ = ["rectangle", "midpoint", "trapezoid"]


def rectangle(
    xp: np.ndarray,
    yp: np.ndarray,
    method: Literal["left", "right"] = "left",
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Approximate the integral of a function using the rectangle (left or right) rule.

    This method implements a basic numerical integration technique, where the area
    under the curve is approximated by summing the areas of rectangles. The height
    of each rectangle is determined by either the left or right endpoint of the subinterval.

    Args:
        xp (np.ndarray): A 1D array of x-values representing partition points of the interval.
        yp (np.ndarray): A 1D array of y-values corresponding to f(x) evaluated at each x in xp.
        method (Literal["left", "right"], optional): Choose "left" for left-endpoint rectangles,
            or "right" for right-endpoint rectangles. Defaults to "left".
        return_type (Literal["Decimal", "float"], optional): Determines the return type of the result.
            Use "Decimal" for high-precision computation or "float" for NumPy float. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: The approximate value of the integral using the rectangle rule.

    Raises:
        ValueError: If the input arrays have less than two points or mismatched lengths (checked in utility).
    """
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    if method == "left":
        h = yp[:-1]
    elif method == "right":
        h = yp[1:]

    areas = h * np.diff(xp)
    res = np.sum(areas)

    return np.float64(res) if return_type == "float" else to_decimal(res)


def midpoint(
    xp: np.ndarray,
    yp: np.ndarray,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Approximate the integral of a function using the midpoint rule.

    This method approximates the integral by evaluating the function at the midpoint of
    each subinterval defined by `xp` and multiplying by the subinterval width. It is more
    accurate than basic left/right rectangle rules for smooth functions.

    Args:
        xp (np.ndarray): A 1D array of x-values representing partition points of the interval.
        yp (np.ndarray): A 1D array of y-values corresponding to f(x) evaluated at each x in xp.
        return_type (Literal["Decimal", "float"], optional): Determines the return type of the result.
            Use "Decimal" for high-precision computation or "float" for NumPy float. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: The approximate value of the integral using the midpoint rule.

    Raises:
        ValueError: If fewer than two x-points are provided.
    """
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    if len(xp) < 2:
        raise ValueError("At least two points are required for midpoint rule")

    res = to_decimal(0)
    for i in range(len(xp) - 1):
        x0, x1 = xp[i], xp[i + 1]
        y0, y1 = yp[i], yp[i + 1]

        xm = (x0 + x1) / 2
        ym = y0 + (y1 - y0) * (xm - x0) / (x1 - x0)

        res += (x1 - x0) * ym
    return np.float64(res) if return_type == "float" else to_decimal(res)


def trapezoid(
    xp: np.ndarray,
    yp: np.ndarray,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Approximate the integral of a function using the trapezoidal rule.

    This method approximates the integral by summing the areas of trapezoids under
    the curve. Each trapezoid spans a subinterval [x_i, x_{i+1}] and has bases equal
    to the function values at the endpoints.

    Args:
        xp (np.ndarray): A 1D array of x-values representing partition points of the interval.
        yp (np.ndarray): A 1D array of y-values corresponding to f(x) evaluated at each x in xp.
        return_type (Literal["Decimal", "float"], optional): Determines the return type of the result.
            Use "Decimal" for high-precision computation or "float" for NumPy float. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: The approximate value of the integral using the trapezoidal rule.

    Raises:
        ValueError: If xp and yp have inconsistent lengths (checked in utility).
    """
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    n = len(xp) - 1
    h = np.diff(xp)
    res = sum(h[i] * (yp[i] + yp[i + 1]) / 2 for i in range(n))

    return np.float64(res) if return_type == "float" else to_decimal(res)
