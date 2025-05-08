import numpy as np
from ..utils import to_decimal
from typing import Union, Literal
from decimal import Decimal, getcontext


getcontext().prec = 20
__all__ = ["newton_cotes"]


def newton_cotes(
    xp: np.ndarray,
    yp: np.ndarray,
    coeffs: np.ndarray = None,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Approximate the integral of a function using the Newton–Cotes formula.

    This method implements a generalized Newton–Cotes numerical integration rule, which
    approximates the definite integral of a function based on function values at equally or
    unequally spaced nodes. The integration weights (coefficients) are determined automatically
    by solving a linear system that enforces exact integration of monomials up to degree `n`,
    where `n = len(xp) - 1`.

    If coefficients are not provided, they are computed based on the Vandermonde matrix
    and the moments of monomials over the integration interval.

    Args:
        xp (np.ndarray): A 1D array of x-values representing nodes where the function is evaluated.
        yp (np.ndarray): A 1D array of y-values corresponding to f(x) at each x in xp.
        coeffs (np.ndarray, optional): Optional precomputed integration weights. If None, they will be
            computed automatically. Defaults to None.
        return_type (Literal["Decimal", "float"], optional): Determines the return type of the result.
            Use "Decimal" for high-precision computation or "float" for NumPy float. Defaults to "float".

    Returns:
        Union[Decimal, np.float64]: The approximate value of the integral using the Newton–Cotes rule.

    Raises:
        ValueError: If xp and yp have different lengths (handled in `to_decimal` or upstream).
    """
    xp = to_decimal(xp)
    yp = to_decimal(yp)
    coeffs = to_decimal(coeffs) if coeffs is not None else None

    n = len(xp) - 1
    a, b = xp[0], xp[-1]

    A = np.vander(xp, increasing=True).T
    rhs = np.array([(b ** (m + 1) - a ** (m + 1)) / (m + 1) for m in range(n + 1)])
    coeffs = np.linalg.solve(A, rhs)
    res = np.dot(coeffs, yp)

    return np.float64(res) if return_type == "float" else to_decimal(res)
