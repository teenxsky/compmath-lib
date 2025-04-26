import numpy as np
from decimal import Decimal, getcontext
from ..utils import factorial, to_decimal
from typing import Callable, Union, Literal
from sympy import symbols, diff, Symbol, Basic


getcontext().prec = 20


def lagrange(
    x: Union[Decimal, float, int, str],
    xp: np.ndarray,
    yp: np.ndarray,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Estimate a value at a given point using Lagrange interpolation.

    This method uses the Lagrange polynomial to compute the interpolated value
    at the specified point `x`, based on known data points (`xp`, `yp`).

    Lagrange interpolation formula:
        L(x) = Σ (y_i * l_i(x)) for i in [0, n-1]
    where:
        l_i(x) = Π ((x - x_j) / (x_i - x_j)), for all j ≠ i

    Args:
        x (Union[Decimal, float, int, str]): The point at which to evaluate the interpolated polynomial.
        xp (np.ndarray): 1D array of known x-values.
        yp (np.ndarray): 1D array of known y-values corresponding to `xp`.
        return_type (Literal["Decimal", "float"], optional): Specifies the return type.

    Returns:
        Union[Decimal, np.float64]: Interpolated value at x.

    Raises:
        ValueError: If input arrays have different lengths or contain fewer than 2 points.
    """
    if len(xp) < 2 or len(xp) != len(yp):
        raise ValueError(
            "xp and yp must be the same length and contain at least two points."
        )

    x = to_decimal(x)
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    val = to_decimal(0)

    for i in range(len(xp)):
        term = to_decimal(1)
        for j in range(len(xp)):
            if i != j:
                num = x - xp[j]
                den = xp[i] - xp[j]
                term *= num / den
        val += yp[i] * term

    return np.float64(val) if return_type == "float" else to_decimal(val)


def rem(
    x: Union[Decimal, float, int, str],
    xp: np.ndarray,
    f_deriv_at_xi: Union[Decimal, float, int, str] = None,
    f: Callable[[Symbol], Basic] = None,
    xi_val: Union[Decimal, float, int, str] = None,
    sym: str = "x",
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Estimates the interpolation remainder (error term) at a given point for the Lagrange polynomial.

    This function computes the interpolation error at a given point `x`, using either:
    - a provided value of the (n+1)-th derivative of the original function at some ξ,
    - or computes it symbolically from the function if `f` is provided.

    The formula for the remainder term:
        R_n(x) = f^{(n+1)}(ξ) / (n+1)! * Π (x - x_i), for i = 0 to n,
    where ξ ∈ [min(xp), max(xp)].

    Args:
        x (Union[Decimal, float, int, str]): The x-coordinate where the remainder should be evaluated.
        xp (np.ndarray): 1D array of interpolation nodes.
        f_deriv_at_xi (Union[Decimal, float, int, str], optional): Precomputed (n+1)-th derivative value at some point ξ. If not provided, `f` must be specified.
        f (Callable[[Symbol], Basic], optional): Symbolic function (SymPy compatible). Used to compute the (n+1)-th derivative if `f_deriv_at_xi` is not provided.
        xi_val (Union[Decimal, float, int, str], optional): The specific point ξ at which to evaluate the derivative. Defaults to the average of xp if not specified.
        sym (str, optional): Symbolic variable name for differentiation. Default is "x".
        return_type (Literal["Decimal", "float"], optional): Whether to return the result as `Decimal` (high precision) or `float` (NumPy float64). Default is "float".

    Returns:
        Union[Decimal, np.float64]: The estimated remainder value at x.

    Raises:
        ValueError: If neither `f_deriv_at_xi` nor `f` is provided, or if differentiation fails.
    """
    x = to_decimal(x)
    xp = to_decimal(xp)
    n = len(xp) - 1
    x_sym = symbols(sym)

    if f_deriv_at_xi is None:
        if f is None:
            raise ValueError("Either 'f_deriv_at_xi' or 'f' must be provided.")
        try:
            f_derivative = f(x_sym).diff(x_sym, n + 1)
        except Exception as e:
            raise ValueError(f"Failed to compute the (n+1)-th derivative: {e}")

        if xi_val is None:
            xi_val = to_decimal(sum(xp)) / to_decimal(len(xp))
        else:
            xi_val = to_decimal(xi_val)

        f_deriv_at_xi = to_decimal(f_derivative.subs(x_sym, xi_val))
    else:
        f_deriv_at_xi = to_decimal(f_deriv_at_xi)

    omega = to_decimal(1)
    for xi in xp:
        if xi == x:
            continue
        omega *= x_sym - xi

    omega_at_x = omega.subs(x_sym, x)
    remainder = f_deriv_at_xi * omega_at_x / to_decimal(factorial(n + 1))

    return to_decimal(remainder) if return_type == "Decimal" else np.float64(remainder)


def lagrange_deriv(
    x: Union[Decimal, float, int, str],
    k: int,
    xp: np.ndarray,
    yp: np.ndarray,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Computes the k-th derivative of the Lagrange interpolating polynomial at a given point.

    Args:
        x (Union[Decimal, float, int, str]): The point at which the derivative should be evaluated. Will be converted to Decimal for accuracy.

        k (int): The order of the derivative to compute.

        xp (np.ndarray): The array of x-values (nodes) used for interpolation. Should be uniformly spaced.

        yp (np.ndarray): The array of y-values corresponding to xp (i.e., f(xp)).

        return_type (Literal["Decimal", "float"], optional): Specifies the return type.

    Returns:
        Union[Decimal, np.float64]: The k-th derivative of the Lagrange interpolating polynomial evaluated at x.

    Raises:
        ValueError: If less than two points are provided for interpolation.

    Notes:
        - The method constructs a symbolic Lagrange interpolating polynomial using SymPy.
        - It assumes that the x-nodes are equally spaced, which is used for normalization in the denominator.
        - High precision is maintained throughout using Decimal arithmetic.
        - The Lagrange polynomial is differentiated symbolically k times and then evaluated at the target point.
    """
    x = to_decimal(x)
    xp = to_decimal(xp)
    yp = to_decimal(yp)

    x_sym = symbols("x")
    L = Decimal(0)
    n = len(xp)

    if n < 2:
        raise ValueError("At least two points are required for interpolation.")

    h = xp[1] - xp[0]

    for i in range(n):
        li = Decimal(1)
        for j in range(n):
            if j != i:
                li *= (x_sym - xp[j]) / ((i - j) * h)
        L += yp[i] * li

    try:
        L_k = diff(L, x_sym, k)
    except Exception as e:
        raise ValueError(f"Failed to compute the {k}-th derivative: {e}")

    L_k_x = L_k.subs(x_sym, x)

    return np.float64(L_k_x) if return_type == "float" else to_decimal(L_k_x)
