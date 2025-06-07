from decimal import Decimal
from typing import Callable, Literal, Union

import numpy as np
from sympy import Basic, Symbol, diff, lambdify

from ..utils import to_decimal

__all__ = ['secant_solve', 'tangent_solve']


def tangent_solve(
    f_sym: Callable[[Symbol], Basic],
    x0: Union[Decimal, float, int, str],
    x_sym: str = 'x',
    eps: float = 1e-8,
    max_iter: int = 100,
    return_type: Literal['Decimal', 'float'] = 'float',
) -> tuple[Union[Decimal, np.float64], int]:
    """
    Tangent method (Newton's method) for solving the equation f(x) = 0.

    Args:
        f_sym (Callable): Symbolic function f(x) defined using sympy.
        x0 (Union[Decimal, float, int, str]): Initial approximation.
        x_sym (str): Symbolic variable name.
        eps (float): Precision.
        return_type (Literal['Decimal', 'float']): Type of the returned value.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: (root approximation, number of iterations)
    """
    x_sym = Symbol(x_sym)
    f_expr = f_sym(x_sym)
    f = lambdify(x_sym, f_expr)
    f_prime_expr = diff(f_expr, x_sym)
    f_prime = lambdify(x_sym, f_prime_expr)

    x_n = x0

    for i in range(1, max_iter + 1):
        f_val = to_decimal(f(np.float64(x_n)))
        f_deriv = to_decimal(f_prime(np.float64(x_n)))

        if f_deriv == 0:
            raise ZeroDivisionError('Derivative is zero — the tangent is vertical.')

        x_next = to_decimal(x_n) - f_val / f_deriv

        if abs(x_next - to_decimal(x_n)) < to_decimal(eps):
            return (np.float64(x_next) if return_type == 'float' else x_next), i

        x_n = x_next

    raise RuntimeError(
        'The tangent method did not converge in the given number of iterations.'
    )


def secant_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x0: Union[Decimal, float, int, str],
    x1: Union[Decimal, float, int, str],
    eps: float = 1e-8,
    max_iter: int = 1000,
    return_type: Literal['Decimal', 'float'] = 'float',
) -> tuple[Union[Decimal, np.float64], int]:
    """
    Secant method for solving the equation f(x) = 0.

    Args:
        f (Callable): Function f(x) that accepts and returns numpy arrays.
        x0 (Union[Decimal, float, int, str]): First approximation.
        x1 (Union[Decimal, float, int, str]): Second approximation.
        eps (float): Precision.
        return_type (Literal['Decimal', 'float']): Type of the returned value.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: (root approximation, number of iterations)
    """
    x_prev = to_decimal(x0)
    x_curr = to_decimal(x1)

    for i in range(1, max_iter + 1):
        f_prev = to_decimal(f(np.array([x_prev], dtype=np.float64))[0])
        f_curr = to_decimal(f(np.array([x_curr], dtype=np.float64))[0])

        denominator = f_curr - f_prev
        if denominator == 0:
            raise ZeroDivisionError(
                'Denominator is zero — two identical values of f(x).'
            )

        x_next = x_curr - f_curr * (x_curr - x_prev) / denominator

        if abs(x_next - x_curr) < to_decimal(eps):
            return (np.float64(x_next) if return_type == 'float' else x_next), i

        x_prev, x_curr = x_curr, x_next

    raise RuntimeError(
        'The chord method did not converge in the given number of iterations.'
    )
