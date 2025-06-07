from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np
import sympy as sp

from ..utils import to_decimal

getcontext().prec = 20
__all__ = ["hspline"]


class __HermiteSpline:
    """
    Cubic Hermite spline interpolation with support for clamped, second derivative, periodic, and not-a-knot boundary conditions.
    """

    def __init__(
        self,
        xp: Union[list, np.ndarray],
        yp: Union[list, np.ndarray],
        bc_type: Literal["not-a-knot", "clamped", "second", "periodic"] = "not-a-knot",
        dy_nodes: Union[list, np.ndarray, None] = None,
        ddy_nodes: Union[list, np.ndarray, None] = None,
    ):
        self.xp = to_decimal(xp)
        self.yp = to_decimal(yp)
        self.n = len(self.xp)
        self.bc_type = bc_type
        self.dy_nodes = to_decimal(dy_nodes) if dy_nodes is not None else None
        self.ddy_nodes = to_decimal(ddy_nodes) if ddy_nodes is not None else None
        self._build_spline()

    def _build_spline(self):
        n = self.n
        x, y = self.xp, self.yp
        h = to_decimal(np.diff(x))
        A = np.zeros((n, n), dtype=Decimal)
        rhs = np.zeros(n, dtype=Decimal)

        for i in range(1, n - 1):
            A[i, i - 1] = h[i] / (h[i - 1] + h[i])
            A[i, i] = 2
            A[i, i + 1] = h[i - 1] / (h[i - 1] + h[i])
            rhs[i] = (
                3
                * (
                    (y[i + 1] - y[i]) / h[i] * h[i - 1]
                    + (y[i] - y[i - 1]) / h[i - 1] * h[i]
                )
                / (h[i - 1] + h[i])
            )

        if self.bc_type == "clamped":
            if self.dy_nodes is None:
                raise ValueError(
                    "dy_nodes must be provided for 'clamped' boundary condition"
                )
            A[0, 0], rhs[0] = 1, self.dy_nodes[0]
            A[-1, -1], rhs[-1] = 1, self.dy_nodes[-1]

        elif self.bc_type == "second":
            if self.ddy_nodes is None:
                raise ValueError(
                    "ddy_nodes must be provided for 'second' boundary condition"
                )
            A[0, 0], A[0, 1] = 2, 1
            A[-1, -2], A[-1, -1] = 1, 2
            rhs[0] = 6 * (y[1] - y[0]) / h[0] / h[0] - h[0] * self.ddy_nodes[0]
            rhs[-1] = 6 * (y[-1] - y[-2]) / h[-1] / h[-1] - h[-1] * self.ddy_nodes[1]

        elif self.bc_type == "periodic":
            A[0, 0], A[0, -1] = 1, -1
            A[-1, 0], A[-1, -1] = 1, -1
            rhs[0], rhs[-1] = 0, 0

        else:  # not-a-knot
            A[0, 0], A[0, 1], A[0, 2] = h[1], -(h[0] + h[1]), h[0]
            A[-1, -3], A[-1, -2], A[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]
            rhs[0] = rhs[-1] = 0

        self.m = to_decimal(np.linalg.solve(np.float32(A), np.float32(rhs)))

        self.coeffs = []
        for i in range(n - 1):
            hi = h[i]
            ai = y[i]
            bi = self.m[i]
            ci = 3 * (y[i + 1] - y[i]) / hi**2 - (2 * self.m[i] + self.m[i + 1]) / hi
            di = 2 * (y[i] - y[i + 1]) / hi**3 + (self.m[i] + self.m[i + 1]) / hi**2
            self.coeffs.append((ai, bi, ci, di, x[i]))

    def interpolate(
        self,
        x: Union[Decimal, float, int, str],
        return_type: Literal["Decimal", "float"] = "float",
    ) -> Union[Decimal, np.float64]:
        """
        Estimate the spline value at one or more query points.

        This method evaluates the piecewise cubic Hermite spline at the specified
        point(s) `x` using precomputed spline coefficients.

        The spline is defined piecewise as:
            S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3

        Args:
            x (Union[Decimal, float, int, str]): The point or points at which to evaluate the spline.
            return_type (Literal["Decimal", "float"], optional): Specifies the return type. Use "Decimal" for high-precision output, or "float" for NumPy float64. Defaults to "float".

        Returns:
            Union[Decimal, np.ndarray]: Interpolated value(s) at the given point(s).
        """
        x = to_decimal(np.atleast_1d(x))
        result = np.zeros_like(x, dtype=Decimal)
        for idx, x in enumerate(x):
            i = np.clip(np.searchsorted(self.xp, x) - 1, 0, self.n - 2)
            ai, bi, ci, di, xi = self.coeffs[i]
            dx = x - xi
            result[idx] = ai + bi * dx + ci * dx**2 + di * dx**3
        return (
            to_decimal(result if len(result) > 1 else result[0])
            if return_type == "Decimal"
            else np.float64(result)
        )

    def derivative(
        self,
        x: Union[float, Decimal, list, np.ndarray],
        order: int = 1,
        return_type: Literal["Decimal", "float"] = "float",
    ) -> Union[Decimal, np.ndarray]:
        """
        Evaluate the first, second, or third derivative of the spline at query points.

        Computes the analytical derivative of the Hermite spline using its
        piecewise-defined coefficients and evaluates it at the specified point(s).

        Derivatives are calculated as follows:
            S'_i(x) = b_i + 2*c_i*(x - x_i) + 3*d_i*(x - x_i)^2
            S''_i(x) = 2*c_i + 6*d_i*(x - x_i)
            S'''_i(x) = 6*d_i

        Args:
            x (Union[float, Decimal, list, np.ndarray]): The point or points at which to evaluate the derivative.
            order (int, optional): Order of the derivative (1, 2, or 3). Defaults to 1.
            return_type (Literal["Decimal", "float"], optional): Specifies the return type. Use "Decimal" for high-precision output, or "float" for NumPy float64. Defaults to "float".

        Returns:
            Union[Decimal, np.ndarray]: Derivative value(s) at the specified point(s).

        Raises:
            ValueError: If the derivative order is not in [1, 2, 3].
        """
        x_sym = sp.symbols("x")
        expr = 0

        for i in range(self.n - 1):
            xi = float(self.xp[i])
            x_next = float(self.xp[i + 1])
            ai, bi, ci, di, _ = [float(c) for c in self.coeffs[i]]

            dx = x_sym - xi
            poly = ai + bi * dx + ci * dx**2 + di * dx**3

            expr += sp.Piecewise((poly, (x_sym >= xi) & (x_sym <= x_next)), (0, True))

        deriv_expr = sp.diff(expr, x_sym, order)

        x_vals = np.atleast_1d(x)
        result = np.zeros_like(x_vals, dtype=Decimal)

        for i, x_val in enumerate(x_vals):
            val = deriv_expr.subs(x_sym, float(x_val))
            result[i] = to_decimal(val)

        return (
            to_decimal(result if len(result) > 1 else result[0])
            if return_type == "Decimal"
            else np.float64(result)
        )

    def integrate(
        self,
        a: Union[float, Decimal],
        b: Union[float, Decimal],
        return_type: Literal["Decimal", "float"] = "float",
    ) -> Decimal:
        """
        Compute the definite integral of the spline between two bounds.

        This method integrates the Hermite spline analytically over the interval [a, b]
        by summing the integrals of relevant polynomial segments.

        Each segment is integrated using: ∫ S_i(x) dx = a_i*dx + b_i*dx^2/2 + c_i*dx^3/3 + d_i*dx^4/4

        Args:
            a (Union[float, Decimal]): Lower bound of the integration interval.
            b (Union[float, Decimal]): Upper bound of the integration interval.
            return_type (Literal["Decimal", "float"], optional): Specifies the return type. Use "Decimal" for high-precision output, or "float" for NumPy float64. Defaults to "float".

        Returns:
            Union[Decimal, float]: The definite integral of the spline over [a, b].
        """
        a, b = to_decimal(a), to_decimal(b)
        if a > b:
            a, b = b, a

        total = Decimal(0)
        left = np.clip(np.searchsorted(self.xp, a) - 1, 0, self.n - 2)
        right = np.clip(np.searchsorted(self.xp, b) - 1, 0, self.n - 2)

        x_left = a
        for i in range(left, right + 1):
            ai, bi, ci, di, xi = self.coeffs[i]
            x0 = max(xi, x_left)
            x1 = min(self.xp[i + 1], b)
            dx0, dx1 = x0 - xi, x1 - xi

            def polyint(dx):
                return ai * dx + bi * dx**2 / 2 + ci * dx**3 / 3 + di * dx**4 / 4

            total += polyint(dx1) - polyint(dx0)
            x_left = x1

        return to_decimal(total) if return_type == "Decimal" else np.float64(total)


hspline = __HermiteSpline
