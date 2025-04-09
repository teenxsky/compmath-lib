import numpy as np
import sympy as sp
from typing import Callable, Union
from decimal import Decimal, getcontext


getcontext().prec = 20


class __LagrangeInterpolation:
    """
    Class for performing Lagrange interpolation.
    """

    @staticmethod
    def interpolate(
        x: Union[Decimal, float, int, str],
        x_points: np.ndarray,
        y_points: np.ndarray,
    ) -> Decimal:
        """
        Performs Lagrange interpolation to estimate the value of a function at a given point `x` based on a set of known data points `(x_points, y_points)`.

        The Lagrange interpolation formula is given by:
            L_n(x) = Σ (y_i * l_i(x)) for i = 0 to n-1
        where:
            l_i(x) = Π ((x - x_j) / (x_i - x_j)) for j = 0 to n-1, j ≠ i

        Args:
            x (Union[Decimal, float, int, str]): The x-coordinate at which to interpolate.
            x_points (np.ndarray): An array of x-coordinates used for interpolation.
            y_points (np.ndarray): An array of y-coordinates corresponding to x_points.

        Raises:
            ValueError: If the number of x_points and y_points do not match or if less than two points are provided.

        Returns:
            Decimal: The interpolated y-coordinate.
        """
        if len(x_points) < 2 or len(x_points) != len(y_points):
            raise ValueError(
                "x_points and y_points must have at least two elements and be of the same length."
            )

        L_n = Decimal("0")
        x = Decimal(str(x))
        x_points = np.array([Decimal(str(x)) for x in x_points], dtype=Decimal)
        y_points = np.array([Decimal(str(y)) for y in y_points], dtype=Decimal)

        for i in range(len(x_points)):
            l_i = Decimal("1")
            for j in range(len(x_points)):
                if i != j:
                    l_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
            L_n += y_points[i] * l_i

        return L_n

    @staticmethod
    def est_rem(
        x: Union[Decimal, float, int, str],
        nodes: np.ndarray,
        derivative_x: Union[Decimal, float, int, str] = None,
        f: Callable[[sp.Symbol], sp.Basic] = None,
        func_sym: str = "x",
    ) -> Decimal:
        """
        Estimate the remainder term of the Lagrange interpolation polynomial.

        This function calculates the remainder term of the Lagrange interpolation polynomial for a given set of nodes and a function or its derivative. It uses either the provided (n+1)-th derivative value or computes it symbolically if a function is provided.

        Args:
            x (Union[Decimal, float, int, str]): The x-coordinate at which to estimate the remainder.
            nodes (np.ndarray): An array of interpolation nodes.
            derivative_x (Union[Decimal, float, int, str], optional): The value of the (n+1)-th derivative at `x`. Defaults to None. Will be computed if `f` is provided.
            f (Callable[[sp.Symbol], sp.Basic], optional): A sympy-compatible function representing the function being interpolated. Defaults to None. If `derivative_x` is not provided, this function must be provided.
            func_sym (str, optional): The symbolic variable name to be used in the sympy function. Defaults to "x".

        Returns:
            Decimal: The estimated remainder term of the Lagrange interpolation polynomial.

        Raises:
            ValueError: If the (n+1)-th derivative of the function cannot be computed or if neither `f` nor `derivative_x` is provided.
        """
        x = Decimal(str(x))
        nodes = np.array([Decimal(str(node)) for node in nodes], dtype=Decimal)

        n = len(nodes) - 1

        if derivative_x is None:
            x_sym = sp.symbols(func_sym)
            try:
                f_derivative = sp.diff(f(x_sym), x_sym, n + 1)
            except Exception:
                raise ValueError(
                    f"Failed to compute the {n + 1}-th derivative of the function"
                )

            derivative_x = f_derivative.evalf(subs={x_sym: x})
        elif f is None:
            raise ValueError(
                "Either f or derivative_x must be provided to estimate the remainder."
            )

        omega_x = Decimal("1")
        for x_i in nodes:
            omega_x *= x - x_i

        factorial = Decimal(str(np.prod([i for i in range(1, n + 2)])))

        return derivative_x / factorial * omega_x


class __NewtonInterpolation:
    """
    Class for performing Newton interpolation with recursive methods.
    """

    def interpolate(
        self,
        x: Decimal,
        x_points: np.ndarray,
        y_points: np.ndarray,
        dd_table: np.ndarray = None,
    ):
        """
        Interpolates the value of a function at a given point `x` using the Newton's divided differences method.

        The interpolation is based on the provided x_points and y_points, which represent the known data points.
        Optionally, a precomputed divided differences table (`dd_table`) can be provided to optimize the computation.

        The interpolation formula used is:
            P(x) = f[x0] + f[x0, x1](x - x0) + f[x0, x1, x2](x - x0)(x - x1) + ... + f[x0, x1, ..., xn](x - x0)(x - x1)...(x - xn-1)

        Where:
            - f[x0], f[x0, x1], ..., f[x0, x1, ..., xn] are the divided differences.
            - x0, x1, ..., xn are the x_points.

        Parameters:
            x (Decimal): The point at which to interpolate the value.
            x_points (np.ndarray): An array of x-coordinates of the known data points.
            y_points (np.ndarray): An array of y-coordinates of the known data points.
            dd_table (np.ndarray, optional): A precomputed divided differences table. Defaults to None.

        Returns:
            Decimal: The interpolated value at the given point `x`.

        Raises:
            ValueError: If `dd_table` is provided but its length does not match the length of `y_points`.
        """
        x = Decimal(str(x))
        x_points = np.array([Decimal(str(x)) for x in x_points], dtype=Decimal)
        y_points = np.array([Decimal(str(y)) for y in y_points], dtype=Decimal)

        if dd_table is None:
            table = self.dd(x_points, y_points)
        else:
            if len(dd_table) != len(y_points):
                raise ValueError("dd_table must have the same length as y_points.")

        n = len(y_points)
        result = Decimal(str(table[0, 0]))

        for i in range(1, n):
            term = Decimal(str(table[0, i]))
            for j in range(i):
                term *= x - x_points[j]
            result += term

        return result

    def dd(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """
        Compute the divided differences table for a given set of points.

        This method calculates the divided differences table, which is used in polynomial interpolation (e.g., Newton's interpolation).

        The interpolation formula used is:

            f[x_i, ..., x_{i+j}] = (f[x_{i+1}, ..., x_{i+j}] - f[x_i, ..., x_{i+j-1}]) / (x_{i+j} - x_i)

        Where `f[x_i, ..., x_{i+j}]` represents the divided difference of order `j`.

        Args:
            x_points (np.ndarray): A 1D array of x-coordinates of the data points.
            y_points (np.ndarray): A 1D array of y-coordinates of the data points.

        Returns:
            np.ndarray: A 2D array representing the divided differences table. The
            first column contains the `y_points`, and the subsequent columns contain
            the divided differences of increasing order.
        """
        x_points = np.array([Decimal(str(x)) for x in x_points], dtype=Decimal)
        y_points = np.array([Decimal(str(y)) for y in y_points], dtype=Decimal)

        n = len(y_points)
        table = np.zeros([n, n])

        table[:, 0] = y_points

        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (
                    Decimal(str(table[i + 1, j - 1])) - Decimal(str(table[i, j - 1]))
                ) / (x_points[i + j] - x_points[i])

        return table


lagrange = __LagrangeInterpolation()
newton = __NewtonInterpolation()
