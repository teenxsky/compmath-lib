import numpy as np
import sympy as sp
from typing import Tuple, Callable
from decimal import Decimal, getcontext


getcontext().prec = 20


class __LagrangeInterpolation:
    """
    Class for performing Lagrange interpolation.
    """

    def interpolate(
        self,
        x: Decimal,
        x0: Decimal,
        y0: Decimal,
        x1: Decimal,
        y1: Decimal,
        x2: Decimal = None,
        y2: Decimal = None,
        degree: int = 1,
    ) -> Decimal:
        """
        Performs Lagrange interpolation first or second degree (default first degree).

        Args:
            x (Decimal): The x-coordinate at which to interpolate.
            x0 (Decimal): The first x-coordinate.
            y0 (Decimal): The first y-coordinate.
            x1 (Decimal): The second x-coordinate.
            y1 (Decimal): The second y-coordinate.
            x2 (Decimal, optional): The third x-coordinate. Defaults to None. But if degree = 2, it should be provided.
            y2 (Decimal, optional): The third y-coordinate. Defaults to None. But if degree = 2, it should be provided.
            degree (int, optional): The degree of the interpolating polynomial. If degree = 1 then Linear, if degree = 2 then Quadratic. Defaults to 1.

        Raises:
            ValueError: If degree is not supported or missing points for quadratic interpolation.

        Returns:
            Decimal: The interpolated y-coordinate.
        """
        if degree == 1:
            return self.__linear(x, x0, y0, x1, y1)
        elif degree == 2 and x2 and y2:
            return self.__quad(x, x0, y0, x1, y1, x2, y2)
        else:
            raise ValueError(
                "Unsupported degree or missing points for quadratic interpolation."
            )

    def __linear(
        self, x: Decimal, x0: Decimal, y0: Decimal, x1: Decimal, y1: Decimal
    ) -> Decimal:
        """
        Performs linear interpolation using Lagrange's method.

        Args:
            x (Decimal): The x-coordinate at which to interpolate.
            x0 (Decimal): The first x-coordinate.
            y0 (Decimal): The first y-coordinate.
            x1 (Decimal): The second x-coordinate.
            y1 (Decimal): The second y-coordinate.

        Returns:
            Decimal: The interpolated y-coordinate.
        """
        return y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0)

    def __quad(
        self,
        x: Decimal,
        x0: Decimal,
        y0: Decimal,
        x1: Decimal,
        y1: Decimal,
        x2: Decimal,
        y2: Decimal,
    ) -> Decimal:
        """
        Performs quadratic interpolation using Lagrange's method.

        Args:
            x (Decimal): The x-coordinate at which to interpolate (in Lagrange's formula is x*).
            x0 (Decimal): The first x-coordinate (in Lagrange's formula is x_{i-1}).
            y0 (Decimal): The first y-coordinate (in Lagrange's formula is f(x_{i-1}) ).
            x1 (Decimal): The second x-coordinate (in Lagrange's formula is x_{i}).
            y1 (Decimal): The second y-coordinate (in Lagrange's formula is f(x_{i}) ).
            x2 (Decimal): The third x-coordinate (in Lagrange's formula is x_{i+1}).
            y2 (Decimal): The third y-coordinate (in Lagrange's formula is f(x_{i+1}) ).

        Returns:
            Decimal: The interpolated y-coordinate.
        """
        term1 = y0 * (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
        term2 = y1 * (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
        term3 = y2 * (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
        return term1 + term2 + term3

    def est_rem(
        self,
        x: Decimal,
        f: Callable[[sp.Symbol], sp.Basic],
        interval: Tuple[Decimal, Decimal],
        mid: Decimal = None,
        degree: int = 1,
        func_sym: str = "x",
    ) -> Decimal:
        """
        Estimates the remainder term of the Lagrange interpolation using symbolic computation.

        Args:
            x (Decimal): The x-coordinate at which to estimate the remainder.
            f (Callable[[sp.Symbol], sp.Basic]): A sympy-compatible function representing the function being interpolated. The function should use sympy symbols for its definition.
            interval (Tuple[Decimal, Decimal]): The interval over which to estimate the remainder [a, b].
            mid (Decimal, optional): The midpoint of the interval for quadratic interpolation. Defaults to None, but if degree = 2, it should be provided.
            degree (int): The degree of the interpolating polynomial. If Linear, degree = 1; if Quadratic, degree = 2.
            func_sym (str, optional): The symbolic variable name to be used in the sympy function. Defaults to "x".

        Raises:
            ValueError: If the degree is not supported or if the midpoint is missing for quadratic interpolation.

        Returns:
            Decimal: The estimated remainder term.
        """

        a, b = interval
        step = (b - a) / Decimal("10")
        points = [a + step * i for i in range(11)]

        x_sym = sp.symbols(func_sym)
        f_sym = f(x_sym)

        if degree == 1:
            f_derivative = sp.diff(f_sym, x_sym, 2)
            f_derivative_func = sp.lambdify(x_sym, f_derivative, "numpy")
            derivatives = [f_derivative_func(float(p)) for p in points]
            min_deriv = Decimal(min(derivatives))
            max_deriv = Decimal(max(derivatives))

            omega_val = self.__omega_linear(x, (a, b))
            return omega_val / Decimal("2") * (min_deriv + max_deriv) / Decimal("2")
        elif degree == 2 and mid:
            f_derivative = sp.diff(f_sym, x_sym, 3)
            f_derivative_func = sp.lambdify(x_sym, f_derivative, "numpy")
            derivatives = [f_derivative_func(float(p)) for p in points]
            min_deriv = Decimal(min(derivatives))
            max_deriv = Decimal(max(derivatives))

            mid = (a + b) / 2
            omega_val = self.__omega_quad(x, (a, b), mid)
            return omega_val / Decimal("6") * (min_deriv + max_deriv) / Decimal("2")
        else:
            raise ValueError(
                "Unsupported degree or missing mid point for quadratic interpolation (with degree = 2)."
            )

    def __omega_linear(self, x: Decimal, interval: Tuple[Decimal, Decimal]) -> Decimal:
        """
        Auxiliary function for the linear remainder term.

        Args:
            x (Decimal): The x-coordinate at which to evaluate.
            interval (Tuple[Decimal, Decimal]): The interval over which to evaluate [a, b].

        Returns:
            Decimal: The value of the omega function.
        """
        x_i1, x_i2 = interval
        return (x - x_i1) * (x - x_i2)

    def __omega_quad(
        self, x: Decimal, interval: Tuple[Decimal, Decimal], mid: Decimal
    ) -> Decimal:
        """
        Auxiliary function for the quadratic remainder term.

        Args:
            x (Decimal): The x-coordinate at which to evaluate.
            interval (Tuple[Decimal, Decimal]): The interval over which to evaluate [a, b].
            mid (Decimal): The midpoint of the interval.

        Returns:
            Decimal: The value of the omega function.
        """
        x_i1, x_i2 = interval
        return (x - x_i1) * (x - mid) * (x - x_i2)


class __NewtonInterpolation:
    """
    Class for performing Newton interpolation with recursive methods.
    """

    @staticmethod
    def interpolate(
        table: np.ndarray, x_nodes: np.ndarray, x: Decimal, degree: int
    ) -> Decimal:
        """
        Performs interpolation using Newton's method with recursive helper.

        Args:
            table (np.ndarray): The table of divided differences.
            x_nodes (np.ndarray): The list of x-coordinates.
            x (Decimal): The x-coordinate at which to interpolate.
            degree (int): The degree of the interpolating polynomial.

        Returns:
            Decimal: The interpolated y-coordinate.
        """

        def recursive_interp(
            current_degree: int, product: Decimal, result: Decimal
        ) -> Decimal:
            if current_degree > degree:
                return result
            else:
                new_product = (
                    product * (x - x_nodes[current_degree])
                    if current_degree >= 0
                    else product
                )
                new_result = result + (
                    table[0][current_degree + 1] * new_product
                    if current_degree >= 0
                    else table[0][0]
                )
                return recursive_interp(current_degree + 1, new_product, new_result)

        return recursive_interp(0, Decimal("1"), table[0][0])

    @staticmethod
    def dd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Constructs a table of divided differences for Newton's interpolation.

        Args:
            x (np.ndarray): The list of x-coordinates.
            y (np.ndarray): The list of y-coordinates.

        Returns:
            np.ndarray: The table of divided differences.
        """

        n = len(x)
        table = [[Decimal("0") for _ in range(n)] for _ in range(n)]

        for i in range(n):
            table[i][0] = y[i]

        for j in range(1, n):
            for i in range(n - j):
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (
                    x[i + j] - x[i]
                )

        return table


# class __NewtonInterpolation:
#     """
#     Class for performing Newton interpolation.
#     """

#     @staticmethod
#     def interpolate(
#         table: List[List[Decimal]], x_nodes: List[Decimal], x: Decimal, degree: int
#     ) -> Decimal:
#         """
#         Performs interpolation using Newton's method.

#         Args:
#             table (List[List[Decimal]]): The table of divided differences.
#             x_nodes (List[Decimal]): The list of x-coordinates.
#             x (Decimal): The x-coordinate at which to interpolate.
#             degree (int): The degree of the interpolating polynomial.

#         Returns:
#             Decimal: The interpolated y-coordinate.
#         """
#         result = table[0][0]
#         product = Decimal("1")

#         for i in range(1, degree + 1):
#             product *= x - x_nodes[i - 1]
#             result += table[0][i] * product

#         return result

#     @staticmethod
#     def dd(x: List[Decimal], y: List[Decimal]) -> List[List[Decimal]]:

#         n = len(x)
#         table = [[Decimal("0") for _ in range(n)] for _ in range(n)]

#         for i in range(n):
#             table[i][0] = y[i]

#         for j in range(1, n):
#             for i in range(n - j):
#                 table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (
#                     x[i + j] - x[i]
#                 )

#         return table


lagrange = __LagrangeInterpolation()
newton = __NewtonInterpolation()
