from decimal import Decimal, getcontext

import numpy as np

from ..utils import to_decimal

getcontext().prec = 20
__all__ = ["difftabs"]


class __DifferenceTables:
    """
    Class for computing difference tables for interpolation methods.
    """

    @staticmethod
    def div(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Compute the divided differences table used in Newton's interpolation.

        This method constructs a one-dimensional array representing the
        coefficients of the Newton interpolating polynomial using divided differences.
        The result can be directly used for evaluating the polynomial.

        Divided difference formula used:
            f[x_i, ..., x_{i+k}] = (f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]) / (x_{i+k} - x_i)

        Args:
            xp (np.ndarray): A 1D array of x-coordinates of known data points.
            yp (np.ndarray): A 1D array of corresponding y-coordinates.

        Returns:
            np.ndarray: A 1D array where each element is a divided difference of increasing order.

        Raises:
            ValueError: If x and y do not have the same length.
        """
        xp = to_decimal(xp)
        yp = to_decimal(yp)

        if len(xp) != len(yp):
            raise ValueError("x and y must have the same length")

        n = len(xp)
        table = yp.copy()

        for i in range(1, n):
            num = table[i:n] - table[i - 1 : n - 1]
            den = xp[i:n] - xp[: n - i]
            table[i:n] = num / den

        return table

    def fin(self, yp: np.ndarray) -> np.ndarray:
        """
        Generate a table of finite differences for a given set of function values.

        This method constructs a list of arrays, each representing a row of
        forward finite differences of increasing order. It is commonly used for
        Newton-Gregory interpolation when x-values are equally spaced.

        Args:
            yp (np.ndarray): A 1D array of function values.

        Returns:
            List[np.ndarray]: A list where the first array is the original values, and each subsequent array contains higher-order finite differences.
        """
        yp = to_decimal(yp)
        table = [np.array(yp, dtype=Decimal)]

        for _ in range(1, len(yp)):
            prev_row = table[-1]
            next_row = np.array(
                [prev_row[i] - prev_row[i - 1] for i in range(1, len(prev_row))],
                dtype=Decimal,
            )
            table.append(next_row)

        return table

    def fwd(self, yp: np.ndarray) -> np.ndarray:
        """
        Extract the forward differences from the finite difference table.

        This method returns the first element of each row in the finite
        difference table, which corresponds to the coefficients of the
        Newton forward interpolation formula.

        Args:
            yp (np.ndarray): A 1D array of function values.

        Returns:
            np.ndarray: A 1D array of forward differences.
        """
        diff = self.fin(yp)
        return np.array([row[0] for row in diff])

    def bwd(self, yp: np.ndarray) -> np.ndarray:
        """
        Extract the backward differences from the finite difference table.

        This method returns the last element of each row in the finite
        difference table, which corresponds to the coefficients of the
        Newton backward interpolation formula.

        Args:
            y (np.ndarray): A 1D array of function values.

        Returns:
            np.ndarray: A 1D array of backward differences.
        """
        diff = self.fin(yp)
        return np.array([row[-1] for row in diff])


difftabs = __DifferenceTables()
