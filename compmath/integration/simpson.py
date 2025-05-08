import numpy as np
from ..utils import to_decimal
from typing import Union, Literal
from decimal import Decimal, getcontext


getcontext().prec = 20
__all__ = ["simpson"]


class __Simpson:
    """
    Class implementing Simpson's numerical integration rules.

    Provides methods for computing definite integrals using:
    - Simpson's 1/3 rule (requires an even number of subintervals)
    - Simpson's 3/8 rule (requires number of subintervals divisible by 3)

    All calculations support high-precision arithmetic using the `Decimal` type.
    """

    @staticmethod
    def quad(
        xp: np.ndarray,
        yp: np.ndarray,
        return_type: Literal["Decimal", "float"] = "float",
    ) -> Union[Decimal, np.float64]:
        """
        Approximate the integral of a function using Simpson's 1/3 rule.

        This method computes the integral of tabulated data over an interval by applying
        Simpson's rule to each pair of adjacent subintervals (grouped in 2s). The method
        requires an even number of intervals (i.e., an odd number of nodes).

        Formula used over each subinterval group [x0, x1, x2]:
            ∫ f(x) dx ≈ (h / 6) * (f(x0) + 4f(x1) + f(x2)),
            where h = x2 - x0

        Args:
            xp (np.ndarray): A 1D array of x-values (must have odd length).
            yp (np.ndarray): A 1D array of y-values corresponding to f(x) at each xp.
            return_type (Literal["Decimal", "float"], optional): Desired return type.
                Defaults to "float".

        Returns:
            Union[Decimal, np.float64]: Approximate integral of the function.

        Raises:
            ValueError: If the number of intervals is not even.
        """
        xp = to_decimal(xp)
        yp = to_decimal(yp)

        n = len(xp) - 1
        if n % 2 != 0:
            raise ValueError("Number of intervals must be even for Simpson's rule")
        res = to_decimal(0)
        for i in range(0, n, 2):
            h = xp[i + 2] - xp[i]
            res += (h / 6) * (yp[i] + 4 * yp[i + 1] + yp[i + 2])

        return np.float64(res) if return_type == "float" else to_decimal(res)

    @staticmethod
    def cubic(
        xp: np.ndarray,
        yp: np.ndarray,
        return_type: Literal["Decimal", "float"] = "float",
    ) -> Union[Decimal, np.float64]:
        """
        Approximate the integral of a function using Simpson's 3/8 rule.

        This method computes the integral by applying the 3/8 rule over groups of
        three subintervals (i.e., four nodes). It requires that the number of intervals
        is divisible by 3.

        Formula used over each subinterval group [x0, x1, x2, x3]:
            ∫ f(x) dx ≈ (3h / 8) * (f(x0) + 3f(x1) + 3f(x2) + f(x3)),
            where h = (x3 - x0) / 3

        Args:
            xp (np.ndarray): A 1D array of x-values (length must be 3k + 1).
            yp (np.ndarray): A 1D array of y-values corresponding to f(x) at each xp.
            return_type (Literal["Decimal", "float"], optional): Desired return type.
                Defaults to "float".

        Returns:
            Union[Decimal, np.float64]: Approximate integral of the function.

        Raises:
            ValueError: If the number of intervals is not divisible by 3.
        """
        xp = to_decimal(xp)
        yp = to_decimal(yp)

        n = len(xp) - 1
        if n % 3 != 0:
            raise ValueError(
                "Number of intervals must be divisible by 3 for Simpson's 3/8 rule"
            )
        res = to_decimal(0)
        for i in range(0, n, 3):
            h = (xp[i + 3] - xp[i]) / 3
            res += (3 * h / 8) * (yp[i] + 3 * yp[i + 1] + 3 * yp[i + 2] + yp[i + 3])

        return np.float64(res) if return_type == "float" else to_decimal(res)


simpson = __Simpson()
