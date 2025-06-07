from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np

from ..utils import factorial, to_decimal
from .difftabs import difftabs

getcontext().prec = 20
__all__ = ['newton']


class _Newton:
    """
    Newton interpolation methods including divided, forward, and backward formulas.
    """

    @staticmethod
    def poly(
        x: Union[Decimal, float, int, str],
        xp: np.ndarray,
        yp: np.ndarray,
        dd: np.ndarray = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Estimate value at `x` using Newton's divided differences interpolation.

        Args:
            x (Union[Decimal, float, int, str]): The point to interpolate.
            xp (np.ndarray): Array of known x-coordinates.
            yp (np.ndarray): Array of known y-coordinates.
            dd (np.ndarray, optional): Precomputed divided differences table. If not provided, it will be calculated.
            return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Returns:
            Union[Decimal, np.float64]: Interpolated value at x.

        Raises:
            ValueError: If the input lengths mismatch or dd table is invalid.
        """
        x = to_decimal(x)
        xp = to_decimal(xp)
        yp = to_decimal(yp)

        if len(xp) != len(yp):
            raise ValueError('xp and yp must be of equal length.')

        if dd is None:
            dd = difftabs.div(xp, yp)
        elif len(dd) != len(yp):
            raise ValueError('Length of dd must match yp.')

        n = len(xp)
        x = np.atleast_1d(x)
        val = np.full_like(x, dd[0], dtype=Decimal)

        for i in range(1, n):
            term = dd[i]
            for j in range(i):
                term *= x - xp[j]
            val += term

        result = val[0] if val.size > 0 else val
        return np.float64(result) if return_type == 'float' else result

    @staticmethod
    def fwd(
        x: Union[Decimal, float, int, str],
        xp: np.ndarray,
        yp: np.ndarray,
        fd: np.ndarray = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Estimate value at `x` using Newton's forward difference formula.

        Args:
            x (Union[Decimal, float, int, str]): The point to interpolate.
            xp (np.ndarray): Equally spaced x-coordinates.
            yp (np.ndarray): Corresponding function values.
            fd (np.ndarray, optional): Precomputed forward difference table.
            return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Returns:
            Union[Decimal, np.float64]: Interpolated value at x.

        Raises:
            ValueError: If input lengths mismatch or fd is invalid.
        """
        x = to_decimal(x)
        xp = to_decimal(xp)
        yp = to_decimal(yp)

        if len(xp) != len(yp):
            raise ValueError('xp and yp must be of equal length.')

        n = len(xp)
        if fd is None:
            fd = difftabs.fwd(yp)
        elif len(fd) != n:
            raise ValueError('fd must have the same length as yp.')

        h = xp[1] - xp[0]
        t = (x - xp[0]) / h

        val = fd[0]
        t_prod = to_decimal(1)

        for i in range(1, n):
            t_prod *= t - (i - 1)
            val += t_prod * fd[i] / factorial(i)

        return np.float64(val) if return_type == 'float' else val

    @staticmethod
    def bwd(
        x: Union[Decimal, float, int, str],
        xp: np.ndarray,
        yp: np.ndarray,
        bd: np.ndarray = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Estimate value at `x` using Newton's backward difference formula.

        Args:
            x (Union[Decimal, float, int, str]): The point to interpolate.
            xp (np.ndarray): Equally spaced x-coordinates.
            yp (np.ndarray): Corresponding function values.
            bd (np.ndarray, optional): Precomputed backward difference table.
            return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Returns:
            Union[Decimal, np.float64]: Interpolated value at x.

        Raises:
            ValueError: If input lengths mismatch or bd is invalid.
        """
        x = to_decimal(x)
        xp = to_decimal(xp)
        yp = to_decimal(yp)

        if len(xp) != len(yp):
            raise ValueError('xp and yp must be of equal length.')

        n = len(xp)
        if bd is None:
            bd = difftabs.bwd(yp)
        elif len(bd) != n:
            raise ValueError('bd must have the same length as yp.')

        h = xp[1] - xp[0]
        t = (x - xp[-1]) / h * -1

        val = bd[0]
        t_prod = to_decimal(1)

        for i in range(1, n):
            t_prod *= t - (i - 1)
            val += ((-1) ** i) * t_prod * bd[i] / factorial(i)

        return np.float64(val) if return_type == 'float' else val


newton = _Newton()
