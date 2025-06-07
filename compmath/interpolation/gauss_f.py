from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np

from ..utils import factorial, to_decimal
from .difftabs import difftabs

getcontext().prec = 20
__all__ = ['gauss']


class _Gauss:
    """
    Gauss interpolation methods (forward and backward) for equally spaced nodes.
    """

    @staticmethod
    def fwd(
        x: Union[Decimal, float, int, str],
        xp: np.ndarray,
        yp: np.ndarray,
        fd: np.ndarray = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Gauss forward interpolation for evenly spaced nodes.

        Args:
            x (Union[Decimal, float, int, str]): Point at which to interpolate.
            xp (np.ndarray): Equally spaced x-coordinates.
            yp (np.ndarray): Corresponding y-values.
            fd (np.ndarray, optional): Precomputed finite difference table.
            return_type (Literal["Decimal", "float"], optional): Return type of the result. Defaults to "float".

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

        if fd is None:
            fd = difftabs.fin(yp)
        elif len(fd) != len(yp):
            raise ValueError('fd must have the same length as yp.')

        n = len(xp) - 1
        m = n // 2
        h = xp[1] - xp[0]
        t = (x - xp[m]) / h

        val = fd[0][m]
        t_prod = to_decimal(1)

        for k in range(1, n + 1):
            d = k // 2
            i = m - d
            t_prod *= (t - d) if k % 2 == 0 else (t + d)
            val += t_prod * fd[k][i] / factorial(k)

        return np.float64(val) if return_type == 'float' else val

    @staticmethod
    def bwd(
        x: Union[Decimal, float, int, str],
        xp: np.ndarray,
        yp: np.ndarray,
        fd: np.ndarray = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Gauss backward interpolation for evenly spaced nodes.

        Args:
            x (Union[Decimal, float, int, str]): Point at which to interpolate.
            xp (np.ndarray): Equally spaced x-coordinates.
            yp (np.ndarray): Corresponding y-values.
            fd (np.ndarray, optional): Precomputed finite difference table.
            return_type (Literal["Decimal", "float"], optional): Return type of the result. Defaults to "float".

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

        if fd is None:
            fd = difftabs.fin(yp)
        elif len(fd) != len(yp):
            raise ValueError('fd must have the same length as yp.')

        n = len(xp) - 1
        m = n // 2
        h = xp[1] - xp[0]
        t = (x - xp[m]) / h

        val = fd[0][m]
        t_prod = to_decimal(1)

        for k in range(1, n + 1):
            d = k // 2
            i = m - (k + 1) // 2
            t_prod *= (t + d) if k % 2 == 0 else (t - d)
            val += t_prod * fd[k][i] / factorial(k)

        return np.float64(val) if return_type == 'float' else val


gauss = _Gauss()
