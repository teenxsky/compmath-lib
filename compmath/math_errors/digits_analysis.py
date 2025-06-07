from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np

from .errors import absolute_error

getcontext().prec = 20
__all__ = ['digits_analysis']


class __DigitsAnalysis:
    """
    A class to analyze the significant, valid, and doubtful digits of a number.

    Attributes:
      value (Union[Decimal, float, int, str]): The number to be analyzed.
    """

    def __init__(self, value: Union[Decimal, float, int, str]):
        """
        Initializes the __DigitsAnalysis class.

        Args:
          value (Union[Decimal, float, int, str]): The number to be analyzed.
        """
        self._dec_number = Decimal(str(value))
        self._int_part = (
            str(self._dec_number).split('.')[0][1:]
            if self._dec_number < 0
            else str(self._dec_number).split('.')[0]
        )
        self._frac_part = (
            str(self._dec_number).split('.')[1]
            if '.' in str(self._dec_number)
            else ''
        )

    def sd(
        self, return_type: Literal['Decimal', 'float'] = 'float'
    ) -> np.ndarray:
        """
        Returns the significant digits of the number.

        Args:
          return_type (Literal["Decimal", "float"], optional): The type of the returned digits. Defaults to "float".

        Returns:
          np.ndarray: A list of significant digits.
        """
        str_num = (
            f'{self._int_part}.{self._frac_part}'
            if self._frac_part
            else self._int_part
        )

        digits = [
            Decimal(digit) if return_type == 'Decimal' else np.float64(digit)
            for digit in str_num.replace('.', '').lstrip('0')
        ]
        return np.array(digits)

    def vd(
        self,
        abs_err: Union[Decimal, float, int, str] = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> np.ndarray:
        """
        Finds the valid digits of the number based on the absolute error.

        Args:
          abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).
          return_type (Literal["Decimal", "float"], optional): The type of the returned digits. Defaults to "float".

        Returns:
          np.ndarray: A list of valid digits.
        """
        if abs_err is None:
            abs_err = absolute_error(self._dec_number)
        else:
            abs_err = Decimal(str(abs_err))

        valid_digits = []
        digits = self._int_part + self._frac_part

        for i, digit in enumerate(digits):
            alpha = len(self._int_part) - 2 - i
            threshold = Decimal('5') * Decimal(10) ** Decimal(alpha)

            if threshold >= abs_err:
                valid_digits.append(
                    Decimal(digit)
                    if return_type == 'Decimal'
                    else np.float64(digit)
                )

        return np.array(valid_digits)

    def dd(
        self,
        abs_err: Union[Decimal, float, int, str] = None,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> np.ndarray:
        """
        Finds the doubtful digits of the number based on the absolute error.

        Args:
          abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).
          return_type (Literal["Decimal", "float"], optional): The type of the returned digits. Defaults to "float".

        Returns:
          np.ndarray: A list of doubtful digits.
        """
        if abs_err is None:
            abs_err = absolute_error(self._dec_number)
        else:
            abs_err = Decimal(str(abs_err))

        doubtful_digits = []
        digits = self._int_part + self._frac_part

        for i, digit in enumerate(digits):
            alpha = len(self._int_part) - 2 - i
            threshold = Decimal('5') * Decimal(10) ** Decimal(alpha)

            if threshold < abs_err:
                doubtful_digits.append(
                    Decimal(digit)
                    if return_type == 'Decimal'
                    else np.float64(digit)
                )

        return np.array(doubtful_digits)


digits_analysis = __DigitsAnalysis
