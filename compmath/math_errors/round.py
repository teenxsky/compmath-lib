from decimal import Decimal, getcontext
from typing import Literal, Union

import numpy as np

from .errors import absolute_error

getcontext().prec = 20
__all__ = ['round_to']


class __RoundTo:
    """
    A class to round a number to a specified number of significant, valid, or doubtful digits.

    Attributes:
      value (float): Union[Decimal, float, int, str].
    """

    def __init__(self, value: Union[Decimal, float, int, str]):
        """
        Initializes the __RoundTo class.

        Args:
          value (float): Union[Decimal, float, int, str].
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
        self,
        num_digits: int = 1,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Rounds the number to the specified number of significant digits.

        Args:
          num_digits (int, optional): The number of significant digits. Defaults to 1.
          return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Raises:
          ValueError: If num_digits is not greater than 0.

        Returns:
          Union[Decimal, np.float64]: The rounded number.
        """
        if num_digits <= 0:
            raise ValueError('num_digits must be greater than 0')

        round_index = 0
        significant_digits_count = 0
        for i, digit in enumerate(self._int_part + self._frac_part):
            if significant_digits_count == num_digits:
                round_index = -(len(self._int_part) - i)
                break

            if digit != '0':
                significant_digits_count += 1

        result = (
            self._dec_number
            if round_index <= 0
            else round(self._dec_number, round_index)
        )
        return np.float64(result) if return_type == 'float' else result

    def vd(
        self,
        abs_err: Union[Decimal, float, int, str] = None,
        num_digits: int = 1,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Rounds the number to the specified number of valid digits based on the absolute error.

        Args:
          abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).
          num_digits (int, optional): The number of valid digits. Defaults to 1.
          return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Raises:
          ValueError: If num_digits is not greater than 0.

        Returns:
          Union[Decimal, np.float64]: The rounded number.
        """
        if num_digits <= 0:
            raise ValueError('num_digits must be greater than 0')

        if abs_err is None:
            abs_err = absolute_error(self._dec_number)
        else:
            abs_err = Decimal(str(abs_err))

        round_index = 0
        valid_digits_count = 0
        for i in range(len(self._int_part) + len(self._frac_part)):
            if valid_digits_count == num_digits:
                round_index = -(len(self._int_part) - i)
                break

            alpha = len(self._int_part) - 2 - i
            threshold = Decimal('5') * Decimal(10) ** Decimal(alpha)

            if threshold >= abs_err:
                valid_digits_count += 1

        result = (
            self._dec_number
            if round_index <= 0
            else round(self._dec_number, round_index)
        )
        return np.float64(result) if return_type == 'float' else result

    def dd(
        self,
        abs_err: Union[Decimal, float, int, str] = None,
        num_digits: int = 1,
        return_type: Literal['Decimal', 'float'] = 'float',
    ) -> Union[Decimal, np.float64]:
        """
        Rounds the number to the specified number of doubtful digits based on the absolute error.

        Args:
          abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).
          num_digits (int, optional): The number of doubtful digits. Defaults to 1.
          return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Raises:
          ValueError: If num_digits is not greater than 0.

        Returns:
          Union[Decimal, np.float64]: The rounded number.
        """
        if num_digits <= 0:
            raise ValueError('num_digits must be greater than 0')

        if abs_err is None:
            abs_err = absolute_error(self._dec_number)
        else:
            abs_err = Decimal(str(abs_err))

        round_index = 0
        doubtful_digits_count = 0
        for i in range(len(self._int_part) + len(self._frac_part)):
            if doubtful_digits_count == num_digits:
                round_index = -(len(self._int_part) - i)
                break

            alpha = len(self._int_part) - 2 - i
            threshold = Decimal('5') * Decimal(10) ** Decimal(alpha)

            if threshold < abs_err:
                doubtful_digits_count += 1

        result = (
            self._dec_number
            if round_index <= 0
            else round(self._dec_number, round_index)
        )
        return np.float64(result) if return_type == 'float' else result


round_to = __RoundTo
