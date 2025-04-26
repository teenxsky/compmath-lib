import numpy as np
from decimal import Decimal, getcontext
from typing import Union, Optional, Literal


getcontext().prec = 20


def absolute_error(
    value: Union[Decimal, float, int, str],
    exact_value: Optional[Union[Decimal, float, int, str]] = None,
    valid_digits: Optional[int] = None,
    rel_err: Optional[Union[Decimal, float, int, str]] = None,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Calculates the absolute error in various ways:

      1. If exact_value is provided - by definition (using the exact value)
      2. If valid_digits is provided - by the number of valid digits
      3. If rel_err is provided - from the relative error
      4. If nothing is provided - calculates the absolute error for a relative error equal to five units of the next digit place

    Args:
      value (Union[Decimal, float, int, str]): The approximate value.
      exact_value (Optional[Union[Decimal, float, int, str]], optional): The exact value for calculation by definition. Defaults to None.
      valid_digits (Optional[int], optional): The number of valid digits for calculation by valid digits. Defaults to None.
      rel_err (Optional[Union[Decimal, float, int, str]], optional): The relative error for calculation from the relative error. Defaults to None.
      return_type (Union[Decimal, np.float64], optional): The type of the return value. Defaults to "float".

    Raises:
      ValueError: If valid_digits is not positive.

    Returns:
      Union[Decimal, np.float64]: The absolute error.
    """
    number = Decimal(str(value))

    if exact_value is not None:
        exact_value = Decimal(str(exact_value))
        result = abs(exact_value - number)
    elif valid_digits is not None:
        if valid_digits <= 0:
            raise ValueError("The number of valid digits must be positive")

        str_value = format(number, "f").lower().replace(".", "").lstrip("0")
        if not str_value:
            result = Decimal("0")
        else:
            first_non_zero_pos = len(str(number).replace(".", "").lstrip("0")) - len(
                str_value
            )
            order = len(str_value) - valid_digits - first_non_zero_pos
            result = Decimal("5") * Decimal("10") ** Decimal(order)
    elif rel_err is not None:
        rel_err = Decimal(str(rel_err))
        if number == 0:
            result = Decimal("0")
        else:
            result = abs(number) * rel_err
    else:
        str_value = format(number, "f")
        if "." in str_value:
            fractional_part = str_value.split(".")[1]
            if fractional_part:
                order = -len(fractional_part)
            else:
                order = 0
        else:
            order = 0
        result = Decimal("5") * Decimal("10") ** Decimal(order - 1)

    return np.float64(result) if return_type == "float" else result


def relative_error(
    value: Union[Decimal, float, int, str],
    exact_value: Optional[Union[Decimal, float, int, str]] = None,
    valid_digits: Optional[int] = None,
    abs_err: Optional[Union[Decimal, float, int, str]] = None,
    return_type: Literal["Decimal", "float"] = "float",
) -> Union[Decimal, np.float64]:
    """
    Calculates the relative error in various ways:

      1. If exact_value is provided - by definition (using the exact value)
      2. If valid_digits is provided - by the number of valid digits
      3. If abs_err is provided - from the absolute error
      4. If nothing is provided - calculates the relative error for an absolute error equal to five units of the next digit place

    Args:
      value (Union[Decimal, float, int, str]): The approximate value.
      exact_value (Optional[Union[Decimal, float, int, str]], optional): The exact value for calculation by definition. Defaults to None.
      valid_digits (Optional[int], optional): The number of valid digits for calculation by valid digits. Defaults to None.
      abs_err (Optional[Union[Decimal, float, int, str]], optional): The absolute error for calculation from the absolute error. Defaults to None.
      return_type (Union[Decimal, np.float64], optional): The type of the return value. Defaults to "float".

    Raises:
      ValueError: If value is zero or if invalid arguments are provided.

    Returns:
      Union[Decimal, np.float64]: The relative error.
    """
    number = Decimal(str(value))
    if number == 0:
        raise ValueError("Cannot calculate relative error for zero")

    if exact_value is not None:
        exact_value = Decimal(str(exact_value))
        result = abs((exact_value - number) / exact_value)
    elif valid_digits is not None:
        if valid_digits <= 0:
            raise ValueError("The number of valid digits must be positive")

        str_value = format(number, "f").lower().replace(".", "").lstrip("0")
        if not str_value:
            result = Decimal("0")
        else:
            first_non_zero_pos = len(str(number).replace(".", "").lstrip("0")) - len(
                str_value
            )
            order = len(str_value) - valid_digits - first_non_zero_pos
            result = Decimal("5") * Decimal("10") ** Decimal(order) / abs(number)
    elif abs_err is not None:
        abs_err = Decimal(str(abs_err))
        result = abs_err / abs(number)
    else:
        result = absolute_error(number, return_type="Decimal") / abs(number)

    return np.float64(result) if return_type == "float" else result
