import math
from decimal import Decimal, getcontext
from typing import Callable, Union, Optional


def absolute_error(
    value: Union[Decimal, float, int, str],
    exact_value: Optional[Union[Decimal, float, int, str]] = None,
    valid_digits: Optional[int] = None,
    rel_err: Optional[Union[Decimal, float, int, str]] = None,
) -> Decimal:
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

    Raises:
        ValueError: If valid_digits is not positive.

    Returns:
        Decimal: The absolute error.
    """
    number = Decimal(str(value))
    getcontext().prec = 20

    if exact_value is not None:
        exact_value = Decimal(str(exact_value))
        return abs(exact_value - number)

    if valid_digits is not None:
        if valid_digits <= 0:
            raise ValueError("The number of valid digits must be positive")

        str_value = format(number, "f").lower().replace(".", "").lstrip("0")
        if not str_value:
            return Decimal("0")

        first_non_zero_pos = len(str(number).replace(".", "").lstrip("0")) - len(
            str_value
        )
        order = len(str_value) - valid_digits - first_non_zero_pos

        return Decimal("5") * Decimal("10") ** Decimal(order)

    if rel_err is not None:
        rel_err = Decimal(str(rel_err))
        if number == 0:
            return Decimal("0")
        return abs(number) * rel_err

    str_value = format(number, "f")
    if "." in str_value:
        fractional_part = str_value.split(".")[1]
        if fractional_part:
            order = -len(fractional_part)
        else:
            order = 0
    else:
        order = 0

    return Decimal("5") * Decimal("10") ** Decimal(order - 1)


def relative_error(
    value: Union[Decimal, float, int, str],
    exact_value: Optional[Union[Decimal, float, int, str]] = None,
    valid_digits: Optional[int] = None,
    abs_err: Optional[Union[Decimal, float, int, str]] = None,
) -> Decimal:
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

    Raises:
        ValueError: If value is zero or if invalid arguments are provided.

    Returns:
        Decimal: The relative error.
    """
    number = Decimal(str(value))
    if number == 0:
        raise ValueError("Cannot calculate relative error for zero")

    getcontext().prec = 20

    if exact_value is not None:
        exact_value = Decimal(str(exact_value))
        return abs((exact_value - number) / exact_value)

    if valid_digits is not None:
        if valid_digits <= 0:
            raise ValueError("The number of valid digits must be positive")

        str_value = format(number, "f").lower().replace(".", "").lstrip("0")
        if not str_value:
            return Decimal("0")

        first_non_zero_pos = len(str(number).replace(".", "").lstrip("0")) - len(
            str_value
        )
        order = len(str_value) - valid_digits - first_non_zero_pos

        return Decimal("5") * Decimal("10") ** Decimal(order) / abs(number)

    if abs_err is not None:
        abs_err = Decimal(str(abs_err))
        return abs_err / abs(number)

    return absolute_error(number) / abs(number)


class __ConditionalNumbers:
    """
    A class to compute the absolute and relative condition numbers of a given function using Decimal.

    The condition number provides a measure of how sensitive the output of a function is to changes in its input.
    This class uses Decimal for precise calculations of both absolute and relative condition numbers.

    Attributes:
        x (Decimal): The point at which the condition number is evaluated.
        dX (Decimal): The small change in x used to compute the derivative approximation.
        fX (Decimal): The value of the function at x.
        fXdX (Decimal): The value of the function at x + dX.
    """

    def __init__(
        self,
        f: Callable[[Decimal], Decimal],
        x: Union[Decimal, float, int, str] = Decimal("1"),
        dX: Union[Decimal, float, int, str] = Decimal("0.001"),
        precision: int = 20,
    ) -> None:
        """
        Initializes the __ConditionalNumbers class with Decimal precision.

        Args:
            f (Callable[[Decimal], Decimal]): The function for condition numbers.
            x (Union[Decimal, float, int, str], optional): Evaluation point. Defaults to '1'.
            dX (Union[Decimal, float, int, str], optional): Small change for derivative. Defaults to '0.001'.
            precision (int, optional): Decimal precision. Defaults to 20.

        Raises:
            ValueError: If dX is zero.
            ValueError: If the function value f(x) is zero.
        """
        getcontext().prec = precision

        self._x = Decimal(str(x))
        self._dX = Decimal(str(dX))

        if self._dX == 0:
            raise ValueError("Parameter dX cannot be zero!")

        self._fX = f(self._x)
        if self._fX == 0:
            raise ValueError("Function value f(x) cannot be zero!")

        self._fXdX = f(self._x + self._dX)

    @property
    def x(self) -> Decimal:
        """Get/Set the evaluation point."""
        return self._x

    @property
    def dX(self) -> Decimal:
        """Get/Set the small change used for derivative approximation."""
        return self._dX

    @property
    def fX(self) -> Decimal:
        """Get/Set the function value at x."""
        return self._fX

    @property
    def fXdX(self) -> Decimal:
        """Get/Set the function value at x + dX."""
        return self._fXdX

    def set_precision(self, precision: int) -> None:
        """
        Sets the Decimal precision for calculations.

        Args:
            precision (int): The number of significant digits.
        """
        getcontext().prec = precision

    def absCondNumber(self) -> Decimal:
        """
        Calculates the absolute condition number using Decimal arithmetic.

        Returns:
            Decimal: The absolute condition number.
        """
        return abs((self._fXdX - self._fX) / self._dX)

    def relativeCondNumber(self) -> Decimal:
        """
        Calculates the relative condition number using Decimal arithmetic.

        Returns:
            Decimal: The relative condition number.
        """
        return abs((self._fXdX - self._fX) * self._x / (self._fX * self._dX))


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
        self._int_part = str(self._dec_number).split(".")[0]
        self._frac_part = (
            str(self._dec_number).split(".")[1] if "." in str(self._dec_number) else ""
        )

    def sd(self) -> list[int]:
        """
        Returns the significant digits of the number.

        Returns:
            list[int]: A list of significant digits.
        """
        str_num = (
            f"{self._int_part}.{self._frac_part}" if self._frac_part else self._int_part
        )

        return [int(digit) for digit in str_num.replace(".", "").lstrip("0")]

    def vd(self, abs_err: Union[Decimal, float, int, str] = None) -> list[int]:
        """
        Finds the valid digits of the number based on the absolute error.

        Args:
            abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).

        Returns:
            list[int]: A list of valid digits.
        """
        if abs_err is None:
            abs_err = absolute_error(self._dec_number)
        else:
            abs_err = Decimal(str(abs_err))

        valid_digits = []
        digits = self._int_part + self._frac_part

        for i, digit in enumerate(digits):
            alpha = len(self._int_part) - 2 - i
            threshold = Decimal("5") * Decimal(10) ** Decimal(alpha)

            if threshold >= abs_err:
                valid_digits.append(int(digit))

        return valid_digits

    def dd(self, abs_err: Union[Decimal, float, int, str] = None) -> list[int]:
        """
        Finds the doubtful digits of the number based on the absolute error.

        Args:
            abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant)

        Returns:
            list[int]: A list of doubtful digits.
        """
        if abs_err is None:
            abs_err = absolute_error(self._dec_number)
        else:
            abs_err = Decimal(str(abs_err))

        doubtful_digits = []
        digits = self._int_part + self._frac_part

        for i, digit in enumerate(digits):
            alpha = len(self._int_part) - 2 - i
            threshold = Decimal("5") * Decimal(10) ** Decimal(alpha)

            if threshold < abs_err:
                doubtful_digits.append(int(digit))

        return doubtful_digits


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
        self._int_part = str(self._dec_number).split(".")[0]
        self._frac_part = (
            str(self._dec_number).split(".")[1] if "." in str(self._dec_number) else ""
        )

    def sd(self, num_digits: int = 1) -> Decimal:
        """
        Rounds the number to the specified number of significant digits.

        Args:
            num_digits (int, optional): The number of significant digits. Defaults to 1.

        Raises:
            ValueError: If num_digits is not greater than 0.

        Returns:
            Decimal: The rounded number.
        """
        if num_digits <= 0:
            raise ValueError("num_digits must be greater than 0")

        round_index = 0
        significant_digits_count = 0
        for i, digit in enumerate(self._int_part + self._frac_part):
            if significant_digits_count == num_digits:
                round_index = -(len(self._int_part) - i)
                break

            if digit != "0":
                significant_digits_count += 1

        if round_index <= 0:
            return self._dec_number
        return round(self._dec_number, round_index)

    def vd(
        self, abs_err: Union[Decimal, float, int, str] = None, num_digits: int = 1
    ) -> Decimal:
        """
        Rounds the number to the specified number of valid digits based on the absolute error.

        Args:
            abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).
            num_digits (int, optional): The number of valid digits. Defaults to 1.

        Raises:
            ValueError: If num_digits is not greater than 0.

        Returns:
            Decimal: The rounded number.
        """
        if num_digits <= 0:
            raise ValueError("num_digits must be greater than 0")

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
            threshold = Decimal("5") * Decimal(10) ** Decimal(alpha)

            if threshold >= abs_err:
                valid_digits_count += 1

        if round_index <= 0:
            return self._dec_number
        return round(self._dec_number, round_index)

    def dd(
        self, abs_err: Union[Decimal, float, int, str] = None, num_digits: int = 1
    ) -> Decimal:
        """
        Rounds the number to the specified number of doubtful digits based on the absolute error.

        Args:
            abs_err (Union[Decimal, float, int, str], optional): The absolute error. If not provided, it will be calculated (all numbers are considered significant).
            num_digits (int, optional): The number of doubtful digits. Defaults to 1.

        Raises:
            ValueError: If num_digits is not greater than 0.

        Returns:
            Decimal: The rounded number.
        """
        if num_digits <= 0:
            raise ValueError("num_digits must be greater than 0")

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
            threshold = Decimal("5") * Decimal(10) ** Decimal(alpha)

            if threshold < abs_err:
                doubtful_digits_count += 1

        if round_index <= 0:
            return self._dec_number
        return round(self._dec_number, round_index)


class ApproxNum:
    """A class representing approximate numbers with absolute error bounds.

    This class stores a numeric value along with its absolute error and supports
    basic arithmetic operations with proper error propagation. All calculations
    are performed using Decimal for high precision.

    Attributes:
        value (Union[Decimal, float, int, str]):
            The central value of the approximate number.
        abs_err (Union[Decimal, float, int, str], optional):
            The absolute error (uncertainty) of the number. If not provided, it will be calculated (all numbers are considered significant).
        rel_err (Union[Decimal, float, int, str], optional):
            The relative error (uncertainty) of the number. If not provided, it will be calculated (all numbers are considered significant).
    """

    def __init__(
        self,
        value: Union[Decimal, float, int, str],
        abs_err: Union[Decimal, float, int, str] = None,
        rel_err: Union[Decimal, float, int, str] = None,
        precision: int = 20,
    ):
        getcontext().prec = precision
        self._value = Decimal(str(value))

        if abs_err is None and rel_err is None:
            self._abs_err = absolute_error(value)
            self._rel_err = relative_error(value=value, abs_err=abs_err)
        elif abs_err is None:
            self._abs_err = absolute_error(value=value, rel_err=rel_err)
            self._rel_err = abs(Decimal(str(rel_err)))
        elif rel_err is None:
            self._abs_err = abs(Decimal(str(abs_err)))
            self._rel_err = relative_error(value=value, abs_err=abs_err)
        else:
            self._abs_err = abs(Decimal(str(abs_err)))
            self._rel_err = abs(Decimal(str(rel_err)))

    def __repr__(self) -> str:
        return f"ApproxNum(value={self.value}, abs_err={self._abs_err}, rel_err={self._rel_err})"

    def __str__(self) -> str:
        return f"{self.value} ± {self._abs_err} (δ = {self._rel_err})"

    def __add__(self, other: Union["ApproxNum", Decimal, float, int]) -> "ApproxNum":
        if isinstance(other, ApproxNum):
            value = self.value + other.value
            abs_err = self._abs_err + other.abs_err
            rel_err = (
                abs_err / abs(value)
                if (self.value + other.value) != 0
                else Decimal("Infinity")
            )
        else:
            other = Decimal(str(other))
            value = self.value + other
            abs_err = self._abs_err
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __radd__(self, other: Union[Decimal, float, int]) -> "ApproxNum":
        return self.__add__(other)

    def __sub__(self, other: Union["ApproxNum", Decimal, float, int]) -> "ApproxNum":
        if isinstance(other, ApproxNum):
            value = self.value - other.value
            abs_err = self._abs_err + other.abs_err
            rel_err = (
                abs_err / abs(value)
                if (self.value - other.value) != 0
                else Decimal("Infinity")
            )
        else:
            other = Decimal(str(other))
            value = self.value - other
            abs_err = self._abs_err
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __rsub__(self, other: Union[Decimal, float, int]) -> "ApproxNum":
        if isinstance(other, ApproxNum):
            return other.__sub__(self)
        other = Decimal(str(other))
        return ApproxNum(other - self.value, self._abs_err, self._rel_err)

    def __mul__(self, other: Union["ApproxNum", Decimal, float, int]) -> "ApproxNum":
        if isinstance(other, ApproxNum):
            value = self.value * other.value
            abs_err = abs(other.value) * self._abs_err + abs(self.value) * other.abs_err
            rel_err = self._rel_err + other.rel_err
        else:
            other = Decimal(str(other))
            value = self.value * other
            abs_err = abs(other) * self._abs_err
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __rmul__(self, other: Union[Decimal, float, int]) -> "ApproxNum":
        return self.__mul__(other)

    def __truediv__(
        self, other: Union["ApproxNum", Decimal, float, int]
    ) -> "ApproxNum":
        if isinstance(other, ApproxNum):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            value = self.value / other.value
            abs_err = (
                abs(other.value) * self._abs_err + abs(self.value) * other.abs_err
            ) / (other.value**2)
            rel_err = self._rel_err + other.rel_err
        else:
            other = Decimal(str(other))
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            value = self.value / other
            abs_err = self._abs_err / abs(other)
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __rtruediv__(self, other: Union[Decimal, float, int]) -> "ApproxNum":
        if isinstance(other, ApproxNum):
            return other.__truediv__(self)
        other = Decimal(str(other))
        if self.value == 0:
            raise ZeroDivisionError("Division by zero")
        return ApproxNum(
            other / self.value, abs(other) * self._abs_err / (self.value**2)
        )

    def __pow__(self, power: Union[int, float, Decimal]) -> "ApproxNum":
        power = Decimal(str(power))
        new_value = self.value**power
        new_error = abs(power) * (self.value ** (power - 1)) * self._abs_err
        return ApproxNum(new_value, new_error)

    def sqrt(self) -> "ApproxNum":
        if self.value < 0:
            raise ValueError("Square root of negative number")
        value = self.value.sqrt()
        abs_err = (
            self._abs_err / (Decimal(2) * value) if value != 0 else Decimal("Infinity")
        )
        rel_err = (
            self._abs_err / (Decimal(2) * self.value)
            if self.value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def sin(self) -> "ApproxNum":
        value = Decimal(str(math.sin(float(self.value))))
        abs_err = abs(Decimal(str(math.cos(float(self.value))))) * self._abs_err
        rel_err = (
            abs(Decimal(str(math.cos(float(self.value))))) / value * self._abs_err
            if value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def cos(self) -> "ApproxNum":
        value = Decimal(str(math.cos(float(self.value))))
        abs_err = abs(Decimal(str(math.sin(float(self.value))))) * self._abs_err
        rel_err = (
            abs(Decimal(str(math.tan(float(self.value))))) * self._abs_err
            if value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def tg(self) -> "ApproxNum":
        value = Decimal(str(math.tan(float(self.value))))
        cos_x = Decimal(str(math.cos(float(self.value))))
        abs_err = self._abs_err / (cos_x**2)
        sin_2x = Decimal(str(math.sin(2 * float(self.value))))
        rel_err = (
            (2 * self._abs_err) / abs(sin_2x) if sin_2x != 0 else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def ln(self) -> "ApproxNum":
        if self.value <= 0:
            raise ValueError("Natural logarithm is defined only for positive numbers")
        value = Decimal(str(math.log(float(self.value))))
        abs_err = self._abs_err / self.value
        rel_err = (
            self._abs_err / (abs(value) * self.value)
            if value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def lg(self) -> "ApproxNum":
        if self.value <= 0:
            raise ValueError("Decimal logarithm is defined only for positive numbers")
        value = Decimal(str(math.log10(float(self.value))))
        ln10 = Decimal(str(math.log(10)))
        abs_err = self._abs_err / (self.value * ln10)
        rel_err = (
            self._abs_err / (abs(value) * self.value * ln10)
            if value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def exp(self) -> "ApproxNum":
        value = Decimal(str(math.exp(float(self.value))))
        abs_err = value * self._abs_err
        rel_err = abs(self.value * self._abs_err)
        return ApproxNum(value, abs_err, rel_err)

    def pow10(self) -> "ApproxNum":
        value = Decimal(10) ** self.value
        ln10 = Decimal(str(math.log(10)))
        abs_err = value * ln10 * self._abs_err
        rel_err = ln10 * self._abs_err
        return ApproxNum(value, abs_err, rel_err)

    def arcsin(self) -> "ApproxNum":
        if abs(self.value) >= 1:
            raise ValueError("Arcsin is defined only for |x| < 1")
        value = Decimal(str(math.asin(float(self.value))))
        sqrt_val = (1 - self.value**2).sqrt()
        abs_err = self._abs_err / sqrt_val
        rel_err = (
            self._abs_err / (abs(value) * sqrt_val)
            if value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def arccos(self) -> "ApproxNum":
        if abs(self.value) >= 1:
            raise ValueError("Arccos is defined only for |x| < 1")
        value = Decimal(str(math.acos(float(self.value))))
        sqrt_val = (1 - self.value**2).sqrt()
        abs_err = self._abs_err / sqrt_val
        rel_err = (
            self._abs_err / (abs(value) * sqrt_val)
            if value != 0
            else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    def arctg(self) -> "ApproxNum":
        value = Decimal(str(math.atan(float(self.value))))
        denom = 1 + self.value**2
        abs_err = self._abs_err / denom
        rel_err = (
            self._abs_err / (abs(value) * denom) if value != 0 else Decimal("Infinity")
        )
        return ApproxNum(value, abs_err, rel_err)

    @property
    def value(self) -> Decimal:
        return self._value

    @property
    def abs_err(self) -> Decimal:
        return self._abs_err
    
    @abs_err.setter
    def abs_err(self, value: Union[Decimal, float, int, str]) -> None:
        if isinstance(value, ApproxNum):
            value = value.abs_err
        self._abs_err = Decimal(str(value))
        self._rel_err = relative_error(value=self.value, abs_err=self._abs_err)

    @property
    def rel_err(self) -> Decimal:
        return self._rel_err

    @rel_err.setter
    def rel_err(self, value: Union[Decimal, float, int, str]) -> None:
        if isinstance(value, ApproxNum):
            value = value.rel_err
        self._rel_err = Decimal(str(value))
        self._abs_err = absolute_error(value=self.value, rel_err=self._rel_err)


round_to = __RoundTo
cond_nums = __ConditionalNumbers
digits_analysis = __DigitsAnalysis
