import math
from decimal import Decimal, getcontext
from typing import Union

from .errors import absolute_error, relative_error

getcontext().prec = 20
__all__ = ['ApproxNum']


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
            self._abs_err = absolute_error(value, return_type='Decimal')
            self._rel_err = relative_error(
                value=value, abs_err=abs_err, return_type='Decimal'
            )
        elif abs_err is None:
            self._abs_err = absolute_error(
                value=value, rel_err=rel_err, return_type='Decimal'
            )
            self._rel_err = abs(Decimal(str(rel_err)))
        elif rel_err is None:
            self._abs_err = abs(Decimal(str(abs_err)))
            self._rel_err = relative_error(
                value=value, abs_err=abs_err, return_type='Decimal'
            )
        else:
            self._abs_err = abs(Decimal(str(abs_err)))
            self._rel_err = abs(Decimal(str(rel_err)))

    def __repr__(self) -> str:
        return f'ApproxNum(value={self.value}, abs_err={self._abs_err}, rel_err={self._rel_err})'

    def __str__(self) -> str:
        return f'{self.value} ± {self._abs_err} (δ = {self._rel_err})'

    def __add__(
        self, other: Union['ApproxNum', Decimal, float, int]
    ) -> 'ApproxNum':
        if isinstance(other, ApproxNum):
            value = self.value + other.value
            abs_err = self._abs_err + other.abs_err
            rel_err = (
                abs_err / abs(value)
                if (self.value + other.value) != 0
                else Decimal('Infinity')
            )
        else:
            other = Decimal(str(other))
            value = self.value + other
            abs_err = self._abs_err
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __radd__(self, other: Union[Decimal, float, int]) -> 'ApproxNum':
        return self.__add__(other)

    def __sub__(
        self, other: Union['ApproxNum', Decimal, float, int]
    ) -> 'ApproxNum':
        if isinstance(other, ApproxNum):
            value = self.value - other.value
            abs_err = self._abs_err + other.abs_err
            rel_err = (
                abs_err / abs(value)
                if (self.value - other.value) != 0
                else Decimal('Infinity')
            )
        else:
            other = Decimal(str(other))
            value = self.value - other
            abs_err = self._abs_err
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __rsub__(self, other: Union[Decimal, float, int]) -> 'ApproxNum':
        if isinstance(other, ApproxNum):
            return other.__sub__(self)
        other = Decimal(str(other))
        return ApproxNum(other - self.value, self._abs_err, self._rel_err)

    def __mul__(
        self, other: Union['ApproxNum', Decimal, float, int]
    ) -> 'ApproxNum':
        if isinstance(other, ApproxNum):
            value = self.value * other.value
            abs_err = (
                abs(other.value) * self._abs_err
                + abs(self.value) * other.abs_err
            )
            rel_err = self._rel_err + other.rel_err
        else:
            other = Decimal(str(other))
            value = self.value * other
            abs_err = abs(other) * self._abs_err
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __rmul__(self, other: Union[Decimal, float, int]) -> 'ApproxNum':
        return self.__mul__(other)

    def __truediv__(
        self, other: Union['ApproxNum', Decimal, float, int]
    ) -> 'ApproxNum':
        if isinstance(other, ApproxNum):
            if other.value == 0:
                raise ZeroDivisionError('Division by zero')
            value = self.value / other.value
            abs_err = (
                abs(other.value) * self._abs_err
                + abs(self.value) * other.abs_err
            ) / (other.value**2)
            rel_err = self._rel_err + other.rel_err
        else:
            other = Decimal(str(other))
            if other == 0:
                raise ZeroDivisionError('Division by zero')
            value = self.value / other
            abs_err = self._abs_err / abs(other)
            rel_err = None
        return ApproxNum(value, abs_err, rel_err)

    def __rtruediv__(self, other: Union[Decimal, float, int]) -> 'ApproxNum':
        if isinstance(other, ApproxNum):
            return other.__truediv__(self)
        other = Decimal(str(other))
        if self.value == 0:
            raise ZeroDivisionError('Division by zero')
        return ApproxNum(
            other / self.value, abs(other) * self._abs_err / (self.value**2)
        )

    def __pow__(self, power: Union[int, float, Decimal]) -> 'ApproxNum':
        power = Decimal(str(power))
        new_value = self.value**power
        new_error = abs(power) * (self.value ** (power - 1)) * self._abs_err
        return ApproxNum(new_value, new_error)

    def sqrt(self) -> 'ApproxNum':
        if self.value < 0:
            raise ValueError('Square root of negative number')
        value = self.value.sqrt()
        abs_err = (
            self._abs_err / (Decimal(2) * value)
            if value != 0
            else Decimal('Infinity')
        )
        rel_err = (
            self._abs_err / (Decimal(2) * self.value)
            if self.value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def sin(self) -> 'ApproxNum':
        value = Decimal(str(math.sin(float(self.value))))
        abs_err = abs(Decimal(str(math.cos(float(self.value))))) * self._abs_err
        rel_err = (
            abs(Decimal(str(math.cos(float(self.value)))))
            / value
            * self._abs_err
            if value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def cos(self) -> 'ApproxNum':
        value = Decimal(str(math.cos(float(self.value))))
        abs_err = abs(Decimal(str(math.sin(float(self.value))))) * self._abs_err
        rel_err = (
            abs(Decimal(str(math.tan(float(self.value))))) * self._abs_err
            if value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def tg(self) -> 'ApproxNum':
        value = Decimal(str(math.tan(float(self.value))))
        cos_x = Decimal(str(math.cos(float(self.value))))
        abs_err = self._abs_err / (cos_x**2)
        sin_2x = Decimal(str(math.sin(2 * float(self.value))))
        rel_err = (
            (2 * self._abs_err) / abs(sin_2x)
            if sin_2x != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def ln(self) -> 'ApproxNum':
        if self.value <= 0:
            raise ValueError(
                'Natural logarithm is defined only for positive numbers'
            )
        value = Decimal(str(math.log(float(self.value))))
        abs_err = self._abs_err / self.value
        rel_err = (
            self._abs_err / (abs(value) * self.value)
            if value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def lg(self) -> 'ApproxNum':
        if self.value <= 0:
            raise ValueError(
                'Decimal logarithm is defined only for positive numbers'
            )
        value = Decimal(str(math.log10(float(self.value))))
        ln10 = Decimal(str(math.log(10)))
        abs_err = self._abs_err / (self.value * ln10)
        rel_err = (
            self._abs_err / (abs(value) * self.value * ln10)
            if value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def exp(self) -> 'ApproxNum':
        value = Decimal(str(math.exp(float(self.value))))
        abs_err = value * self._abs_err
        rel_err = abs(self.value * self._abs_err)
        return ApproxNum(value, abs_err, rel_err)

    def pow10(self) -> 'ApproxNum':
        value = Decimal(10) ** self.value
        ln10 = Decimal(str(math.log(10)))
        abs_err = value * ln10 * self._abs_err
        rel_err = ln10 * self._abs_err
        return ApproxNum(value, abs_err, rel_err)

    def arcsin(self) -> 'ApproxNum':
        if abs(self.value) >= 1:
            raise ValueError('Arcsin is defined only for |x| < 1')
        value = Decimal(str(math.asin(float(self.value))))
        sqrt_val = (1 - self.value**2).sqrt()
        abs_err = self._abs_err / sqrt_val
        rel_err = (
            self._abs_err / (abs(value) * sqrt_val)
            if value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def arccos(self) -> 'ApproxNum':
        if abs(self.value) >= 1:
            raise ValueError('Arccos is defined only for |x| < 1')
        value = Decimal(str(math.acos(float(self.value))))
        sqrt_val = (1 - self.value**2).sqrt()
        abs_err = self._abs_err / sqrt_val
        rel_err = (
            self._abs_err / (abs(value) * sqrt_val)
            if value != 0
            else Decimal('Infinity')
        )
        return ApproxNum(value, abs_err, rel_err)

    def arctg(self) -> 'ApproxNum':
        value = Decimal(str(math.atan(float(self.value))))
        denom = 1 + self.value**2
        abs_err = self._abs_err / denom
        rel_err = (
            self._abs_err / (abs(value) * denom)
            if value != 0
            else Decimal('Infinity')
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
        self._rel_err = relative_error(
            value=self.value, abs_err=self._abs_err, return_type='Decimal'
        )

    @property
    def rel_err(self) -> Decimal:
        return self._rel_err

    @rel_err.setter
    def rel_err(self, value: Union[Decimal, float, int, str]) -> None:
        if isinstance(value, ApproxNum):
            value = value.rel_err
        self._rel_err = Decimal(str(value))
        self._abs_err = absolute_error(
            value=self.value, rel_err=self._rel_err, return_type='Decimal'
        )
