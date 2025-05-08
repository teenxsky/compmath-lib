import numpy as np
from decimal import Decimal, getcontext
from typing import Callable, Union, Literal


getcontext().prec = 20
__all__ = ["cond_nums"]


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

    def abs(
        self, return_type: Literal["Decimal", "float"] = "float"
    ) -> Union[Decimal, np.float64]:
        """
        Calculates the absolute condition number using Decimal arithmetic.

        Args:
          return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Returns:
          Union[Decimal, np.float64]: The absolute condition number.
        """
        result = abs((self._fXdX - self._fX) / self._dX)
        return result if return_type == "Decimal" else np.float64(result)

    def rel(
        self, return_type: Literal["Decimal", "float"] = "float"
    ) -> Union[Decimal, np.float64]:
        """
        Calculates the relative condition number using Decimal arithmetic.

        Args:
          return_type (Literal["Decimal", "float"], optional): The return type of the result. Defaults to "float".

        Returns:
          Union[Decimal, np.float64]: The relative condition number.
        """
        result = abs((self._fXdX - self._fX) * self._x / (self._fX * self._dX))
        return result if return_type == "Decimal" else np.float64(result)


cond_nums = __ConditionalNumbers
