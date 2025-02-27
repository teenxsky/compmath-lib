from math import isclose
from typing import Callable


class ConditionalNumbers:
    """A class to compute the absolute and relative condition numbers of a given function.

    The condition number provides a measure of how sensitive the output of a function is to changes in its input.
    This class allows users to calculate both absolute and relative condition numbers for a specified function
    at a given point.

    Attributes:
        x (float): The point at which the condition number is evaluated.
        dX (float): The small change in x used to compute the derivative approximation.
        fX (float): The value of the function at x.
        fXdX (float): The value of the function at x + dX.
    """

    def __init__(
        self, f: Callable[[float], float], x: float = 1, dX: float = 0.001
    ) -> None:
        """Initializes the ConditionalNumbers class with a function, a point, and a small change.

        Args:
            f (Callable[[float], float]): The function for which the condition numbers are to be calculated.
            x (float, optional): The point at which to evaluate the function. Defaults to 1.
            dX (float, optional): The small change in x for derivative approximation. Defaults to 0.001.

        Raises:
            ValueError: If dX is zero.
            ValueError: If the function value f(x) is zero.
        """

        if isclose(dX, 0):
            raise ValueError("Parameter dX cannot be zero!")
        if isclose(f(x), 0):
            raise ValueError("Func value f(x) cannot be zero!")

        self.x = x
        self.dX = dX
        self.fX = f(x)
        self.fXdX = f(x + dX)

    def absCondNumber(self) -> float:
        """Calculates the absolute condition number of the function at the specified point.

        The absolute condition number is defined as the absolute value of the derivative of the function
        at the point x, approximated using the small change dX.

        Returns:
            float: The absolute condition number.
        """

        return abs((self.fXdX - self.fX) / self.dX)

    def relativeCondNumber(self) -> float:
        """Calculates the relative condition number of the function at the specified point.

        The relative condition number is defined as the absolute value of the ratio of the absolute change
        in the function value to the product of the function value and the change in x.

        Returns:
            float: The relative condition number.
        """
        
        return abs(((self.fXdX - self.fX) * self.x / (self.fX * self.dX)))
