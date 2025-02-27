from math import isclose
from typing import Callable


class ConditionalNumbers:
    def __init__(
        self, f: Callable[[float], float], x: float = 1, dX: float = 0.001
    ) -> None:
        if isclose(dX, 0):
            raise ValueError("Parameter dX cannot be zero!")
        if isclose(f(x), 0):
            raise ValueError("Func value f(x) cannot be zero!")

        self.x = x
        self.dX = dX
        self.fX = f(x)
        self.fXdX = f(x + dX)

    def absCondNumber(self) -> float:
        return abs((self.fXdX - self.fX) / self.dX)

    def relativeCondNumber(self) -> float:
        return abs(((self.fXdX - self.fX) * self.x / (self.fX * self.dX)))
