import numpy as np


__all__ = ["factorial"]


def factorial(value: int) -> int:
    """
    Calculate the factorial of a given integer.

    Args:
        n (int): Input integer.

    Returns:
        int: Factorial of n.
    """
    return np.prod([i for i in range(1, value + 1)])
