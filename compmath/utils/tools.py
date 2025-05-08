import numpy as np
from typing import Union
from decimal import Decimal


__all__ = ["to_decimal"]


def to_decimal(
    obj: Union[float, int, str, list, np.ndarray],
) -> Union[Decimal, np.ndarray]:
    """
    Convert input to Decimal or an array of Decimals.

    Args:
        x: A number or list/array of numbers.

    Returns:
        Decimal or np.ndarray: Converted decimal(s).
    """
    if isinstance(obj, list) or isinstance(obj, np.ndarray):
        return np.array([Decimal(str(x)) for x in obj], dtype=Decimal)
    return Decimal(str(obj))
