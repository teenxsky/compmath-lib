from decimal import Decimal
from typing import Union

import numpy as np

__all__ = ['to_decimal']


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
        decimal_list = [Decimal(str(x)) for x in obj]
        return np.array(decimal_list, dtype=object)
    return Decimal(str(obj))
