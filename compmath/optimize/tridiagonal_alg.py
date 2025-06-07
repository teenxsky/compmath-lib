from typing import List, Union

import numpy as np

__all__ = ['thomasalg']


def thomasalg(
    main_diag: Union[List[float], np.ndarray],
    lower_diag: Union[List[float], np.ndarray],
    upper_diag: Union[List[float], np.ndarray],
    rhs_vector: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Solves lower_diag tridiagonal system of linear equations using the Thomas algorithm (tridiagonal matrix algorithm).

    The system must be in the form:
        a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i, for i = 0, ..., n-1,
    where:
        - a[0] is unused and typically set to 0
        - b[n-1] is unused and typically set to 0

    Args:
        main_diag (Union[List[float], np.ndarray]): Main diagonal coefficients (b_0 to b_{n-1}), length n.
        lower_diag (Union[List[float], np.ndarray]): Sub-diagonal coefficients (a_1 to a_{n-1}), length n-1.
        upper_diag (Union[List[float], np.ndarray]): Super-diagonal coefficients (c_0 to c_{n-2}), length n-1.
        rhs_vector (Union[List[float], np.ndarray]): Right-hand side vector, length n.

    Returns:
        np.ndarray: Solution vector x of length n.

    Raises:
        ValueError: If the lengths of input arrays do not match the expected sizes.
    """
    n = len(main_diag)

    if len(lower_diag) != n - 1:
        raise ValueError('Length of lower_diag must be n - 1')
    if len(upper_diag) != n - 1:
        raise ValueError('Length of upper_diag must be n - 1')
    if len(rhs_vector) != n:
        raise ValueError('Length of rhs_vector must be n')

    lower_diag = np.concatenate([[0], np.asarray(lower_diag, dtype=float)])
    upper_diag = np.concatenate([np.asarray(upper_diag, dtype=float), [0]])
    main_diag = np.asarray(main_diag, dtype=float)
    rhs_vector = np.asarray(rhs_vector, dtype=float)

    alpha = np.zeros(n)
    beta = np.zeros(n)

    for i in range(n - 1):
        denominator = main_diag[i] + lower_diag[i] * alpha[i]
        alpha[i + 1] = -upper_diag[i] / denominator
        beta[i + 1] = (rhs_vector[i] - lower_diag[i] * beta[i]) / denominator

    x = np.zeros(n)
    x[n - 1] = (rhs_vector[n - 1] - lower_diag[n - 1] * beta[n - 1]) / (
        main_diag[n - 1] + lower_diag[n - 1] * alpha[n - 1]
    )

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x
