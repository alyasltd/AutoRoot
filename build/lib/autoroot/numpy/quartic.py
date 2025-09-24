import numpy as np
from autoroot.numpy.cubic import polynomial_root_calculation_3rd_degree_numpy


def polynomial_root_calculation_4th_degree_ferrari_numpy(
    a0: float, a1: float, a2: float, a3: float, a4: float
) -> np.ndarray:
    """
    Calculate the roots of a quartic polynomial using Ferrari's method.
    https://en.wikipedia.org/wiki/Quartic_function#Ferrari's_method
    Args:
        a0 : Constant term, shape (batch_size, 1)
        a1 : Coefficient of x^1, shape (batch_size, 1)
        a2 : Coefficient of x^2, shape (batch_size, 1)
        a3 : Coefficient of x^3, shape (batch_size, 1)
        a4 : Coefficient of x^4, shape (batch_size, 1)
    Returns:
        Array: Roots of the polynomial, shape (4,)
        Each root is represented as a complex number
    """

    # Reduce the quartic equation to the form : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    a: float = a3 / a4
    b: float = a2 / a4
    c: float = a1 / a4
    d: float = a0 / a4

    # Computation of the coefficients of the Ferrari's Method
    S: float = a / 4
    b0: float = d - c * S + b * S**2 - 3 * S**4
    b1: float = c - 2 * b * S + 8 * S**3
    b2: float = b - 6 * S**2

    # Solve the cubic equation m^3 + b2*m^2 + (b2^2/4  - b0)*m - b1^2/8 = 0
    x_cube: np.ndarray = polynomial_root_calculation_3rd_degree_numpy(
        1, b2, (b2**2) / 4 - b0, (-(b1**2)) / 8
    )

    # Find a real and positive solution
    alpha_0: float = 0
    for r in x_cube:
        if np.isclose(np.imag(r), 0) and np.real(r) > 0:
            alpha_0 = r

    if alpha_0 != 0:
        x1: np.complex128 = (
            np.sqrt(alpha_0 / 2)
            - S
            + np.sqrt(-alpha_0 / 2 - b2 / 2 - b1 / (2 * np.sqrt(2 * alpha_0)), dtype=complex)
        )
        x2: np.complex128 = (
            np.sqrt(alpha_0 / 2)
            - S
            - np.sqrt(-alpha_0 / 2 - b2 / 2 - b1 / (2 * np.sqrt(2 * alpha_0)), dtype=complex)
        )
        x3: np.complex128 = (
            -np.sqrt(alpha_0 / 2)
            - S
            + np.sqrt(-alpha_0 / 2 - b2 / 2 + b1 / (2 * np.sqrt(2 * alpha_0)), dtype=complex)
        )
        x4: np.complex128 = (
            -np.sqrt(alpha_0 / 2)
            - S
            - np.sqrt(-alpha_0 / 2 - b2 / 2 + b1 / (2 * np.sqrt(2 * alpha_0)), dtype=complex)
        )

    else:
        x1 = -S + np.sqrt(-b2 / 2 + np.sqrt((b2**2) / 4 - b0), dtype=complex)
        x2 = -S - np.sqrt(-b2 / 2 + np.sqrt((b2**2) / 4 - b0), dtype=complex)
        x3 = -S + np.sqrt(-b2 / 2 - np.sqrt((b2**2) / 4 - b0), dtype=complex)
        x4 = -S - np.sqrt(-b2 / 2 - np.sqrt((b2**2) / 4 - b0), dtype=complex)
    return np.array([x1, x2, x3, x4])
