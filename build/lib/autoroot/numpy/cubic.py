import numpy as np


def cube_root(x: np.complex128) -> np.complex128:
    """
    Compute the cube root for a complex number
    Args :
        x : complex number
    Returns :
        Cube root of x : complex number
    """
    if np.real(x) >= 0:
        return x ** (1 / 3)
    else:
        return -((-x) ** (1 / 3))


def polynomial_root_calculation_3rd_degree_numpy(
    a: float, b: float, c: float, d: float
) -> np.ndarray:
    """
    Calculate the roots of a cubic polynomial using Cardano's method.
    https://en.wikipedia.org/wiki/Cubic_equation
    Args:
        a : Coefficient of x^3, shape (batch_size, 1)
        b : Coefficient of x^2, shape (batch_size, 1)
        c : Coefficient of x, shape (batch_size, 1)
        d : Constant term, shape (batch_size, 1)
    Returns:
        Array of complex number : Roots of the polynomial, shape (3,)
    """
    # Calculation of the discriminant
    p: float = (3 * a * c - b**2) / (3 * a**2)
    q: float = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    delta: float = -4 * p**3 - 27 * q**2

    roots: list = []

    j_: np.complex128 = np.exp((2 * 1j * np.pi) / 3)

    for k in range(3):
        u_k: np.complex128 = j_**k * cube_root(0.5 * (-q + np.sqrt(-delta / 27, dtype=complex)))
        v_k: np.complex128 = j_ ** (-k) * cube_root(
            0.5 * (-q - np.sqrt(-delta / 27, dtype=complex))
        )

        roots.append((u_k + v_k) - b / (3 * a))

    return np.array(roots)
