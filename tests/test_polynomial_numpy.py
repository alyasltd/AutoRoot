import numpy as np
import pytest
from autoroot.numpy.cubic import polynomial_root_calculation_3rd_degree_numpy
from autoroot.numpy.quartic import polynomial_root_calculation_4th_degree_ferrari_numpy
from numpy.polynomial import Polynomial

from .conftest import compute_diff, sort_roots

precision = 1e-7


def compute_diff_numpy(roots_np, roots_gt):
    diff = roots_np[:] - roots_gt[:]
    dist_real = np.real(diff)
    dist_imag = np.imag(diff)
    dist = dist_real**2 + dist_imag**2
    return np.min(dist, -1)


def sort_roots_numpy(roots):
    # sort solution by increasing real values + (max(real)+1)*img
    roots_real = roots.real
    roots_imag = roots.imag
    indices = np.argsort(roots_real + np.max(roots_real + 1) * roots_imag)

    sorted_roots = roots[indices[np.arange(len(roots))]]

    return sorted_roots


@pytest.mark.parametrize(
    "a, b, c, d",
    [
        (1, -6, 11, -6),  # Roots: 1, 2, 3
        (1, -3, 3, -1),  # Roots: 1, 1, 1
        (2, -4, 2, -1),
        (1, 0, -4, 4),
    ],
)
def test_cubic_numpy(a, b, c, d):
    """
    Test the polynomial root calculation for a cubic polynomial.
    This function uses pytest to run the test.
    """

    # Calculation of the polynomial
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    roots_np = sort_roots_numpy(polynomial_root_calculation_3rd_degree_numpy(a, b, c, d))
    print(f"Roots of the cubic polynomial {a}x^3 + {b}x^2 + {c}x + {d}: {roots_np}")

    poly = Polynomial([d, c, b, a])
    roots_gt = sort_roots_numpy(poly.roots())
    print("\n roots_gt", roots_gt)

    for r in roots_np:
        # Calculation of the polynomial applied to the root
        y = f(r, a, b, c, d)
        np.testing.assert_allclose(
            np.abs(np.real(y)), 0, atol=precision
        )  # Check if the polynomial evaluated at the root is close to zero (<10^(-10))
        np.testing.assert_allclose(
            np.abs(np.imag(y)), 0, atol=precision
        )  # Check if the polynomial evaluated at the root is close to zero (<10^(-10))

    dist = compute_diff_numpy(roots_np, roots_gt)
    np.testing.assert_allclose(dist, 0.0 * dist, atol=precision)


@pytest.mark.parametrize(
    "a0, a1, a2, a3, a4",
    [
        (1, -6, 11, -6, 1),  # 2 real roots
        (1, -4, 6, -4, 1),  # Roots: 1, 1, 1,1
        (2, -8, 8, -2, 2),  # 2 reals roots, 2 complex conjugates
        (1, -5, 6, -4, 1),  # 2 reals roots, 2 complex conjugates
        (1, 3, -4, 1, 1),  # 2 reals roots, 2 complex conjugates
        (1, 0, 0, 0, 1),  # 4 complex roots
        (1, -2, 7, -8, 1),  # 2 reals roots, 2 complex conjugates
        (0.0, -6, 11, -6, 1),  #
    ],
)
def test_quartic(a0, a1, a2, a3, a4):
    """
    Test the polynomial root calculation for a quartic polynomial using Ferrari's method.
    This function uses pytest to run the test.
    """

    # Calculation of the polynomial
    def f(x, a0, a1, a2, a3, a4):
        return a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0

    # Computation of the roots
    roots_numpy = sort_roots_numpy(
        polynomial_root_calculation_4th_degree_ferrari_numpy(a0, a1, a2, a3, a4)
    )

    for r in roots_numpy:
        # Calculation of the polynomial applied to the root
        y = f(r, a0, a1, a2, a3, a4)
        np.testing.assert_allclose(
            np.abs(np.real(y)), 0, atol=precision
        )  # Check if the polynomial evaluated at the root is close to zero (<10^(-10))
        np.testing.assert_allclose(
            np.abs(np.imag(y)), 0, atol=precision
        )  # Check if the polynomial evaluated at the root is close to zero (<10^(-10))
        # Check if the polynomial evaluated at the root is close to zero (<10^(-10))

    # compare the roots with the one found using numpy
    poly = Polynomial([a0, a1, a2, a3, a4])
    roots_gt = poly.roots()
    roots_gt = sort_roots_numpy(roots_gt)

    dist = compute_diff_numpy(roots_numpy, roots_gt)
    np.testing.assert_allclose(dist, 0.0 * dist, atol=precision)
