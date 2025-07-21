import pytest
from autoroot.torch.complex.complex_utils import *

from .conftest import *


@pytest.mark.parametrize(
    "a,b",
    [
        (1 + 2j, 3 + 4j),  # Simple complex numbers
        (0 + 0j, 0 + 0j),  # Zero complex numbers
        (1 - 1j, -1 + 1j),  # Complex conjugates
        (2 + 3j, -2 - 3j),  # Negative complex numbers
        (5 + 0j, 0 + 5j),  # Purely real and purely imaginary
    ],
)
def test_addition_batch(a, b):
    """
    Test the addition of two complex numbers using the addition_batch function.
    This function uses pytest to run the test.
    """
    check_addition_batch(a, b)
    inputs = [a, b]
    func = addition_batch
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a, b",
    [
        (1 + 2j, 3 + 4j),  # Simple complex numbers
        (0 + 0j, 0 + 0j),  # Zero complex numbers
        (1 - 1j, -1 + 1j),  # Complex conjugates
        (2 + 3j, -2 - 3j),  # Negative complex numbers
        (5 + 0j, 0 + 5j),  # Purely real and purely imaginary
    ],
)
def test_product_of_2_complex_numbers_batch(a, b):
    """
    Test the product of two complex numbers using the product_of_2_complex_numbers_batch function.
    This function uses pytest to run the test.
    """
    check_product_of_2_complex_numbers_batch(a, b)
    inputs = [a, b]
    func = product_of_2_complex_numbers_batch
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a",
    [(4), (0), (-15), (-1), (3), (12.7)],
)
def test_sqrt_batch(a):
    """
    Test the square root calculation using the sqrt_batch function.
    This function uses pytest to run the test.
    """
    check_sqrt_batch(a)
    check_backward_sqrt_batch(a)


@pytest.mark.parametrize(
    "a, b",
    [
        (1 + 2j, 3),
        (0 + 4j, 0),
        (1 - 1j, -1),
        (2 + 3j, -2),
        (4j, 0),
    ],
)
def test_product_complex_real_batch(a, b):
    """
    Test the product of a complex number and a real number using the product_of_2_complex_numbers_batch function.
    This function uses pytest to run the test.
    """
    check_product_complex_real_batch(a, b)
    inputs = [a, b]
    func = product_complex_real_batch
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a",
    [(1 + 2j), (0 + 4j), (3 - 4j), (-5 + 6j), (7 + 8j), (-9 - 10j)],
)
def test_inverse_complex_number(a):
    """
    Test the inverse of a complex number
    This function uses pytest to run the test.
    """
    check_inverse_complex_batch(a)
    inputs = [a]
    func = inverse_complex_number
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a, k",
    [
        (1 + 2j, 2),  # Square of a complex number
        (0 + 4j, -3),  # Cube of a purely imaginary number
        (3 - 4j, 1),  # Identity case
        (-5 + 6j, 0),  # Zero power
        (7 + 8j, -1),  # Inverse power
        (-9 - 10j, 4),  # Higher power
    ],
)
def test_complex_number_power_k_batch(a, k):
    """
    Test the power of a complex number using the complex_number_power_k_batch function.
    This function uses pytest to run the test.
    """
    check_complex_number_power_k_batch(a, k)
    inputs = [a]
    func = lambda x: complex_number_power_k_batch(x, k)
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a",
    [(1 + 2j), (0 + 4j), (3 - 4j), (-5 + 6j), (7 + 8j), (-9 - 10j)],
)
def test_argument_batch(a):
    """
    Test the argument of a complex number using the argument_batch function.
    This function uses pytest to run the test.
    """
    check_argument_batch(a)
    inputs = [a]
    func = argument_batch
    check_backward(inputs, func, [0])
    output_shape = (1, 1)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a",
    [(1 + 2j), (0 + 4j), (3 - 4j), (-5 + 6j), (7 + 8j), (-9 - 10j)],
)
def test_module_batch(a):
    """
    Test the module of a complex number using the module_batch function.
    This function uses pytest to run the test.
    """
    check_module_batch(a)
    inputs = [a]
    func = module_batch
    check_backward(inputs, func, [0])
    output_shape = (1, 1)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a",
    [(1), (68.4), (4), (0), (-15), (-1), (3), (12.7)],
)
def test_cube_root_batch(a):
    """
    Test the square root of 3 using the cube_root_batch function.
    This function uses pytest to run the test.
    """
    check_cube_root_batch(a)
    inputs = [a]
    func = cube_root_batch
    # check_backward(inputs, func) # not working
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a",
    [(1 + 2j), (0 + 4j), (3 - 4j), (-5 + 6j), (7 + 8j), (-9 - 10j)],
)
def test_sqrt_complex_batch(a):
    """
    Test the square root of a complex number using the sqrt_complex_batch function.
    This function uses pytest to run the test.
    """
    check_sqrt_complex_batch(a)
    inputs = [a]
    func = sqrt_complex_batch
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a, b",
    [
        (1 + 2j, 3 + 4j),  # Simple complex numbers
        (0 + 0j, 0 + 8j),  # Zero complex numbers
        (1 - 1j, -1 + 1j),  # Complex conjugates
        (2 + 3j, -2 - 3j),  # Negative complex numbers
        (5 + 0j, 0 + 5j),  # Purely real and purely imaginary
    ],
)
def test_division_2_complex_numbers(a, b):
    """
    Test the division of two complex numbers using the division_2_complex_numbers function.
    This function uses pytest to run the test.
    """
    check_division_2_complex_numbers(a, b)
    inputs = [a, b]
    func = division_2_complex_numbers
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)


@pytest.mark.parametrize(
    "a, b",
    [
        (1 + 2j, 3),  # Simple complex number and real number
        (0 + 4j, 0),  # Zero complex number and zero real number
        (1 - 1j, -1),  # Complex conjugate and negative real number
        (2 + 3j, -2),  # Negative real number
        (4j, 0),  # Purely imaginary and zero real number
    ],
)
def test_addition_complex_real_batch(a, b):
    """
    Test the addition of a complex number and a real number using the addition_complex_real_batch function.
    This function uses pytest to run the test.
    """
    check_addition_complex_real_batch(a, b)
    inputs = [a]
    func = lambda x: addition_complex_real_batch(x, torch.tensor([b], dtype=dtype).unsqueeze(0))
    check_backward(inputs, func)
    output_shape = (1, 2)
    check_shape(inputs, func, output_shape)
