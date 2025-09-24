import math

import torch

# from types import Tensor
from torch import Tensor


def addition_batch(a: Tensor, b: Tensor) -> Tensor:
    """
    Adds two batches of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the first
        batch of complex numbers.
        b : A tensor of shape (batch_size, 2) representing the second
        batch of complex numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the sum of the two
        batches of complex numbers.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])
    return torch.stack([a[:, 0] + b[:, 0], a[:, 1] + b[:, 1]], dim=-1)  # (batch_size, 2)


def product_of_2_complex_numbers_batch(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiplies two batches of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the first
        batch of complex numbers.
        b : A tensor of shape (batch_size, 2) representing the second
        batch of complex numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the product of the two
        batches of complex numbers.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])

    real_part: Tensor = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1]  # (batch_size, )
    imag_part: Tensor = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]  # (batch_size, )
    return torch.stack([real_part, imag_part], dim=-1)  # (batch_size, 2)


def sqrt_batch(a: Tensor) -> Tensor:
    """
    Computes the square root of a batch of real numbers.
    Each number is represented as a tensor of shape (batch_size, 1).
    Args:
        a : A tensor of shape (batch_size, 1) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the square root of the
        batch of real numbers, where the first element is the real part and the
        second element is the imaginary part.
    """
    eps = 1e-12

    # Clamp absolute value to avoid sqrt(0) in backward pass
    abs_a: Tensor = torch.clamp(torch.abs(a), min=eps)  # (batch_size,1)
    sqrt_a: Tensor = torch.sqrt(abs_a)  # element-wise (batch_size,1)

    # real if a >= 0, imaginary otherwise
    real_part: Tensor = torch.where(a >= 0, sqrt_a, torch.zeros_like(a))  # (batch_size,1)
    imag_part: Tensor = torch.where(a < 0, sqrt_a, torch.zeros_like(a))  # (batch_size,1)

    return torch.cat((real_part, imag_part), dim=1)  # (batch_size, 2)


def product_complex_real_batch(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiplies a batch of complex numbers by a batch of real numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part. Each real number is represented as a tensor of shape
    (batch_size, 1).
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
        b : A tensor of shape (batch_size, 1) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the product of the batch
        of complex numbers and the batch of real numbers, where the first element
        is the real part and the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[,a_real, a_imag]) # (batch_size, 2)
    # b is a real number (batch_size,1) # (batch_size, 1)
    return a * b  # (batch_size, 2)


def inverse_complex_number(a: Tensor) -> Tensor:
    """
    Computes the inverse of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the inverse of the batch
        of complex numbers, where the first element is the real part and the
        second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # a need to be =/= 0
    # Returns the inverse of the complex number a

    denom: Tensor = a[:, 0] ** 2 + a[:, 1] ** 2
    if torch.any(denom == 0):
        raise ValueError("Cannot compute inverse of zero complex number")
    return torch.stack([a[:, 0] / denom, -a[:, 1] / denom], dim=-1)  # (batch_size, 2)


def complex_number_power_k_batch(a: Tensor, k: int) -> Tensor:
    """
    Computes the k-th power of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
        k : An integer representing the power to which the complex numbers are
        raised.
    Returns:
        A tensor of shape (batch_size, 2) representing the k-th power of the
        batch of complex numbers, where the first element is the real part and
        the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # k is an integer

    real_a: Tensor = a[:, :1]  # (batch_size, 1)
    imag_a: Tensor = a[:, 1:]  # (batch_size, 1)
    r: Tensor = torch.sqrt(real_a**2 + imag_a**2)
    # eps_tensor: Tensor = torch.finfo(torch.float32).eps + torch.tensor(0.0)*real_a

    theta: Tensor = torch.atan2(imag_a, real_a)  # safe even when real_a == 0

    result_real: Tensor = r**k * torch.cos(theta * k)
    result_imag: Tensor = r**k * torch.sin(theta * k)

    result: Tensor = torch.cat((result_real, result_imag), dim=1)  # (batch_size, 2)

    return result


def argument_batch(a: Tensor) -> Tensor:  # potentiellemen pb si (0,0)
    """
    Computes the argument (angle) of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 1) representing the argument of the
        batch of complex numbers, where each element is the angle in radians.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    real: Tensor = a[:, :1]
    imag: Tensor = a[:, 1:]
    theta: Tensor = torch.atan2(imag, real)  # safe computation for all quadrants
    return theta  # (batch_size, 1)


def module_batch_old(a: Tensor) -> Tensor:
    """
    Computes the modulus of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 1) representing the modulus of the
        batch of complex numbers, where each element is the modulus.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    return torch.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2).unsqueeze(-1)  # (batch_size, 1)


def module_batch(a: Tensor) -> Tensor:
    """
    Computes the modulus of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 1) representing the modulus of the
        batch of complex numbers, where each element is the modulus.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    return torch.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2).unsqueeze(-1)  # (batch_size, 1)


def cube_root_batch(a: Tensor) -> Tensor:
    """
    Computes the cube root of a batch of complex numbers.
    Each number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the cube root of the
        batch of real numbers, where the first element is the real part and the
        second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])

    real_part_if_real: Tensor = torch.where(
        a[:, 0] >= 0, a[:, 0] ** (1 / 3), -((-a[:, 0]) ** (1 / 3))
    )  # (batch_size, 1)
    imag_part_if_real: Tensor = real_part_if_real * 0.0  # (batch_size, 1)
    result_if_real: Tensor = torch.stack(
        (real_part_if_real, imag_part_if_real), dim=-1
    )  # (batch_size, 2)

    r: Tensor = module_batch(a)  # (batch_size, 1)
    teta: Tensor = argument_batch(a)  # (batch_size, 1)
    r_pow_1_div3: Tensor = torch.pow(r[:, 0], 1 / 3)  # (batch_size, 1)
    teta_div_3: Tensor = teta[:, 0] / 3  # (batch_size, 1)
    real_part_if_not_real: Tensor = r_pow_1_div3 * torch.cos(teta_div_3)  # (batch_size, 1)
    imag_part_if_not_real: Tensor = r_pow_1_div3 * torch.sin(teta_div_3)  # (batch_size, 1)
    result_if_not_real: Tensor = torch.stack(
        (real_part_if_not_real, imag_part_if_not_real), dim=-1
    )  # (batch_size, 2)
    is_real: Tensor = (a[:, 1] == 0).unsqueeze(-1)  # (batch_size,1)

    result = torch.where(is_real, result_if_real, result_if_not_real)
    return result  # (batch_size, 2)


def sqrt_complex_batch(a: Tensor) -> Tensor:
    """
    Computes the square root of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the square root of the
        batch of complex numbers, where the first element is the real part and
        the second element is the imaginary part.
    """
    real: Tensor = a[:, :1]  # (batch_size, 1)
    imag: Tensor = a[:, 1:]  # (batch_size, 1)

    r: Tensor = torch.sqrt(real**2 + imag**2)  # (batch_size, 1)             # Magnitude
    theta: Tensor = torch.atan2(imag, real)  # (batch_size, 1)             # Argument
    sqrt_r: Tensor = torch.sqrt(r)  # (batch_size, 1)                     # sqrt(magnitude)
    half_theta: Tensor = theta / 2  # (batch_size, 1)

    sqrt_real: Tensor = sqrt_r * torch.cos(half_theta)  # (batch_size, 1)
    sqrt_imag: Tensor = sqrt_r * torch.sin(half_theta)  # (batch_size, 1)
    return torch.cat([sqrt_real, sqrt_imag], dim=1)  # (batch_size, 2)


def division_2_complex_numbers(a: Tensor, b: Tensor) -> Tensor:
    """
    Divides two batches of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the first
        batch of complex numbers.
        b : A tensor of shape (batch_size, 2) representing the second
        batch of complex numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the division of the two
        batches of complex numbers, where the first element is the real part and
        the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])
    inv_b: Tensor = inverse_complex_number(b)  # (batch_size, 2)
    return product_of_2_complex_numbers_batch(a, inv_b)  # (batch_size, 2)


def addition_complex_real_batch(a: Tensor, b: Tensor) -> Tensor:
    """
    Adds a batch of complex numbers to a batch of real numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part. Each real number is represented as a tensor of shape
    (batch_size, 1).
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
        b : A tensor of shape (batch_size, 1) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the sum of the batch
        of complex numbers and the batch of real numbers, where the first element
        is the real part and the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    #  b is a real number (batch_size,1)

    return torch.stack([a[:, 0] + b[:, 0], a[:, 1]], dim=-1)  # (batch_size, 2)
