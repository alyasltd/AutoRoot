from autoroot.torch.complex.complex_utils import *
from autoroot.torch.cubic.cubic import polynomial_root_calculation_3rd_degree
from torch import Tensor


def polynomial_root_calculation_4th_degree_ferrari(
    a0: Tensor, a1: Tensor, a2: Tensor, a3: Tensor, a4: Tensor
) -> Tensor:  # Ferrari's Method
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
        Tensor: Roots of the polynomial, shape (batch_size, 4, 2)
        Each root is represented as a complex number (real, imaginary)
    """

    batch_size: int = a0.shape[0]  # Get the batch size from the shape of a0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reduce the quartic equation to the form : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    a: Tensor = a3 / a4  # (batch_size, 1)
    b: Tensor = a2 / a4
    c: Tensor = a1 / a4
    d: Tensor = a0 / a4

    # Computation of the coefficients of the Ferrari's Method
    S: Tensor = a / 4  # (batch_size, 1)
    b0: Tensor = d - c * S + b * S**2 - 3 * S**4  # (batch_size, 1)
    b1: Tensor = c - 2 * b * S + 8 * S**3  # (batch_size, 1)
    b2: Tensor = b - 6 * S**2  # (batch_size, 1)

    # Solve the cubic equation m^3 + b2*m^2 + (b2^2/4  - b0)*m - b1^2/8 = 0
    x_cube: Tensor = polynomial_root_calculation_3rd_degree(
        torch.tensor(1, dtype=torch.float32, device=device).repeat(b2.shape),
        b2,
        (b2**2) / 4 - b0,
        (-(b1**2)) / 8,
    )

    x_cube_real: Tensor = x_cube[:, :, 0]  #  (batch_size, 3)
    x_cube_imag: Tensor = x_cube[:, :, 1]  #    (batch_size, 3)

    is_real: Tensor = torch.isclose(
        x_cube_imag, torch.tensor(0.0, dtype=torch.float32, device=device), atol=1e-10, rtol=1e-5
    )
    is_positive: Tensor = x_cube[:, :, 0] > 0
    condition: Tensor = (
        is_real & is_positive
    )  # Condition to check if the root is real and positive   (batch_size, 3)

    real_filtered: Tensor = x_cube_real.clone()
    real_filtered[~condition] = float(
        "inf"
    )  # if root real and positive, keep it, else set to infinity   (batch_size, 3)

    alpha_0_real: Tensor
    alpha_0_real, _ = real_filtered.min(
        dim=1
    )  # Get the minimum real part of the roots (if doesn't exist, returns inf)
    alpha_0: Tensor = torch.stack(
        [alpha_0_real, torch.zeros(batch_size, dtype=torch.float32, device=device)], dim=-1
    )  # (batch_size,2)

    # do the calculation for alpha_0_nul and not alpha_0_nul and then affects the good value

    # if alpha_0_nul == False
    alpha0_div_2: Tensor = 0.5 * alpha_0  # beacause alpha_0 is real  # (batch_size, 2)
    sqrt_alpha: Tensor = sqrt_batch(
        alpha0_div_2[:, 0].unsqueeze(-1)
    )  # input : (batch_size, 1) // output : (batch_size, 2)
    term: Tensor = addition_complex_real_batch(-alpha0_div_2, -b2 / 2)  # (batch_size, 2)
    denom: Tensor = 2 * torch.sqrt(2 * alpha_0)  # beacause alpha_0 is real  # (batch_size, 2)
    num: Tensor = torch.stack(
        [b1, torch.zeros(batch_size, 1, dtype=torch.float32, device=device)], dim=-1
    ).squeeze(
        1
    )  # (batch_size, 2)

    frac: Tensor = division_2_complex_numbers(num, denom)  # (batch_size, 2)

    x1_false: Tensor = addition_complex_real_batch(sqrt_alpha, -S) + sqrt_complex_batch(
        addition_batch(term, -frac)
    )  # (batch_size, 2)
    x2_false: Tensor = addition_complex_real_batch(sqrt_alpha, -S) - sqrt_complex_batch(
        addition_batch(term, -frac)
    )  # (batch_size, 2)
    x3_false: Tensor = addition_complex_real_batch(-sqrt_alpha, -S) + sqrt_complex_batch(
        addition_batch(term, frac)
    )  # (batch_size, 2)
    x4_false: Tensor = addition_complex_real_batch(-sqrt_alpha, -S) - sqrt_complex_batch(
        addition_batch(term, frac)
    )  # (batch_size, 2)

    # if alpha_0_nul == True
    sqrt_inner1: Tensor = sqrt_batch((b2**2) / 4 - b0)  # complex

    x1_true: Tensor = addition_complex_real_batch(
        sqrt_complex_batch(addition_complex_real_batch(sqrt_inner1, -b2 / 2)), -S
    )  # (batch_size, 2)
    x2_true: Tensor = addition_complex_real_batch(
        -sqrt_complex_batch(addition_complex_real_batch(sqrt_inner1, -b2 / 2)), -S
    )  # (batch_size, 2)
    x3_true: Tensor = addition_complex_real_batch(
        sqrt_complex_batch(addition_complex_real_batch(-sqrt_inner1, -b2 / 2)), -S
    )  # (batch_size, 2)
    x4_true: Tensor = addition_complex_real_batch(
        -sqrt_complex_batch(addition_complex_real_batch(-sqrt_inner1, -b2 / 2)), -S
    )  # (batch_size, 2)

    real_positive_root_exist: Tensor = (
        (alpha_0_real == float("inf")).unsqueeze(-1).unsqueeze(-1)
    )  # (batch_size, 1)

    result: Tensor = torch.where(
        real_positive_root_exist,
        torch.stack([x1_true, x2_true, x3_true, x4_true], dim=1),
        torch.stack([x1_false, x2_false, x3_false, x4_false], dim=1),
    )

    return result  # (batch_size,4, 2)
