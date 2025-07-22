<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/banner.png">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/banner.png">
                <img alt="Library Banner" src="./docs/assets/logo.svg" width="500" height="300">
        </picture>
</div>
<br>

<div align="center">
  <a href="#">
        <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-efefef">
    </a>
    <a href="https://github.com/Pruneeuh/AutoRoot/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/Pruneeuh/AutoRoot/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/Pruneeuh/AutoRoot/actions/workflows/python-linter.yml">
        <img alt="Lint" src="https://github.com/Pruneeuh/AutoRoot/actions/workflows/python-linter.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/autoroot_torch">
        <img alt="Pepy" src="https://static.pepy.tech/badge/autoroot_torch">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://Pruneeuh.github.io/AutoRoot/"><strong>Explore AutoRoot docs »</strong></a>
</div>
<br>

# 👋 AutoRoot
***AutoRoot*** : Differentiable Root Solvers for Cubic and Quartic Polynomials
AutoRoot is a fast, fully differentiable PyTorch library for solving cubic and quartic equations


## 📚 Table of Contents
- [Motivation](#motivation)
- [Usage](#usage)
- [Tutorials](#-tutorials)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Tests](#tests)
- [Installation](#-installation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

# Motivation
AutoRoot serves as a standalone and versatile solution for anyone needing to solve polynomial equations of degrees 3 to 4 with *PyTorch*. This library leverages classical algebraic methods, specifically **Cardano's method** for cubic equations and **Ferrari's method** for quartic equations, to provide accurate and fully differentiable root solutions. The differentiability allows for seamless integration into PyTorch's autograd system, making it ideal for applications requiring gradient-based optimization and neural network integration.

# Usage
The main functions are located in the `autoroot.torch.cubic` and `autoroot.torch.quartic` modules.

- **Input:** Each coefficient (a, b, ..., e) is a tensor of shape `(batch_size, 1)`.
- **Output:** A tensor of shape `(batch_size, N, 2)` where `N = 3` for cubic or `4` for quartic. Each root is represented as a pair `[real, imaginary]`.

Please note that these methods are designed for polynomials with real coefficients. Complex numbers are represented as `torch.Tensor` of shape `(..., 2)`, where the first value is the real part and the second is the imaginary part. Utilities for these operations are available in `autoroot.torch.complex.complex_utils.py`.

### Example: Solving a Cubic Equation

```python
import torch
from autoroot.torch.cubic import solve_cubic

# Example: x^3 - 6x^2 + 11x - 6 = 0 (roots are 1, 2, 3)
# Coefficients are ordered as [a, b, c, d] for ax^3 + bx^2 + cx + d = 0
a = torch.tensor([[1.0]])
b = torch.tensor([[-6.0]])
c = torch.tensor([[11.0]])
d = torch.tensor([[-6.0]])

roots = solve_cubic(a, b, c, d)
print("Cubic roots:", roots)
# Expected output (order might vary, imaginary parts should be close to zero):
# Cubic roots: tensor([[[1.0000e+00, 0.0000e+00],
#                        [3.0000e+00, 0.0000e+00],
#                        [2.0000e+00, 0.0000e+00]]])

# Example: With a batch of inputs
A = torch.tensor([[1.0], [1.0]])
B = torch.tensor([[-6.0], [0.0]])
C = torch.tensor([[11.0], [0.0]])
D = torch.tensor([[-6.0], [-8.0]]) # x^3 - 8 = 0 (roots: 2, -1+sqrt(3)i, -1-sqrt(3)i)

roots_batch = solve_cubic(A, B, C, D)
print("Batch cubic roots:", roots_batch)
```

# 🔥 Tutorials

| **Tutorial Name**           | Notebook                                                                                                                                                           |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Find Roots of a Cubic polynom - ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pruneeuh/AutoRoot/blob/main/tutorials/Tutorial_1_cubic_solving_with_Cardano.ipynb)            |
| Find Roots of a Quartic polynom - ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pruneeuh/jacobinet/blob/main/tutorials/Tutorial_1_quartic_solving_with_Ferrari.ipynb)            |



# Project Structure
```
📂 autoroot/

├── 📁 torch/                   # Torch-related implementations
│   ├── 📁 complex/             # Complex functions utils
│   │   ├── 📄 complex_utils.py
│   ├── 📁 cubic/               # Cubic solver
│   │   ├── 📄 cubic.py
│   ├── 📁 quartic/             # Quartic solver
│   │   ├── 📄 quartic.py
│   ├── 📄 types.py

├── 📁 numpy/                   # Numpy-related implementations
│   ├── 📄 cubic.py             # Cubic solver
│   ├── 📄 quartic.py           # Quartic solver

📂 tests/                        # Unit tests
├── 📄 conftest.py
├── 📄 test_cubic.py
├── 📄 test_quartic.py
├── 📄 test_complex.py
├── 📄 test_polynomial_numpy.py


📂 tutorials/
├── 📄 Tutorial_1_cubic_solving_with_Cardano.ipynb

📄 .pre-commit-config.yaml       # Pre-commit hooks config
📄 pyproject.toml                # Build system and tool configs
📄 README.md                     # Project overview
📄 tox.ini                       # Tox testing configuration
```
# Dependencies
AutoRoot relies primarily on:
  - Python ≥ 3.8
  - PyTorch ≥ 1.10 (for tensor operations and autograd)
  - NumPy (for testing and reference comparisons)
  - Pytest (for running the test suite, only for contributors)

All dependencies are declared in the pyproject.toml.


# Tests
A comprehensive test suite is provided in the tests/ folder to ensure the reliability and robustness of the root calculations.
Tests are written using pytest and utilize numpy for result verification (especially for comparison with reference values or handling tolerances).


# 🚀 Installation
You can install AutoRoot using `pip`:
> [!IMPORTANT]
> 🚨 There is another package named `autoroot` on PyPI. Please ensure you install **`autoroot_torch`** to get this library.


```python
pip install autoroot_torch
```

Alternatively, to install from source:

```python
git clone https://github.com/your-repo/autoroot.git # Replace with your actual repo URL
cd autoroot
pip install .
```

# 👍 Contributing

#To contribute, you can open an
#[issue](https://github.com/Pruneeuh/AutoRoot/issues), or fork this
#repository and then submit changes through a
#[pull-request](https://github.com/Pruneeuh/AutoRoot/pulls).
We use [black](https://pypi.org/project/black/) to format the code and follow PEP-8 convention.
To check that your code will pass the lint-checks, you can run:

```python
tox -e py39-lint
```

You need [`tox`](https://tox.readthedocs.io/en/latest/) in order to
run this. You can install it via `pip`:

```python
pip install tox
```


# 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
  <picture>
    <img alt="ANITI Logo" src="https://aniti.univ-toulouse.fr/wp-content/uploads/2023/06/Capture-decran-2023-06-26-a-09.59.26-1.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French program within the <a href="https://aniti.univ-toulouse.fr/">Artificial and Natural Intelligence Toulouse Institute (ANITI)</a>. The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.


## 🗞️ Citation



## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
