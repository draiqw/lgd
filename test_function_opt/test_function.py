"""
Complex non-differentiable test function for optimization benchmarking.

This function has the following properties:
- Non-differentiable (uses absolute values)
- Discontinuous (step function)
- Multiple local minima (>5)
- One global minimum around x ≈ 0
"""

import numpy as np


def test_function(x: float) -> float:
    """
    Complex non-differentiable test function.

    Properties:
    - Domain: typically tested on [-5, 5]
    - Non-differentiable at multiple points (absolute values)
    - Discontinuous (step function regions)
    - Multiple local minima
    - Global minimum around x ≈ 0

    Args:
        x: Input value (float)

    Returns:
        Function value y(x)
    """
    # Main oscillating components (non-differentiable)
    y = np.abs(np.sin(2 * np.pi * x))  # Major oscillations, period = 0.5
    y += 0.5 * np.abs(np.cos(5 * np.pi * x))  # Medium oscillations, period = 0.4
    y += 0.2 * np.abs(np.sin(10 * np.pi * x))  # Fine oscillations, period = 0.2

    # Quadratic bias toward x=0
    y += 0.3 * x**2

    # Non-differentiable at x=0
    y += 2 * np.abs(x)

    # Step function (discontinuous) - creates distinct regions
    if x < -3:
        y += 5
    elif x < -1:
        y += 3
    elif x < 1:
        y += 0  # Best region (minimum here)
    elif x < 3:
        y += 2
    else:
        y += 4

    # Additional high-frequency noise
    y += 0.1 * np.abs(np.sin(20 * np.pi * x))

    return float(y)


def test_function_vectorized(x: np.ndarray) -> np.ndarray:
    """
    Vectorized version of test_function for plotting.

    Args:
        x: Array of input values

    Returns:
        Array of function values
    """
    return np.array([test_function(xi) for xi in x])


def get_approximate_minimum():
    """
    Get approximate global minimum by grid search.

    Returns:
        Tuple of (x_min, y_min)
    """
    x_grid = np.linspace(-5, 5, 10000)
    y_grid = test_function_vectorized(x_grid)
    idx_min = np.argmin(y_grid)
    return x_grid[idx_min], y_grid[idx_min]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Plot the function
    x = np.linspace(-5, 5, 1000)
    y = test_function_vectorized(x)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, linewidth=2, label='Test Function')

    # Mark approximate minimum
    x_min, y_min = get_approximate_minimum()
    plt.plot(x_min, y_min, 'ro', markersize=10, label=f'Approx. minimum: ({x_min:.3f}, {y_min:.3f})')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Complex Non-differentiable Test Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_function.png', dpi=150)
    plt.show()

    print(f"Approximate global minimum: x = {x_min:.6f}, y = {y_min:.6f}")