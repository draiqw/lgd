"""
Generate fixed initial population for test function optimization.

This ensures all algorithms (GA, ES, PABBO) start from the same point.
"""

import json
import random
import numpy as np


def generate_test_init(
    T_range=(1, 1000),
    pop_size=20,
    seed=42
):
    """
    Generate initial population using Latin Hypercube Sampling.

    Args:
        T_range: Range for T parameter (mapped to x via test function)
        pop_size: Population size
        seed: Random seed for reproducibility

    Returns:
        List of T values (integers)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Latin Hypercube Sampling
    population = []
    for i in range(pop_size):
        bin_size = (T_range[1] - T_range[0]) / pop_size
        T = int(T_range[0] + (i + random.random()) * bin_size)
        T = max(T_range[0], min(T_range[1], T))
        population.append(T)

    random.shuffle(population)

    return population


if __name__ == "__main__":
    # Generate and save
    init_pop = generate_test_init(
        T_range=(1, 1000),
        pop_size=20,
        seed=42
    )

    config = {
        "T_range": [1, 1000],
        "pop_size": 20,
        "seed": 42,
        "initial_population": init_pop
    }

    with open("test_init_population.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated initial population: {init_pop}")
    print(f"  min={min(init_pop)}, max={max(init_pop)}, mean={np.mean(init_pop):.1f}")
    print(f"Saved to test_init_population.json")