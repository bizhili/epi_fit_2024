import random
import torch

def population(
    n: int,
    mu: float = 320,
    sigma: float = 40,
    base: int = 100,
    device: str = "cpu",
    ) -> torch.Tensor:
    """Generate a population of n individuals with a Gaussian distribution.

    Args:
        n: The number of individuals in the population.
        mu: The mean of the Gaussian distribution.
        sigma: The standard deviation of the Gaussian distribution.
        base: The base value to multiply the generated numbers by.
        device: The device to place the population tensor on.

    Returns:
        A torch.Tensor containing the population.
    """

    temp = [base * int(random.gauss(mu, sigma)) for i in range(n)]
    popuTen = torch.tensor(temp, dtype=torch.float32, device=device)
    return popuTen

