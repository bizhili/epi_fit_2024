import torch


def create_A_mat(weightA: torch.Tensor, popuT: torch.Tensor) -> torch.Tensor:
  """Creates the adjacency matrix for the weighted network.

  Args:
    weightA: A tensor of shape (n, n), representing the weighted adjacency matrix.
    popuT: A tensor of shape (n,), representing the population of each node.

  Returns:
    A tensor of shape (n, n), representing the adjacency matrix for the weighted
    network.
  """

  omega = popuT * weightA
  dT = weightA.sum(dim=1)
  inpop = omega.sum(dim=1)
  AmatTemp = (dT[:, None] * omega)
  AmatTemp = AmatTemp / (inpop[:, None] + 1e-9) + torch.eye(weightA.shape[0], dtype=torch.float32, device=weightA.device)
  return AmatTemp


def reverse_A_mat(A: torch.Tensor, popuT: torch.Tensor) -> torch.Tensor:
  """Reverses the adjacency matrix for the weighted network.

  Args:
    A: A tensor of shape (n, n), representing the adjacency matrix for the
      weighted network.
    popuT: A tensor of shape (n,), representing the population of each node.

  Returns:
    A tensor of shape (n, n), representing the reversed adjacency matrix for the
    weighted network.
  """

  sumA = A.sum(dim=1)
  tempA = A / popuT
  sumTempA = tempA.sum(dim=1)
  weightA = sumA[:, None] * A / (sumTempA[:, None] * popuT)
  return weightA

