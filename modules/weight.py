import random
import numpy as np
import torch
import networkx as nx

def _gravity_model(pi: float, pj: float, dij: float, a1: float = 0.1,
                    a2: float = 0.008, r: float = 30) -> float:  # from i to j
    """Calculates the gravity model between two nodes.

    Args:
        pi: Population of node i.
        pj: Population of node j.
        dij: Distance between nodes i and j.
        a1: Hyperparameter 1.
        a2: Hyperparameter 2.
        r: Scaling factor.

    Returns:
        The gravity model weight between nodes i and j.
    """

    return np.exp(a1 * np.log(pi) + a2 * np.log(pj)) * np.exp(-dij * r)

def gravity_model(G: nx.Graph, pos: np.ndarray, P: np.ndarray, thre: float = 0.03,
                   device: str = "cpu") -> torch.Tensor:
    """Calculates the gravity model weights for a network.

    Args:
        G: NetworkX graph object.
        pos: Position of the nodes.
        P: Population of the nodes.
        thre: Threshold for the weight.
        device: Device to use for calculations.

    Returns:
        A torch tensor containing the gravity model weights for the network.
    """

    n = G.number_of_nodes()
    weightA = torch.zeros((n, n), dtype=torch.float32, device=device)
    count = 0
    for i, j in G.edges():
        count += 1
        dx = pos[i][0] - pos[j][0]
        dy = pos[i][1] - pos[j][1]
        dij = np.linalg.norm((dx, dy))
        pi = P[i].item()
        pj = P[j].item()
        wij = _gravity_model(pi, pj, dij)
        if wij > thre:
            wij = thre
        weightA[i, j] = wij
        wji = _gravity_model(pj, pi, dij)
        if wji > thre:
            wji = thre
        weightA[j, i] = wji
    return weightA

def degree_model(Gt: torch.Tensor, P: torch.Tensor, theta: float = 4, C: float = 2000,
                   device: str = "cpu") -> torch.Tensor:
    """Calculates the degree model weights for a network.

    Args:
        Gt: Network adjacency matrix, tensor.
        P: Population vector, tensor.
        theta and C are hyper parameters.
        device: Device to use for calculations.

    Returns:
        A torch tensor containing the degree model weights for the network.
    """

    d = Gt.sum(dim=1)
    n = Gt.shape[0]
    weight = torch.zeros_like(Gt, device=device)
    for i in range(n):
        for j in range(n):
            weight[i, j] = C * (Gt[i, j] * d[j]**theta) / (P[j] * torch.sum(Gt[i, :] * (d**theta)))
    return weight

def identical_model(Gt: torch.Tensor, value:float = 0.01,  device: str = "cpu") -> torch.Tensor:
    """Calculates the identical weights for a network.

    Args:
        Gt: Network adjacency matrix, tensor.
        value: ientical value
        device: Device to use for calculations.

    Returns:
        A torch tensor containing the identical weights for the network.
    """

    n = Gt.shape[0]
    weight = torch.zeros_like(Gt, device=device)
    for i in range(n):
        for j in range(n):
            weight[i, j] = value*Gt[i, j]
    return weight

