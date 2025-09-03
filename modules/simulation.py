import torch
import random
import modules.utils as utils


def alpha(i: int, R: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Computes the alpha function for the SIR model.

    Args:
        i: The time step.
        R: The basic reproductive number.
        tau: The mean infectious period.

    Returns:
        The alpha value for the SIR model.
    """
    return 1 - torch.exp(-(R / tau) * i)


def act(
    state: torch.Tensor, R0: torch.Tensor, tau: torch.Tensor, Amat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the change in the SIR state variables.

    Args:
        state: The current SIR state vector.
        R0: The basic reproductive number.
        tau: The mean infectious period.
        Amat: The contact matrix.

    Returns:
        The new SIR state vector and the change in the number of susceptible individuals.
    """
    deltaSIR = torch.zeros_like(state)
    deltaSIR[0] = -state[0]*(Amat@ alpha(state[1], R0, tau))# here to check Amat cell importance
    AmatTrans= Amat* alpha(state[1], R0, tau)*state[0]
    
    deltaSIR[0][-deltaSIR[0]>state[0]]=-state[0][-deltaSIR[0]>state[0]]
    deltaSIR[2] = state[1] / tau
    deltaSIR[1] = -deltaSIR[0] - deltaSIR[2]
    return state + deltaSIR, -deltaSIR[0], AmatTrans


def one_strain(
    R0: torch.Tensor,
    tau0: torch.Tensor,
    maxtimeHorizon: int,
    n: int,
    Amat: torch.Tensor,
    startTime: int = 0,
    fromS: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Simulates the SIR model for a single strain.

    Args:
        R0: The basic reproductive number.
        tau0: The mean infectious period.
        maxtimeHorizon: The max number of time steps to simulate.
        n: The population size.
        Amat: The contact matrix.
        time: The time step to start the simulation at.
        fromS: The index of the individual to start the infection from.
        device: The device to run the simulation on.

    Returns:
        A tensor of the change in the number of susceptible individuals for each time step.
    """
    deltaSs = [torch.zeros(n, dtype=torch.float32, device=device)]
    stateNowS = torch.ones(n, dtype=torch.float32, device=device)
    stateNowI = torch.zeros(n, dtype=torch.float32, device=device)
    stateNowR = torch.zeros(n, dtype=torch.float32, device=device)
    stateNow = torch.stack([stateNowS, stateNowI, stateNowR])
    AmatTransSum= torch.zeros((n, n), dtype=torch.float32, device=device)
    # noise = torch.randn((timeHorizon + 1), dtype=torch.float32, device=device) / 400
    for i in range(maxtimeHorizon):
        if i == startTime:
            stateNow[0, fromS] = 0.99
            stateNow[1, fromS] = 0.01
            deltaSs[-1][fromS] = 0.01
        stateNow, deltaS, AmatTrans= act(stateNow, R0, tau0, Amat)
        AmatTransSum+= AmatTrans
        deltaSMin= torch.min(deltaS)
        deltaSs.append(deltaS.clone())
    deltaSs = torch.stack(deltaSs)  # + noise[:, None]
    return deltaSs.T, AmatTransSum


def multi_strains(
    G, paras: object, Amat: torch.tensor, intense: int =0, lower=20, device: str = "cpu") -> torch.Tensor:
    """Simulates the SIR model for multiple strains.

    Args:
        G: ntwworkX object
        paras: A parameter object containing the number of strains, the population size, and the minimum time horizon.
        Amat: The contact matrix.
        intense: low, middle or high degree nodes
        device: The device to run the simulation on.

    Returns:
        A tensor of the change in the number of susceptible individuals for each strain and time step.
    """
    maxtimeHorizon= 99
    R0s= paras.R0s
    taus= paras.taus
    if intense==-1:
        randomList= utils.select_nodes_linear_degree(G, paras.strains, device= device)
        pass
    else:
        randomList= utils.select_nodes_accroding_to_degree(G, paras.strains, intense)
    for i in range(paras.n):
        Amat[i, i]= 1
    deltaSsList= []
    for i in range(paras.strains):
        deltaS, _= one_strain(R0s[i], taus[i], maxtimeHorizon, paras.n, Amat, startTime= 0, fromS= randomList[i], device= device)
        deltaSsList.append(deltaS)
    deltaSsTensor= torch.stack(deltaSsList[0:paras.strains], dim= -1)

    for i in range(deltaSsTensor.shape[1]):
        tempSlice= deltaSsTensor[:, i, :]
        maxTemp= torch.max(tempSlice)
        if i>lower and maxTemp<1e-2:
            break
    return deltaSsTensor[:, 0:i, :], randomList
