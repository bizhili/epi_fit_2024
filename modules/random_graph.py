import torch
import networkx as nx
import random
import numpy as np
import itertools
import math


# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_ER_random_contact(n: int, avgDegree: float, shuffle= False, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for an Erdős-Rényi random graph.

    Args:
        n: The number of nodes in the graph.
        avgDegree: The average degree of the graph.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
    """

    graph = nx.dense_gnm_random_graph(n, n * avgDegree)
    while nx.is_connected(graph) is False:
        graph = nx.dense_gnm_random_graph(n, n * avgDegree)
    if shuffle:
        # Shuffle the index of the graph
        node_list = list(graph.nodes())
        random.shuffle(node_list)
        graph = nx.relabel_nodes(graph, dict(zip(graph.nodes(), node_list)))
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph

def get_WS_random_contact(n: int, k: int, p: float, shuffle= False, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for a Watts-Strogatz small-world graph.

    Args:
        n: The number of nodes in the graph.
        k: The number of nearest neighbors to connect to.
        p: The probability of rewiring each edge.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
    """

    graph = nx.watts_strogatz_graph(n, k, p)
    while nx.is_connected(graph) is False:
        graph = nx.watts_strogatz_graph(n, k, p)
    if shuffle:
        # Shuffle the index of the graph
        node_list = list(graph.nodes())
        random.shuffle(node_list)
        graph = nx.relabel_nodes(graph, dict(zip(graph.nodes(), node_list)))
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph

def get_BA_random_contact(n: int, m: int, shuffle= False, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for a Barabási-Albert preferential attachment graph.

    Args:
        n: The number of nodes in the graph.
        m: The number of edges to attach from a new node to existing nodes.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
    """
    graph = nx.barabasi_albert_graph(n, m)
    while nx.is_connected(graph) is False:
        graph = nx.barabasi_albert_graph(n, m)
    if shuffle:
        # Shuffle the index of the graph
        node_list = list(graph.nodes())
        random.shuffle(node_list)
        graph = nx.relabel_nodes(graph, dict(zip(graph.nodes(), node_list)))
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph

def get_RGG_random_contact(n: int, m: int, shuffle= False, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for a random RGGmetric graph.

    Args:
        n: The number of nodes in the graph.
        radius: The radius of creating a link.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
        pos: the 2-d position of all nodes
    """
    
    linkNum= n*m
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n)}
    posValue= list(pos.values())
    node_pairs = list(itertools.combinations(range(len(posValue)), 2))
    distance_dict = {(i, j): calculate_distance(posValue[i], posValue[j]) for (i, j) in node_pairs}
    sorted_distance_dict = sorted(distance_dict.items(), key=lambda x: x[1])
    contactNp= np.zeros((len(posValue), len(posValue)))
    for i in range(len(sorted_distance_dict)):
        temp= sorted_distance_dict[i]
        nodei= temp[0][0]
        nodej= temp[0][1]
        contactNp[nodei, nodej]= 1
        contactNp[nodej, nodei]= 1
        if i>= linkNum-1:
            graph= nx.Graph(contactNp)
            if nx.is_connected(graph):
                break
    if shuffle:
        # Shuffle the index of the graph
        node_list = list(graph.nodes())
        random.shuffle(node_list)
        graph = nx.relabel_nodes(graph, dict(zip(graph.nodes(), node_list)))
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph, pos

def read_from_file(FileName, device: str="cpu"):
    Anp= np.load(FileName)
    contact = torch.FloatTensor(Anp).to(device)
    graph = nx.from_numpy_array(Anp)
    return contact, graph

