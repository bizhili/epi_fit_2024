import argparse
import warnings
import random


class Paras:
    """A class to store the parameters for the simulation.

    Attributes:
        randomGraph (str): The type of random graph to use.
        seed (int): The random seed to use.
        n (int): The number of nodes in the graph.
        strains (int): The number of strains in the simulation.
        epoches (int): The number of epoches to run the simulation for.
        plot (bool): How to plot the graph.
        device (str): The device to use for the simulation.
        weightModel (str): The type of adjacency weight model to use.
        intense(int): intense of select nodes degree
        R0Mean(float): mean of R0s
        R0Std(float): std of R0s
        tauMean(float): mean of taus
        tauStd(float): std of taus

    """

    def __init__(self):
        self.randomGraph = None
        self.seed = None # seed used for generating topology and data
        self.n = None
        self.strains = None
        self.epoches = None
        self.plot = None
        self.device = None
        self.weightModel = None 
        self.intense= None
        self.R0Mean= None
        self.R0Std= None
        self.tauMean= None
        self.tauStd= None
        self.taus= None
        self.R0s= None
        self.modelLoad= None
        self.epi= None
        self.dense= None
        self.identicalf= None
        self.wsProbability= None
        self.evaluateEvery= None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    """

    parser.add_argument(
        '--randomGraph',
        type=str,
        default="BA",
        help='Choosing random graph model(str): RGG(defult), ER, WS, BA'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10,
        help='Setting random seed(int): 10(defult)<used for generating topology and data>'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=50,
        help='Setting nodes number(int): 50(defult)'
    )
    parser.add_argument(
        '--strains',
        type=int,
        default=1,
        help='Setting strains number(int): 1(defult)~4'
    )
    parser.add_argument(
        '--epoches',
        type=int,
        default=100000,
        help='Setting stop epoches number(int): 100000(defult)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help='Setting device(str): cuda:0(defult), cpu'
    )
    parser.add_argument(
        '--weightModel',
        type=str,
        default="identical",
        help='Setting adjacency weight model(str): identical(defult), gravity, degree'
    )
    parser.add_argument(
        '--intense',
        type=int,
        default=0,
        help='Setting the intense of selecting nodes degree from low to high(int): 0(defult), 1, 2, -1(linear intense)'
    )
    parser.add_argument(
        '--R0Mean',
        type=float,
        default= 8.3,
        help='Setting the mean value of R0s, average distribution (float): 8.3(defult)'
    )
    parser.add_argument(
        '--R0Std',
        type=float,
        default= 0.5,
        help='Setting the Std value of R0s, average distribution (float): 0.5(defult)'
    )
    parser.add_argument(
        '--tauMean',
        type=float,
        default=7.5,
        help='Setting the mean value of R0s, average distribution (float): 6.2(defult)'
    )
    parser.add_argument(
        '--tauStd',
        type=float,
        default= 0.1,
        help='Setting the Std value of R0s, average distribution (float): 0.1(defult)'
    )
    parser.add_argument(
        '--modelLoad',
        type=str,
        default= "AA",
        help='Setting load model (string): AA(defult), AB, BA, BB, infer2018, ATA'
    )
    parser.add_argument(
        '--dense',
        type=int,
        default= 8,
        help='Setting avg degree of BA, WS, ER, RGG (int), if negative avg degree log(n)-dense: 8(defult)'
    )
    parser.add_argument(
        '--identicalf',
        type=float,
        default= 0.01,
        help='Setting identical float (float): 0.01(defult)'
    )
    parser.add_argument(
        '--wsProbability',
        type=float,
        default= 0.1,
        help='Setting WS model rewiring probability (float): 0.1(defult)'
    )
    parser.add_argument(
        '--evaluateEvery',
        type=int,
        default= 100,
        help='How many epoches to perform evaluate once'
    )
    parser.add_argument(
        '--epi',
        type=str,
        default= "H1N1",
        help='epidemic empirical data read'
    )


def read_arguments(parser: argparse.ArgumentParser) -> Paras:
    """Reads the arguments from the parser and returns a Paras object.

    Args:
        parser (argparse.ArgumentParser): The parser to read the arguments from.

    Returns:
        Paras: A Paras object containing the parsed arguments.
    """

    args, unknown = parser.parse_known_args()
    paras = Paras()

    paras.randomGraph = args.randomGraph
    paras.seed = args.seed
    paras.n = args.n
    paras.strains = args.strains
    paras.plot = "2d_RGG" if args.randomGraph == "RGG" else "spring"
    paras.device = args.device
    paras.weightModel = args.weightModel
    paras.intense = args.intense
    paras.R0Mean= args.R0Mean
    paras.R0Std= args.R0Std
    paras.tauMean= args.tauMean
    paras.tauStd= args.tauStd
    paras.modelLoad= args.modelLoad
    paras.epoches= args.epoches
    paras.dense= args.dense
    paras.identicalf= args.identicalf
    paras.wsProbability= args.wsProbability
    paras.evaluateEvery= args.evaluateEvery
    paras.epi= args.epi


    if paras.weightModel == "gravity" and paras.randomGraph != "RGG":
        paras.weightModel = "degree"
        warnings.warn(
            "Only RGG random graph can apply on gravity model, setting adjacency weight model to degree model!!!",
            RuntimeWarning,
        )

    return paras

