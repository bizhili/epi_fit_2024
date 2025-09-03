import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import scipy.optimize


def continious_to_sparcity(my_array, top= 400):
    # Flatten the array to a 1D array
    flat_array = my_array.flatten()

    # Sort the flattened array in descending order
    sorted_indices = np.argsort(flat_array)[::-1]

    # Set the top 400 elements to 1 and the rest to 0
    flat_array[sorted_indices[:top]] = 1
    flat_array[sorted_indices[top:]] = 0

    # Reshape the modified 1D array back to the original shape
    result_array = flat_array.reshape(my_array.shape)
    return result_array

def square_error(Z, preZ):
    return torch.sqrt((Z-preZ)**2).sum()

def cosine_similarity(Z, preZ):#V
    numerator= torch.sum(Z*preZ)
    denominator= torch.sqrt(torch.sum(Z**2))*torch.sqrt(torch.sum(preZ**2))
    return numerator/denominator

def pearson_correlation(matrix1, matrix2):#V
    # Flatten matrices to 1D arrays
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()
    
    # Calculate mean of each matrix
    mean_matrix1 = torch.mean(flat_matrix1)
    mean_matrix2 = torch.mean(flat_matrix2)
    
    # Calculate Pearson correlation coefficient
    numerator = torch.sum((flat_matrix1 - mean_matrix1) * (flat_matrix2 - mean_matrix2))
    denominator = torch.sqrt(torch.sum((flat_matrix1 - mean_matrix1)**2) * torch.sum((flat_matrix2 - mean_matrix2)**2))
    pearson_corr = numerator / denominator
    return pearson_corr

def jaccard_similarity(matrix1, matrix2):#V
    # Flatten matrices to 1D tensors
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()
    
    # Calculate intersection and union
    intersection = torch.sum(torch.minimum(flat_matrix1, flat_matrix2))
    union = torch.sum(torch.maximum(flat_matrix1, flat_matrix2))
    # Calculate Jaccard similarity index
    jaccard_similarity = intersection / union
    return jaccard_similarity  # Convert to Python float



def spectral_similarity(Z, preZ):#V
    try:
        preEig, preEigV = torch.linalg.eig(preZ)
        Eig, eigV = torch.linalg.eig(Z)
        return cosine_similarity(preEig.real, Eig.real)
    except:
        return 0
    
def ROC_AUC(Z, preZ):
    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    A= torch.tensor(continious_to_sparcity(Z.cpu().detach().numpy(), links))+IMatrix
    y_true = A.cpu().numpy().flatten()
    y_pred = preZ.cpu().numpy().flatten()
    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    return roc_auc

def PR_AUC(Z, preZ):
    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    A= torch.tensor(continious_to_sparcity(Z.cpu().detach().numpy(), links))+IMatrix
    y_true = A.cpu().numpy().flatten()
    y_pred = preZ.cpu().numpy().flatten()
    # Calculate false positive rate, true positive rate, and thresholds
    precision, recall, _  = precision_recall_curve(y_true, y_pred, pos_label= 1)
    # Calculate AUC
    roc_auc = auc(recall, precision)
    return roc_auc

def recall(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix
    numerator= torch.sum(Z*preZ)
    denominator= torch.sum(torch.abs(Z.float()))
    return numerator/denominator

def precision(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum(Z*preZ)
    denominator= torch.sum(torch.abs(preZ.float()))
    return numerator/denominator

def selectivity(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum((1-Z)*(1-preZ))
    denominator= torch.sum(torch.abs((1-Z).float()))
    return numerator/denominator

def negative_predictive(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix


    numerator= torch.sum((1-Z)*(1-preZ))
    denominator= torch.sum(torch.abs((1-preZ).float()))
    return numerator/denominator

def accuracy(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum(1-torch.logical_xor(Z, preZ).float())
    denominator= Z.shape[0]*Z.shape[0]
    return numerator/denominator

def jaccard_index(Z, preZ):

    links= int(torch.sum(Z>5e-3))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.cpu().detach().numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.cpu().detach().numpy(), links))+IMatrix
    
    numerator= torch.sum(Z*preZ)
    denominator= torch.sum((Z+preZ)-Z*preZ)
    return numerator/denominator

def normalized_hamming_distance(Z, preZ):
    """Calculates the Normalized Hamming Distance between two matrices."""

    assert Z.shape == preZ.shape, "Matrices must have the same shape."
    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix
    num_nodes = Z.shape[0]
    max_distance = num_nodes * (num_nodes - 1)   # Maximum possible distance for a full matrix

    hamming_distance = torch.sum(torch.logical_xor(Z, preZ)) #01, 10, (11, 00)

    normalized_distance = torch.sqrt(hamming_distance /max_distance)

    return normalized_distance

def graph_avg_edit_distance(matrix1, matrix2):
    # Convert matrices to numpy arrays
    n=matrix1.shape[0]
    matrix1_np = matrix1.numpy()
    matrix2_np = matrix2.numpy()
    
    # Calculate the cost matrix based on absolute differences in weights
    cost_matrix = np.abs(matrix1_np[:, np.newaxis] - matrix2_np).sum(axis=2)  # Sum along the third axis
    
    # Solve assignment problem using the Hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    # Calculate graph edit distance
    ged = cost_matrix[row_ind, col_ind].sum()
    
    return ged

def draw_auc_roc(As, preAs,  legends= [] , random_real_pre= None):
    plt.figure(figsize=(8, 6))
    fontSize= 27
    for i, A in enumerate(As):
        # Flatten matrices into 1D arrays
        y_true = A.numpy().flatten()
        y_pred = preAs[i].numpy().flatten()

        # Calculate false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{legends[i]}')
    if random_real_pre is not None:
        print("haha")
        random_true= random_real_pre[0].flatten()
        random_pred= random_real_pre[1].numpy().flatten()
        # Calculate false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(random_true, random_pred, pos_label=1)
        # Plot ROC curve
        plt.plot(fpr, tpr,  color='navy', linestyle='--', label= "base")

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize= fontSize)
    plt.ylabel('True Positive Rate', fontsize= fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    #plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize= fontSize-5)
    plt.show()

def draw_prc(As, preAs, legends= [], random_real_pre= None):
    plt.figure(figsize=(8, 6))
    fontSize= 27
    for i, A in enumerate(As):
        # Flatten matrices into 1D arrays
        y_true = A.numpy().flatten()
        y_pred = preAs[i].numpy().flatten()

        # Calculate false positive rate, true positive rate, and thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # Calculate AUC
        #roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(recall, precision, label=f'{legends[i]}')
    if random_real_pre is not None:
        print("haha")
        random_true= random_real_pre[0].flatten()
        random_pred= random_real_pre[1].numpy().flatten()
        # Calculate false positive rate, true positive rate, and thresholds
        precision, recall, thresholds = precision_recall_curve(random_true, random_pred)
        # Plot ROC curve
        plt.plot(recall, precision, color='navy', linestyle='--', label= "Random pred")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize= fontSize)
    plt.ylabel('Precision', fontsize= fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    #plt.title('Precision-Recall')
    # plt.legend(loc="lower right", fontsize= fontSize-5)
    plt.show()