import random
import logging
from datetime import datetime
import torch
import os
import shutil


def select_nodes_accroding_to_degree(G, strains, intense= 0):
    #G:
    #networkx grph object
    #strains:
    #select how much nodes
    #intense:
    #0: random select from these low degree nodes
    #1: random select from these mid degree nodes
    #2: random select from these high degree nodes
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x))
    #print(sorted_nodes)
    #sorted_degree = [G.degree(i) for i in sorted_nodes]
    n= len(sorted_nodes)
    randomList= [random.randint(0, int(n/3))+int(n/3)*intense for _ in range(strains)]
    return [sorted_nodes[i] for i in randomList]

def select_nodes_linear_degree(G, strains, device= "cpu"):
    #G:
    #networkx grph object
    #strains:
    #select how much nodes
    
    # Calculate the degrees of each node
    degrees = dict(G.degree())

    # Extract nodes and their corresponding degrees as lists
    nodes, node_degrees = zip(*degrees.items())

    # Use random.choices to sample 3 nodes based on their degrees
    sampled_nodes = random.choices(nodes, weights=node_degrees, k=40)

    return sampled_nodes[0: strains]



def get_time_string():
    # Get the current date and time
    current_time = datetime.now()

    # Format the time string
    time_string = current_time.strftime("%m_%d_%H_%M")

    return time_string

def log_print(flag, *args):
    if flag == 1:
        # If flag is 1, print to the screen
        print(*args)
    else:
        # If flag is not 1, log to a file
        logging.info(' '.join(map(str, args)))

def continious_to_sparcity(my_tensor, top= 400):
    # Flatten the array to a 1D array
    flat_tensor = my_tensor.flatten()

    # Get the indices of the top 400 elements
    top_indices = torch.topk(flat_tensor, k=int(top.item())).indices

    # Create a new tensor with zeros
    output_tensor = torch.zeros_like(flat_tensor)

    # Set the top 400 elements to 1
    output_tensor[top_indices] = 1

    # Reshape the tensor back to its original shape
    output_tensor = output_tensor.view(my_tensor.shape)

    return output_tensor

def continious_to_sparcity_probability(my_tensor, sumWeight= 4, prob= 0.95):

    recallWeight= sumWeight*prob

    # Flatten the array to a 1D array
    flat_tensor = my_tensor.flatten()

    # Create a new tensor with zeros
    output_tensor = torch.zeros([my_tensor.shape[0]**2], device= my_tensor.device)

    # Get the indices of the top 400 elements
    sorted_indices = torch.argsort(flat_tensor, descending= True)
    accCount= 0
    for i in sorted_indices:
        output_tensor[i]= 1
        accCount+= flat_tensor[i]
        if accCount>recallWeight:
            print(accCount)
            break

    return output_tensor.view([my_tensor.shape[0], my_tensor.shape[0]])

def move_file(source_path, destination_folder):
    try:
        # Check if the source file exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"The file {source_path} does not exist.")

        # Check if the destination folder exists, create if not
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Extract the file name from the source path
        file_name = os.path.basename(source_path)

        # Construct the destination path
        destination_path = os.path.join(destination_folder, file_name)

        # Move the file
        shutil.move(source_path, destination_path)
        print(destination_path)

        print(f"File '{file_name}' successfully moved to '{destination_folder}'.")
    except Exception as e:
        print(f"Error: {e}")

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"File renamed successfully from {old_name} to {new_name}.")
    except FileNotFoundError:
        print(f"Error: File {old_name} not found.")
    except Exception as e:
        print(f"An error occurred: {e}, so move to archive")
        move_file(old_name, "archive")

def rename_files_replace_string(root_folder, old_string, new_string):
    print(root_folder)
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if old_string in filename:
                old_path = os.path.join(foldername, filename)
                new_filename = filename.replace(old_string, new_string)
                new_path = os.path.join(foldername, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} to {new_path}')

        


def empty_folder(folder_path):
    # Delete all remaining files (including sub-folder files)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

def reset_folder(folder_path, archive_folder):
    # Create the archive folder if it doesn't exist
    os.makedirs(archive_folder, exist_ok=True)

    # Move all files to the archive folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            shutil.move(file_path, os.path.join(archive_folder, file))
    empty_folder(folder_path)

