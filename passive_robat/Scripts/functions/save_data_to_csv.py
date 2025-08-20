import csv
import os

def save_data_to_csv(matrix, filename, path):
    """
    Save matrix as csv file
    Parameters:
    - matrix: matrix to save 
    - filename: name of the csv file
    - path: path to save the csv file
    """
    full_path = os.path.join(path, filename)
    with open(full_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)
    print(f"Matrix has been saved as csv to {full_path}")