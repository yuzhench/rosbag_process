import numpy as np

# File paths of the Numpy files
file1_path = "/home/yuzhen/Desktop/pircute_folder/0_semantic.npy"
file2_path = "/home/yuzhen/Desktop/pircute_folder/0_semantic_good.npy"

# Load the arrays from the Numpy files
array1 = np.load(file1_path)
array2 = np.load(file2_path)

# Compare the arrays for equality
if np.array_equal(array1, array2):
    print("The Numpy files are the same.")
else:
    print("The Numpy files are different.")
