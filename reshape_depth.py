import os
import numpy as np

# Input directory containing the Numpy files
input_directory = '/home/yuzhen/SegFormer-tensorrt/rosbag2/depth'

# Output directory to store the reshaped Numpy files
output_directory = '/home/yuzhen/SegFormer-tensorrt/rosbag2/rgb'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of Numpy files in the input directory
file_list = [f for f in os.listdir(input_directory) if f.endswith('.npy')]

# Iterate over each file in the directory
for file_name in file_list:
    # Construct the full path to the input Numpy file
    input_path = os.path.join(input_directory, file_name)

    # Load the Numpy file
    data = np.load(input_path)

    # Reshape the data
    reshaped_data = data.reshape((720, 1280))

    # Construct the full path to the output Numpy file
    output_path = os.path.join(output_directory, file_name)

    # Save the reshaped data as a Numpy file
    np.save(output_path, reshaped_data)