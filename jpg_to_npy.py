import os
import numpy as np
from PIL import Image
import cv2

# input_dir = '/home/yuzhen/SegFormer-tensorrt/rosbag_output_depth_image'  # Replace with the directory containing your JPG files
input_dir = '/home/yuzhen/SegFormer-tensorrt/rosbag_segformer'  # Replace with the directory containing your JPG files
output_dir = '/home/yuzhen/SegFormer-tensorrt/npy_from_rosbag_jpg'  # Replace with the directory where you want to save the NPY files


#--------------------------------------
# Input directory containing the JPEG files
# input_directory = '/home/yuzhen/SegFormer-tensorrt/rosbag_segformer'

# # Output directory to store the grayscale images
# output_directory = '/home/yuzhen/SegFormer-tensorrt/rosbag_segformer/grey_jpg'

# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)

# # Get the list of JPEG files in the input directory
# file_list = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]

# # Iterate over each file in the directory
# for file_name in file_list:
#     # Construct the full path to the input image file
#     input_path = os.path.join(input_directory, file_name)

#     # Read the image
#     image = cv2.imread(input_path)

#     if image is not None:
#         # Convert the image to grayscale
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         print(f"the shape of the image is: {gray_image.shape}")
#         # Construct the full path to the output image file
#         output_path = os.path.join(output_directory, file_name)

#         # Save the grayscale image
#         cv2.imwrite(output_path, gray_image)
#     else:
#         print(f"Failed to read image: {input_path}")

#----------------------------------------------------------


def clean_directory(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)

# Clean the directory
clean_directory(output_dir)


count = 0

# Iterate over JPG files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.jpg'):
        # # Load the JPG image
        image_path = os.path.join(input_dir, file_name)
        # image = Image.open(image_path)


        image_color = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # Convert the image to a NumPy array
        image_array = np.array(gray_image)

        # # Reshape the array to the desired shape
        # desired_shape = (720, 1280)
        # reshaped_array = np.reshape(image_array, desired_shape)

        # # Save the NumPy array to an NPY file
        # output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '_segformer.npy')
        # np.save(output_file, image_array)

         # Save the NumPy array to an NPY file
        #output_file = os.path.join(output_dir,f'{count}_depth.npy')

        output_file = os.path.join(output_dir,f'{count}_semantic.npy')
        np.save(output_file, image_array)
        count+=1
