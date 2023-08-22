import os
import numpy as np
import cv2

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

# Specify the path to your .npy file
npy_file_path = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out/0_semantic.npy"

# npy_file_path = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/red/0_semantic.npy"
# npy_file_path = "/home/yuzhen/SegFormer-tensorrt/red/depth/0_depth.npy"

# npy_file_path = "/home/yuzhen/SegFormer-tensorrt/For_KINFU_mapping_without_tape/depth/0_depth.npy"


# npy_file_path = "/home/yuzhen/github/Azure-Kinect-Samples/opencv-kinfu-samples/build/indoor2/0_semantic.npy"

# jpg_file_path = "/home/yuzhen/SegFormer-tensorrt/rosbag_segformer/grey_jpg/0_semantic.jpg"
# Get the size of the file
#file_size = get_file_size(npy_file_path)

#print(f"The size of the file is {file_size} bytes.")

data = np.load(npy_file_path)

# data = cv2.imread(jpg_file_path)

# Check the shape of the loaded data
print("Shape of the data:", data.shape)

max = 0
for row in data:
    for element in row:
        # print(element, end=" ")
        if element > max:
            max = element
        print(f"the type is {type(element)}")

print(f"the max is {max}")




# # Normalize pixel values if necessary
# # npy_data = (npy_data - npy_data.min()) / (npy_data.max() - npy_data.min()) * 255

# # Convert to image
# image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

# # Save the image as .jpg
# jpg_file_path = "/home/yuzhen/Desktop/pircute_folder/p2.jpg"
# cv2.imwrite(jpg_file_path, image)

