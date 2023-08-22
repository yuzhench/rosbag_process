import numpy as np
import glob

input_file_direction = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out"

file_list = glob.glob(input_file_direction+'/*_semantic.npy')

for file in file_list:
    arr_int64 = np.load(file)
    arr_uint8 = arr_int64.astype(np.uint8)
    np.save(file,arr_uint8)