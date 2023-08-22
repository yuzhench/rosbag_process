import numpy as np

red_direction = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/red_out/0_depth.npy"

data = np.load(red_direction)

for row in data:
    for element in row:
        print(element, end=" ")
        