import onnxruntime
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

from PIL import Image

ignore_index = 255
dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
         24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
         50, 51, 52, 53, 56, ]
t = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
srgb_colormap = [
    t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) if i in dms46
]
srgb_colormap.append([0, 0, 0])  # color for ignored index
srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)


def process(img):
    img = cv2.resize(img, (512, 512),
                     interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = np.copy(img)
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    return image

def main():
 



    #if the imput source is the frames of image 
    npy_folder = '/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out'
    # npy_folder = '/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/red'

    # npy_folder = '/home/yuzhen/SegFormer-tensorrt/result/red_out'

    # Get a list of all JPG files in the folder
    npy_files = [file for file in os.listdir(npy_folder) if file.endswith('_semantic.npy')]

    i = 0

    # new_folder = "jpg_result"
    new_folder = "SamSegformerJpg"


    # output_path = '/home/yuzhen/SegFormer-tensorrt/result'
    
    new_output_path = os.path.join(npy_folder,new_folder)

    try:
        os.makedirs(new_output_path)
    except:
        pass


    #while video.isOpened():
    for file in npy_files:
        npy_path = os.path.join(npy_folder,file)
        npy_data = np.load(npy_path)
        # npy_data = np.load('/home/yuzhen/SegFormer-tensorrt/result/red_out/0_semantic.npy')
        jpg_data = srgb_colormap[npy_data]


        # for row in npy_data:
        #     for element in row:
        #         print(element, end=" ")

        # break
        output_file_name = f"{i}.jpg"
        output_direction = os.path.join(new_output_path,output_file_name)
        cv2.imwrite(output_direction,jpg_data)
        i+=1

    

if __name__ == '__main__':
    main()
