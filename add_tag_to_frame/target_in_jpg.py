import cv2 as cv
import numpy as np
from PIL import Image
import os

# img_dir = "/home/yuzhen/SegFormer-tensorrt/result_red/tag/0_tag.jpg"
img_dir = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/kinfu_tag/1.jpg"
img_data = cv.imread(img_dir)
cv.line(img_data,(0,500),(1280,500),color=(0,255,0),thickness=1)
cv.line(img_data,(600,0),(600,720),color=(0,255,0),thickness=1)
cv.imshow("img_data",img_data)
cv.waitKey(0)
try:
    os.mkdir("/home/yuzhen/SegFormer-tensorrt/result_red/target_jpg")
except:
    pass
output_direction = "/home/yuzhen/SegFormer-tensorrt/result_red/target_jpg"
name = "target_red_kinfu.jpg"
final_direction = os.path.join(output_direction,name)
cv.imwrite(final_direction,img_data)