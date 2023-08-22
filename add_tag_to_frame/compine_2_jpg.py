from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import os

input_kinfu = "/home/yuzhen/SegFormer-tensorrt/result/kinfu_tag"
input_segformer = "/home/yuzhen/SegFormer-tensorrt/result_red/tag"

output_direction = "/home/yuzhen/SegFormer-tensorrt/result_red/kinfu_segformer_tag"

try:
    os.mkdir(output_direction)
except:
    pass

input_kinfu_files = os.listdir(input_kinfu)
input_segformer_files = os.listdir(input_segformer)

file_num = len(input_kinfu_files)

count = 0
for i in range(file_num):

    kinfu = cv.imread(os.path.join(input_kinfu,input_kinfu_files[i]))
    segformer = cv.imread(os.path.join(input_segformer,input_segformer_files[i]))

    cv.imshow("kinfu",kinfu)
    cv.waitKey(0)
    height = kinfu.shape[0]
    width = kinfu.shape[1]
    margin = 30

    back_img_height = height
    back_img_width = 2*width + margin
    
    back_img = Image.new("RGB",(back_img_width,back_img_height),(255,255,255))
    img1 = Image.open(os.path.join(input_kinfu,input_kinfu_files[i]))
    # img1_1 = cv.cvtColor(img1,cv.COLOR_RGB2BGR)
    img2 = Image.open(os.path.join(input_segformer,input_segformer_files[i]))
    # img2_2 = cv.cvtColor(img2,cv.COLOR_RGB2BGR)
    back_img.paste(img2,(0,0))
    back_img.paste(img1,(width+margin,0))

    back_img_np = np.array(back_img)

    back_img_np_1 = cv.cvtColor(back_img_np,cv.COLOR_RGB2BGR)
    # cv.imshow("back_img_np",back_img_np_1)
    # cv.waitKey(0)
    tag_name = os.path.join(output_direction,f"{count}.jpg")
    cv.imwrite(tag_name,back_img_np_1)    

    count+=1