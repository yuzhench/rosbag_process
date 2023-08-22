import numpy as np

import cv2 as cv

from PIL import Image, ImageDraw

import os

import_frame = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/red_tag"
output_folder = "/home/yuzhen/SegFormer-tensorrt/result/kinfu_tag"

try:
    os.mkdir(output_folder)
except:
    pass

file_list = os.listdir(import_frame)

files = [file for file in file_list if file.endswith(".jpg")]
files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

margine = 10
tage_height = 30
tage_width = 120

color_list = ((0,0,0),(10,10,255),(180,0,128))
# color_list = ((0,0,0),(10,10,255),(204,204,0))
name_list = ("none","stone","engineeredstone")
# name_list = ("none","stone","plastic")

count = 0
print(color_list[0])
for file in files:
    file_address = os.path.join(import_frame, file)
    print(file_address)
    #load the tag image
    img = cv.imread(file_address)
    cv.imshow("img",img)
    cv.waitKey(0)

    img = Image.open(file_address)
    img_height = img.size[1]
    img_width = img.size[0]

    back_height = img_height 
    back_width = img_width + 2*margine + tage_width

    #create the background:
    back_image = Image.new('RGB',(back_width, back_height),color = (255,255,255))
    back_image.paste(img,(0,0)) 

    tag_x = img_width + margine
    tag_y = 0

    draw = ImageDraw.Draw(back_image)

    for i in range(3):
        draw.rectangle((tag_x,tag_y,tag_x+tage_width,tag_y+tage_height) ,fill=tuple(color_list[i]))
        draw.text((tag_x+margine,tag_y+margine),name_list[i],fill='white')
        tag_y += tage_height+margine

    back_image_np = np.array(back_image)
    back_image_np_1 = cv.cvtColor(back_image_np,cv.COLOR_RGB2BGR)
    cv.imshow("np",back_image_np_1)

    frame_each_name = f"{count}.jpg"
    final_path = os.path.join(output_folder,frame_each_name)
    cv.imwrite(final_path,back_image_np_1)

    # back_image_np = np.array(back_image)
    # cv.imshow("result",back_image_np)
    # cv.waitKey(0)
    count+=1
    if count == 9:
        break

