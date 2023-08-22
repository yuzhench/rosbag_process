import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageFont
import cv2

dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
         24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
         50, 51, 52, 53, 56, ]
t = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
srgb_colormap = [
    t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) #if i in dms46
]
srgb_colormap.append([0, 0, 0])  # color for ignored index
srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)


# def colorful(data):
#     color_img = np.zeros((data.shape[0],data.shape[1],3),dtype=np.uint8)
#     for row in data:
#         for element in row:
#             if element 


# input_folder = "/home/yuzhen/SegFormer-tensorrt/result/class_out"
# output_folder = "/home/yuzhen/SegFormer-tensorrt/result/tag"
input_folder = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out"
output_folder = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out/SamSegformer_with_tag"



kinfu_folder = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/red/0_semantic.npy"

# test_direction = "/home/yuzhen/SegFormer-tensorrt/result/class_out/0_semantic.npy"
# my_list = []
# data = np.load(test_direction)
# for row in data:
#     for element in row:
#         if element not in my_list:
#             my_list.append(element)

# my_list.sort()
# for index in my_list:
#     print(f"{index}  ")



file_list = [file for file in os.listdir(input_folder) if file.endswith("_semantic.npy")]

count = 0





for file in file_list:
    #get the exact direction for each file 
    file_direction = os.path.join(input_folder,file)
    data = np.load(file_direction)

    # print(f"the shape of the dara is {data.shape}")
    #get the jpg file from npy file:
    jpg_data = srgb_colormap[data]

    #find the set of all the class index 
    list = set()

    for row in data:
        for element in row:
            if element not in list: #and element in dms46:
                list.add(element)

    # print(list)

    #get the source data from 2 json files 
    with open('dms_v1.json') as tag_source:
        materials = json.load(tag_source)

    with open("taxonomy.json") as jpg_source:
        color = json.load(jpg_source)


    #create a dictionary of the realted material name and the related color 
    tag_dic = {}

    for indux in list:
        if str(indux) in materials:
            material_name = materials[str(indux)]
            tag_dic[material_name] = color["srgb_colormap"][indux]

    # print (tag_dic)

    # print(jpg_data.shape)

    tag_num = len(tag_dic)

    #get the size of the picture
    picture_heigth = jpg_data.shape[0]
    picture_width = jpg_data.shape[1]

    #define the size of the tag 
    tag_height = 30
    tag_width = 120
    margin = 10

    #create the new_image
    new_image_width = picture_width + tag_width + 2*margin
    new_image_height = max(picture_heigth,tag_num*(margin+tag_height) + margin)
    
    new_image = Image.new('RGB',(new_image_width,new_image_height),color=(255,255,255))

    # new_image_np = np.array(new_image)

    # cv2.imshow("new_image",new_image_np)
    # cv2.waitKey(0)

    #past the picture to the new_image 
    jpg_image = Image.fromarray(jpg_data, 'RGB')
    new_image.paste(jpg_image,(0,0))

    # new_image_np = np.array(new_image)

    # cv2.imshow("new_image",new_image_np)
    # cv2.waitKey(0)

    #the position of the tag rectangle 
    tag_x = picture_width + margin
    tag_y = tag_height
 

    #draw the tags to the new_image 
    for key in tag_dic:
        #define the tag_box position
        tag_box = (tag_x, tag_y, tag_x+tag_width,tag_y+tag_height)
        draw = ImageDraw.Draw(new_image)
        # font = ImageFont.truetype(None,16)
        #draw the colorful box 
        # print(tag_dic[key])
        color = tag_dic[key]
        # color = tuple(int(c * 255) for c in color)
        # print(color)
        color = tuple(color)
        draw.rectangle(tag_box,fill = color)

        #add the material name 
        text_position = (tag_x+margin,tag_y+margin)
        draw.text(text_position,key,fill='white')

        #udpdate the y direction of the tag
        tag_y += tag_height+margin

        img = np.array(new_image)
        
    # new_image_np = np.array(new_image)
    # cv2.imshow("new_image",new_image_np)
    # cv2.waitKey(0)
    out_file_name = f"{count}_tag.jpg"
    out_file_final_directory = os.path.join(output_folder,out_file_name)
    new_image_np = np.array(new_image)
    cv2.imwrite(out_file_final_directory,new_image_np)
    print(f"{count}_jpg update successfully")
    count+=1



