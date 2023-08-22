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





# clean the directory before we input the new files
import os

def clean_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)




def main():

    #if the input source is a video 


    # # Open the video file
    # video = cv2.VideoCapture('/home/yuzhen/SegFormer-tensorrt/source_video/video2.mp4')  # Replace 'path_to_video_file.mp4' with the actual path to your video file

    # # Check if the video file was successfully opened
    # if not video.isOpened():
    #     print('Error opening video file')
    #     exit()



    #if the imput source is the frames of image 
    image_folder = '/home/yuzhen/SegFormer-tensorrt/class/rgb'

    # Get a list of all JPG files in the folder
    jpg_files = [file for file in os.listdir(image_folder) if file.endswith('.jpg')]

    i = 0

    output_path = '/home/yuzhen/SegFormer-tensorrt/result/only_segfomer'
    

         # Clean the directory
    #clean_directory(output_path)


    #while video.isOpened():
    for file in jpg_files:

        #//if the input file is the video 
        # # Read the current frame
        # ret, frame = video.read()

        image_path = os.path.join(image_folder, file)

        

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        shape = image.shape[0:2][::-1]
        img = process(image)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)

        pixel_values = img
        pixel_values = pixel_values[None, :, :, :]

        ort_session = onnxruntime.InferenceSession("model.onnx")
        pixel_values_numpy = pixel_values.numpy()
        # Run inference
        outputs = ort_session.run(None, {'input.1': pixel_values_numpy})

        logits = outputs[0]  # Assuming that the output is the logits

        pred_seg = logits.argmax(axis=1)[0]

        seg = cv2.resize(pred_seg.astype('uint8'), shape,
                        interpolation=cv2.INTER_NEAREST)
        # seg = np.array(dms46)[seg]
        fig = srgb_colormap[seg]

        objs = np.unique(np.array(dms46)[seg])

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        patches = [mpatches.Patch(color=np.array(t['srgb_colormap'][i]) / 255.,
                                label=t['shortnames'][i]) for i in objs]
        

        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                fontsize='small')




        # plt.imshow(patches)
        # plt.show()




        # output_path = '/home/yuzhen/SegFormer-tensorrt/rosbag_segformer'


        # pil_image_save = Image.fromarray(fig)
        # # Save the image
        # output_filename = f'{i}_semantic.jpg'
        # try:
        #     folder_name =  "only_segformer"
        #     final_direction = os.path.join(output_path,folder_name)
        #     os.makedirs(final_direction)
        # except:
        #     # final_direction = output_path
        #     pass
         
        # output_file_path = os.path.join(final_direction,output_filename)
        # pil_image_save.save(output_file_path)

        # print(f"Image saved at: {output_file_path}")


        #save np file:-----------------------------------------------
        pil_image_save = Image.fromarray(fig)
        # Save the image
        output_filename = f'{i}_semantic.npy'
        output_file_path = os.path.join(output_path, output_filename)
        np.save(output_file_path,seg)

        print(f"npy file saved at: {output_file_path}")
       



        # ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # ax2.imshow(fig, interpolation='none')
        # # np.save('outfile', seg)
        # plt.show()

        # # plt.draw()

        # pdf_file = "/home/yuzhen/SegFormer-tensorrt/result/tag_pdf"
        # pdf_name = f"{i}.pdf"
        # final_direction = os.path.join(pdf_file,pdf_name)
        # plt.savefig(final_direction)


        i+=1

        # # Wait for a key press
        # plt.pause(0.0000000001)
        # plt.waitforbuttonpress()

        # # Close the figure before opening a new one
        # plt.close()
        

    # # Release the video object and close the display window
    # video.release()
    # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
