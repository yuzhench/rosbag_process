from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from natsort import natsorted
from tqdm import tqdm

import torch
import cv2
import time
import os
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, TrainingArguments
import torchvision.transforms as TTR
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    # SegFormer model
    id2label = json.load(open('dms_v1.json'))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        "checkpoint-160000")
    model.to(device='cpu')

    # segment-anything model
    # sam = sam_model_registry["default"](checkpoint='sam_vit_h_4b8939.pth')
    sam = sam_model_registry["vit_b"](checkpoint='sam_vit_b_01ec64.pth')
    sam.to(device='cpu')
    mask_generator = SamAutomaticMaskGenerator(sam)


    folder = "result"
    # path = "demo/d1_2023-06-17-15-24-41"
    path = "class"
    try:
        os.makedirs(folder)
    except:
        pass

    out = folder + '/' + os.path.basename(path) + '_out'
    try:
        os.makedirs(out)
    except:
        pass

    rgb_list = natsorted(os.listdir(path+'/rgb'))
    depth_list = natsorted(os.listdir(path+'/depth'))

    for i, (rgb_file, depth_file) in tqdm(enumerate(zip(rgb_list, depth_list))):
        # if i < 350:
        #     continue

        rgb_file = path + '/rgb/' + rgb_file
        depth_file = path + '/depth/' + depth_file

        # SegFormer inference
        image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        # print (f"the image shape is {image.shape[0:2][::-1]}")

        shape = image.shape[0:2][::-1]
        img = process(image)

        pixel_values = img.to(device='cpu')
        pixel_values = pixel_values[None, :, :, :]

        outputs = model(pixel_values=pixel_values)

        logits = outputs.logits.cpu()

        pred_seg = logits.argmax(dim=1)[0]

        segmentation = cv2.resize(pred_seg.numpy().astype('uint8'), shape,
                         interpolation=cv2.INTER_NEAREST)
        
        # for row in segmentation:
        #     for element in row:

        #       print(element, end=" ")


        # segment-anything inference
        image = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        st = time.time()
        sam_masks = mask_generator.generate(image)
        print(time.time() - st)

        # masks_num = len(sam_masks)
        # print(f"the size of the masks is {masks_num}")

        
        

        cls = np.zeros(image.shape[0:2] + (56,))

        # print(f"the shape of cls is {cls.shape}")

        for mask in sam_masks:
            seg_mask = mask['segmentation']
            # print(f"type of seg_mask is ----{type(seg_mask)}")
            # for row in seg_mask:
            #     for element in row:
            #         print(element, end=" ")
            # break
            vals, cnts = np.unique(segmentation * seg_mask, return_counts=True)
            cnts = cnts[vals != 0]
            vals = vals[vals != 0]

            for val, cnt in zip(vals, cnts):
                cls[:, :, val][seg_mask] += cnt

        # normalization
        total_cnt = np.sum(cls, axis=-1, keepdims=True)
        total_cnt[total_cnt == 0] = 1
        cls /= total_cnt

        # only for 2 channels (cls, depth)
        cls = cls.argmax(-1, keepdims=True).astype(int)

        cls = np.concatenate((np.expand_dims(np.load(depth_file), axis=-1), cls), axis=-1)
        # print(f"-----------------------------------the shape of the cls {cls.shape}")
        only_semantic = cls[:,:,1]
        # for row in only_semantic:
        #     for element in row:
        #         print(element,end=" ")
        
        np.save(out+'/'+str(i), cls)


if __name__ == '__main__':
    main()
