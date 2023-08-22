import torch
import cv2
from torch import nn
import time
import os
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, TrainingArguments
import torchvision.transforms as TTR
import json
from torch.utils.data import DataLoader
# from datasets import load_metric
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
    image = cv2.imread("/home/yuzhen/SegFormer-tensorrt/dog.jpg", cv2.IMREAD_COLOR)
    
    cv2.imshow("img",image)
    cv2.waitKey(0)

    shape = image.shape[0:2][::-1]
    img = process(image)

    # load id2label mapping from a JSON on the hub
    id2label = json.load(open('dms_v1.json'))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        "segformer-b1-segments-outputs/checkpoint-160000")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pixel_values = img.to(device)
    pixel_values = pixel_values[None, :, :, :]

    start_time = time.time()

    outputs = model(pixel_values=pixel_values)

    end_time = time.time()

    print("Inference Time: ", end_time - start_time, "seconds")

    logits = outputs.logits.cpu()  # shape (batch_size, num_labels, height/4, width/4)

    # upsampled_logits = nn.functional.interpolate(
    #     logits,
    #     size=pixel_values.size[::-1],  # (height, width)
    #     mode='bilinear',
    #     align_corners=False
    # )

    pred_seg = logits.argmax(dim=1)[0]

    seg = cv2.resize(pred_seg.numpy().astype('uint8'), shape,
                     interpolation=cv2.INTER_NEAREST)

    # kernel = np.ones((20, 20), np.uint8)
    # seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)

    # seg = np.array(dms46)[seg]
    fig = srgb_colormap[seg]

    objs = np.unique(np.array(dms46)[seg])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


    patches = [mpatches.Patch(color=np.array(t['srgb_colormap'][i]) / 255.,
                              label=t['shortnames'][i]) for i in objs]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
               fontsize='small')

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.imshow(fig, interpolation='none')
    np.save('outfile', seg)
    plt.show()


if __name__ == '__main__':
    main()
