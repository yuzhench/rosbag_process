import cv2
import os
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


def main():

    # load id2label mapping from a JSON on the hub
    id2label = json.load(open('dms_v1.json'))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    mat = np.load("result/d1_2023-06-17-15-24-41_out/18.npy")

    seg = mat[:, :, 1]

    # seg = np.array(dms46)[seg]
    fig = srgb_colormap[seg]

    objs = np.unique(np.array(dms46)[seg])

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    f, ax2 = plt.subplots(1, 1, figsize=(10, 5))


    patches = [mpatches.Patch(color=np.array(t['srgb_colormap'][i]) / 255.,
                              label=t['shortnames'][i]) for i in objs]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
               fontsize='small')

    # ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.imshow(fig, interpolation='none')
    np.save('outfile', seg)
    plt.show()


if __name__ == '__main__':
    main()
