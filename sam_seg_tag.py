import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

sam_seg_folder = "/home/yuzhen/SegFormer-tensorrt/result/red_out"
npy_files = [file for file in os.listdir(sam_seg_folder) if file.endswith("_semantic.npy")]


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

i = 0

for file in npy_files:
    direction = os.path.join(sam_seg_folder,file)
    npy_data = np.load(direction)
    patches = [mpatches.Patch(color=np.array(t['srgb_colormap'][i]) / 255.,
                                label=t['shortnames'][i]) for i in npy_data]
        

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
            fontsize='small')
    
    fig = srgb_colormap[npy_data]
    plt.imshow(fig, interpolation='none')
    plt.show()
    i+=1