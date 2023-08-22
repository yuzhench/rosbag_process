import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import time


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
    image = cv2.imread("demo/rgb_1.jpg", cv2.IMREAD_COLOR)
    shape = image.shape[0:2][::-1]
    img = process(image)

    pixel_values = img
    pixel_values = pixel_values[None, :, :, :]

    # Load the TensorRT engine
    with open("trt_engine.trt", "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Create a context
    context = engine.create_execution_context()

    input_data = pixel_values

    numpy_input_data = input_data.numpy()
    byte_input_data = numpy_input_data.tobytes()
    d_input = cuda.mem_alloc(1 * numpy_input_data.nbytes)
    dims = engine.get_binding_shape(1)
    dims_list = list(dims)
    volume = np.prod(dims_list)

    d_output = cuda.mem_alloc(int(volume * np.dtype(np.float32).itemsize))

    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, byte_input_data, stream)

    # Run inference
    start_time = time.time()

    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    end_time = time.time()

    print("Inference Time: ", end_time - start_time, "seconds")

    output_data = np.empty(engine.get_binding_shape(1), dtype=np.float32)

    cuda.memcpy_dtoh_async(output_data, d_output, stream)

    stream.synchronize()

    logits = output_data

    pred_seg = logits.argmax(axis=1)[0]

    seg = cv2.resize(pred_seg.astype('uint8'), shape,
                     interpolation=cv2.INTER_NEAREST)
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
