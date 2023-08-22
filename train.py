import torch
import cv2
from torch import nn
import evaluate
from torch.utils import data
import os
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer
import torchvision.transforms as TTR
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as transforms
import json
import numpy as np

id2label = json.load(open('dms_v1.json'))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

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


class PairTransform:
    def __init__(self, rotation, flip):
        self.rotation = rotation
        self.flip = flip
        self.color = transforms.ColorJitter(brightness=0.25, contrast=0.10, saturation=0.10)

    def __call__(self, image, mask):
        image = self.color(image)

        mask = mask.unsqueeze(0)

        # Random horizontal flip
        if torch.rand(1) < self.flip:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random rotation
        if self.rotation:
            angle = torch.randint(-self.rotation, self.rotation, (1,)).item()
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        mask = mask.squeeze(0)

        return image, mask


class DatasetReader(data.Dataset):
    def __init__(self, root_dir, split):
        print(f'======={split}:DMS==================')
        self.data_path = root_dir
        self.num_classes = 46
        assert self.num_classes == 46

        self.split = split

        self.images_path = os.path.join(self.data_path, f'images/{self.split}')
        self.anno_path = os.path.join(self.data_path, f'labels/{self.split}')
        data = [i for i in os.listdir(self.images_path) if '.jpg' in i]
        data = [x for x in data if os.path.exists(self.segmap_path(x))]
        self.data = data
        assert len(self.data) > 0, f'No image and label pairs found, check that data_path is correct'

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        self.mean = mean
        self.std = std

        self.setup_class_mapping()

        self.trans = PairTransform(30, 0.5)

        print(f'valid_classes {self.valid_classes}')
        print(f'merge_dict {self.merge_dict}')
        print(f'class_map {self.class_map}')

        print(f'{self.split} has this many images: {len(self.data)}')

    def setup_class_mapping(self):
        # filter classes not predicted
        taxonomy_file = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
        self.all_names = taxonomy_file['names']
        self.valid_classes = dms46
        self.valid_classes.sort()
        assert (
                len(self.valid_classes) == self.num_classes
        ), 'valid_classes doesnt match num_classes'
        self.merge_dict = {
            0: set([i for i in range(len(self.all_names))]) ^ set(self.valid_classes)
        }
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

    def segmap_path(self, filename):
        return os.path.join(self.anno_path, filename[:-4] + '.png')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]
        img_path = os.path.join(self.images_path, filename)
        seg_path = self.segmap_path(filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        img = cv2.resize(img, (512, 512),
                         interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(seg, (512, 512),
                         interpolation=cv2.INTER_LINEAR)

        assert len(seg.shape) == 2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert (
                img.shape[:2] == seg.shape
        ), f'Shape mismatch for {img_path}, {img.shape[:2]} vs. {seg.shape}'
        label = self.encode_segmap(seg)

        image = np.copy(img)

        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        label = torch.from_numpy(label).long()
        image = TTR.Normalize(self.mean, self.std)(image)

        image, label = self.trans(image, label)

        return {'pixel_values': image,
                'labels': label,
                'filename': filename}

    def encode_segmap(self, mask):
        # combine categories
        for i, j in self.merge_dict.items():
            for k in j:
                mask[mask == k] = i

        # assign ignored index
        mask[mask == 0] = ignore_index

        # map valid classes to id
        # use valid_classes sorted so will not remap.
        for valid_class in self.valid_classes:
            assert valid_class > self.class_map[valid_class]
            mask[mask == valid_class] = self.class_map[valid_class]

        return mask


metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=ignore_index,
            # reduce_labels=feature_extractor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics


def main():
    root_dir = 'data/DMS_v1'

    train_dataset = DatasetReader(root_dir=root_dir, split='train')
    valid_dataset = DatasetReader(root_dir=root_dir, split='valid')

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))

    # load id2label mapping from a JSON on the hub
    id2label = json.load(open('dms_v1.json'))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1",
                                                             id2label=id2label,
                                                             label2id=label2id,
                                                             )

    max_iters = 160000
    lr = 0.00006
    batch_size = 10

    training_args = TrainingArguments(
        "segformer-b1-segments-outputs",
        learning_rate=lr,
        max_steps=max_iters,
        per_device_train_batch_size=batch_size,  # TODO
        per_device_eval_batch_size=1,  # TODO
        save_total_limit=5,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=4000,
        eval_steps=500,
        logging_steps=50,
        # eval_accumulation_steps=5
        # load_best_model_at_end=True
    )
    training_args = training_args.set_lr_scheduler(name="polynomial",
                                                   warmup_ratio=1e-06,
                                                   warmup_steps=1500,
                                                   max_steps=max_iters)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()

# fig = plt.figure()
# seg = label
# seg[seg == 255] = 0
# seg = np.array(dms46)[seg]
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.imshow(image.permute(1, 2, 0), interpolation='none')
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.imshow(seg, interpolation='none')

# seg = label
# seg[seg == 255] = 0
# seg = np.array(dms46)[seg]
# ax3 = fig.add_subplot(2, 2, 3)
# ax3.imshow(image.permute(1, 2, 0), interpolation='none')
# ax4 = fig.add_subplot(2, 2, 4)
# ax4.imshow(seg, interpolation='none')
#
# plt.show()
