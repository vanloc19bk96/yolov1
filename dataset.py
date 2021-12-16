from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import cv2.cv2 as cv2

from transform import CustomTransform
from utils import read_annotation


class CustomDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.image_paths = list(glob.glob(self.root + '/*.jpg'))
        self.annotation_paths = list(glob.glob(self.root + '/*.xml'))
        print(len(self.image_paths))
        print(len(self.annotation_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_path = list(filter(lambda x: Path(x).stem == Path(image_path).stem,
                                      self.annotation_paths))[0]

        image = cv2.imread(image_path)
        labels = read_annotation(annotation_path)
        if self.transforms:
            transformed_data = self.transforms(image, labels)
            image = transformed_data['image']
            labels = transformed_data['bboxes']

        return image, labels


def collate_fn(batch):
    img_list = []
    label_list = []
    for a, b in batch:
        img_list.append(a)
        label_list.append(b)
    return torch.stack(img_list, 0), label_list


transforms = CustomTransform()
custom = CustomDataset('data', transforms)
data_loader = DataLoader(dataset=custom, batch_size=32, shuffle=False, collate_fn=collate_fn)
next(iter(data_loader))
