import configparser
import os
from collections import OrderedDict
import xml.etree.ElementTree as ET

import torch

CLASS_NAMES = ['dog', 'cat']


def read_cfg(fname):
    config = configparser.ConfigParser(defaults=None, dict_type=MultiDict, strict=False)
    config.read(fname)
    return config


def read_annotation(path):
    if not os.path.exists(path):
        raise FileExistsError("Annotations file is not exists!")

    tree = ET.parse(path)
    annotation = tree.getroot()
    labels = []
    size = annotation.find('size')
    img_w, img_h = int(size.find('width').text), int(size.find('height').text)
    for boxes in annotation.iter('object'):
        label = boxes.find('name').text
        y_min = int(float(boxes.find("bndbox/ymin").text))
        x_min = int(float(boxes.find("bndbox/xmin").text))
        y_max = int(float(boxes.find("bndbox/ymax").text))
        x_max = int(float(boxes.find("bndbox/xmax").text))

        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        x_center = (x_max + x_min) / 2 / img_w
        y_center = (y_max + y_min) / 2 / img_h
        label = CLASS_NAMES.index(label)
        labels.append([x_center, y_center, w, h, label])
    return labels


class MultiDict(OrderedDict):
    _unique = 0

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += 1
            key += '-' + str(self._unique)
        OrderedDict.__setitem__(self, key, val)


def build_targets(groundtruth, grid_size):
    label = torch.zeros((5 + len(CLASS_NAMES), grid_size, grid_size))

    for g in groundtruth:
        center_x, center_y, w, h, clazz = g
        x = (center_x % (1 / grid_size)) # left offset
        y = (center_y % (1 / grid_size)) # top offset
        x_index = int(center_x * grid_size)
        y_index = int(center_y * grid_size)
        label[0, y_index, x_index] = 1
        label[1:5, y_index, x_index] = torch.Tensor([x, y, w, h])
        label[5 + len(CLASS_NAMES), y_index, x_index] = 1

    return label
