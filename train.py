import argparse

import torch

from model import YoloV1
from parser import ConfigParser

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Yolo-v1')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)

    x = torch.rand((2, 3, 448, 448))
    model = YoloV1(config, 20, 2)
    out = model(x)
    print(out.size())
    print(model)
