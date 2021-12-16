import torch.nn as nn
import torch
import torch.nn as layer_module

num_block_repeats = [2]


class Block(nn.Module):
    def __init__(self, in_channels, intermediate_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels * 2,
                               kernel_size=(3, 3))

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x


class YoloV1(nn.Module):
    def __init__(self, config, num_classes, num_boxes_per_cell):
        super().__init__()

        layers = config.init_layers(layer_module)

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
