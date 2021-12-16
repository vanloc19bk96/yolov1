import torch.nn as nn
import torch

from utils import build_targets


class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobject=0.5, image_size=448):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobject = lambda_noobject
        self.epsilon = 1e-5

        self.grid_size = image_size // 64

    def forward(self, y_pred, y_true):
        y_true = build_targets(y_pred, self.grid_size)
        batch_size = y_pred.size(0)
        y_pred = y_pred.view(size=(batch_size, self.grid_size, self.grid_size, -1))
        print(y_pred.size())


grid = 7
num_class = 2
num_box = 2
y_pred = torch.rand((2, grid * grid * (num_box * 5 + num_class)))
loss = YoloLoss()

y_true = torch.rand((2, 5))
loss(y_pred, None)
