import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self, scale=0.01):
        super(SmoothL1Loss, self).__init__()
        self.scale = scale

    def __repr__(self):
        return "SmoothL1Loss()"

    def forward(self, output, groundtruth):
        """
            input:  b x n x 2
            output: b x n x 1 => 1
        """
        delta_2 = (output - groundtruth).pow(2).sum(dim=-1, keepdim=False)
        #delta = delta_2.clamp(min=1e-12).sqrt()
        delta = delta_2.sqrt()
        loss = torch.where(\
                delta_2 < self.scale * self.scale, \
                0.5 / self.scale * delta_2, \
                delta - 0.5 * self.scale)
        return loss.mean()
