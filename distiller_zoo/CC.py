from __future__ import print_function

import torch
import torch.nn as nn


class Correlation(nn.Module):
    """Correlation congruence for knowledge distillation, ICCV 2019"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss
