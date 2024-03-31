"""
Simplified DemoNet for 2D points.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model


class Block(nn.Module):
    def __init__(self, dim, mode="sum", act=nn.ReLU):

        super().__init__()
        self.mode=mode
        self.norm = nn.LayerNorm(dim)
        self.f = nn.Linear(dim, 6 * dim)
        self.act = act()
        self.g = nn.Linear(3*dim, dim)

    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.f(x)
        B, C = x.size()
        x1, x2 = x.reshape(B, 2, int(C//2)).unbind(1)
        x = self.act(x1)+x2 if self.mode == "sum" else self.act(x1)*x2
        x = self.g(x)
        x = input + x
        return x


class DemoNet(nn.Module):
    def __init__(self, in_chans=2, num_classes=2, depth=4, dim=100, act=nn.ReLU,
                 mode="sum", **kwargs):
        super().__init__()
        assert mode in ["sum", "mul"]
        self.num_classes = num_classes
        self.stem = nn.Linear(in_chans, dim)
        self.blocks = nn.Sequential(*[Block(dim=dim, act=act, mode=mode) for i in range(depth)])

        self.norm = nn.LayerNorm(dim) # final norm layer
        self.head = nn.Linear(dim, self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x

@register_model
def demonet_2d_sum(pretrained=False, num_classes=2,  **kwargs):
    model = DemoNet(depth=4, dim=100, num_classes=num_classes, act=nn.ReLU, mode="sum")
    return model

@register_model
def demonet_2d_mul(pretrained=False, num_classes=2,  **kwargs):
    model = DemoNet(depth=4, dim=100, num_classes=num_classes, act=nn.ReLU, mode="mul")
    return model
