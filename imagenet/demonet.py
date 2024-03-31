"""
A simple isotropic DemoNet to illustrate the effectiveness of star operation.
One can vary the parameters, depth: int, dim: int, mode: string in ["sum", "mul"] to validate.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, mode="sum"):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.f = nn.Linear(dim, 6 * dim)
        self.act = nn.GELU()
        self.g = nn.Linear(3 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else 1.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.f(x)
        B, H, W, C = x.size()
        x1, x2 = x.reshape(B, H, W, 2, int(C // 2)).unbind(3)
        x = self.act(x1) + x2 if self.mode == "sum" else self.act(x1) * x2
        x = self.g(x)
        x = input + self.drop_path(self.gamma * x)
        return x


class DemoNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depth=12, dim=384, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 mode="mul", **kwargs):
        super().__init__()
        assert mode in ["sum", "mul"]
        self.num_classes = num_classes
        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i],
                                            layer_scale_init_value=layer_scale_init_value, mode=mode)
                                      for i in range(depth)])

        self.norm = nn.LayerNorm(dim)  # final norm layer
        self.head = nn.Linear(dim, self.num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x).permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = self.norm(x.mean([1, 2]))
        x = self.head(x)
        return x


@register_model
def DemoNet_d12_w64_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=64, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w64_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=64, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w128_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=128, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w128_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=128, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w256_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=256, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w256_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=256, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w320_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=320, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w320_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=320, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w384_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=384, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w384_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=384, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w448_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=448, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w448_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=448, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w32_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=32, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w32_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=32, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w96_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=96, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w96_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=96, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w160_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=160, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w160_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=160, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w224_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=224, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w224_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=224, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w288_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=288, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w288_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=288, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w352_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=352, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w352_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=352, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w416_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=416, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w416_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=416, mode="sum", **kwargs)
    return model


#### for depth evaluation ####
@register_model
def DemoNet_d6_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=6, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d8_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=8, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d10_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=10, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d12_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d14_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=14, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d16_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=16, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d18_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=18, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d20_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=20, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d22_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=22, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d24_w192_sum(pretrained=False, **kwargs):
    model = DemoNet(depth=24, dim=192, mode="sum", **kwargs)
    return model


@register_model
def DemoNet_d6_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=6, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d8_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=8, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d10_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=10, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d12_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=12, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d14_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=14, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d16_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=16, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d18_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=18, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d20_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=20, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d22_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=22, dim=192, mode="mul", **kwargs)
    return model


@register_model
def DemoNet_d24_w192_mul(pretrained=False, **kwargs):
    model = DemoNet(depth=24, dim=192, mode="mul", **kwargs)
    return model
