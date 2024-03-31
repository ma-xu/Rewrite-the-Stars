"""
Modified from EfficientFormer Toolbox
https://github.com/snap-research/EfficientFormer/blob/main/toolbox.py

Modified by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import argparse
import coremltools as ct
from models import *
import starnet
import timm
import os
from timm.models import create_model


def parse():
    parser = argparse.ArgumentParser(description='EfficientFormer Toolbox')
    parser.add_argument('--model', default="starnet_s1", metavar='ARCH')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, metavar='PATH',
                        help='path to checkpoint')
    parser.add_argument("--resolution", default=224, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    model = create_model(model_name=args.model, pretrained=args.pretrained)
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
        print('load success, model is initialized with pretrained checkpoint')
    except:
        print('model initialized without pretrained checkpoint')

    model.eval()
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)

    example_input = dummy_input
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)

    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(shape=example_input.shape, channel_first=True)]
    )
    model.save(os.path.join("coreml_models", args.model + ".mlmodel"))
    print('successfully export coreML')
