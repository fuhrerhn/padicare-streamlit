"""Merge a saved classifier-only .pth with pretrained EfficientNet-B0.

Run once in Kaggle (internet on) or in a connected computer. Uses ImageNet1K V1,
matching EfficientNet_B0_Weights.DEFAULT in the training notebook.
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def upgrade(input_file: Path, output_file: Path):
    ckpt = torch.load(input_file, map_location='cpu', weights_only=True)
    if 'backbone_state_dict' in ckpt:
        print('Checkpoint already includes backbone state; copying to output.')
    else:
        print('Fetching EfficientNet-B0 ImageNet1K V1 pretrained weights if not cached...')
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()
        ckpt['backbone_state_dict'] = backbone.state_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_file)
    print('Saved:', output_file, '| MB:', round(output_file.stat().st_size/1024**2, 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build standalone PaddyCare checkpoint')
    parser.add_argument('--input', type=Path, required=True, help='Existing head-only .pth')
    parser.add_argument('--output', type=Path, default=Path('models/paddycare_model.pth'))
    args = parser.parse_args()
    upgrade(args.input, args.output)
