import cv2
import numpy as np
import os
import math
import torch
import torchvision as tv
from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_CONFIGS = {
        'CKDN': {
            'net_opts': {
                'type': 'CKDN',
                'pretrained_model_path': './experiments/pretrained_models/CKDN/model_best.pth.tar',
                'use_diff_preprocess': False,
                },
            'metric_mode': 'FR',
            'preprocess_x': tv.transforms.Compose([
                tv.transforms.Resize(int(math.floor(288/0.875)), tv.transforms.InterpolationMode.BICUBIC),
                tv.transforms.CenterCrop(288),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]),
            'preprocess_y': tv.transforms.Compose([
                tv.transforms.Resize(int(math.floor(288/0.875)), tv.transforms.InterpolationMode.NEAREST),
                tv.transforms.CenterCrop(288),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]),
            },
        'LPIPS': {
            'net_opts': {
                'type': 'LPIPS',
                'net': 'alex',
                'version': '0.1',
                'pretrained_model_path': './experiments/pretrained_models/LPIPS/v0.1/alex.pth',
                },
            'metric_mode': 'FR',
            },
        'DISTS': {
            'net_opts': {
                'type': 'DISTS',
                'pretrained_model_path': './experiments/pretrained_models/DISTS/weights.pt',
                },
            'metric_mode': 'FR',
            },
        }
