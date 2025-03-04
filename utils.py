import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr  # Updated import

def weights_init_kaiming(m):
    """Applies Kaiming He initialization to model layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")  # Updated deprecated syntax
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias, 0.0)

def batch_PSNR(img, imclean, data_range=1.0):
    """Computes average PSNR for a batch of images."""
    img_np = img.cpu().numpy().astype(np.float32)
    clean_np = imclean.cpu().numpy().astype(np.float32)
    psnr_total = sum(psnr(clean_np[i], img_np[i], data_range=data_range) for i in range(img_np.shape[0]))
    return psnr_total / img_np.shape[0]

def data_augmentation(image, mode):
    """Applies geometric transformations for data augmentation."""
    out = np.transpose(image, (1, 2, 0))  # Convert to HxWxC
    
    if mode == 1:
        out = np.flipud(out)  # Vertical flip
    elif mode == 2:
        out = np.rot90(out, k=1)  # Rotate 90°
    elif mode == 3:
        out = np.flipud(np.rot90(out, k=1))  # Rotate 90° + Vertical flip
    elif mode == 4:
        out = np.rot90(out, k=2)  # Rotate 180°
    elif mode == 5:
        out = np.flipud(np.rot90(out, k=2))  # Rotate 180° + Vertical flip
    elif mode == 6:
        out = np.rot90(out, k=3)  # Rotate 270°
    elif mode == 7:
        out = np.flipud(np.rot90(out, k=3))  # Rotate 270° + Vertical flip

    return np.transpose(out, (2, 0, 1))  # Convert back to CxHxW
