import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from models import DnCNN
from utils import batch_PSNR  # Ensure this function exists in utils.py

# Set CUDA device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Argument Parser
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='Path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='Test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='Noise level used on test set')
opt = parser.parse_args()

# Normalization function
def normalize(data):
    return data / 255.0

# Main function
def main():
    print('Loading model...\n')

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model = nn.DataParallel(net).to(device)

    # Load trained model
    model_path = os.path.join(opt.logdir, 'net.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print('Loading test data...\n')
    files_source = sorted(glob.glob(os.path.join('data', opt.test_data, '*.png')))

    psnr_test = 0

    for f in files_source:
        # Load and preprocess image
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        Img = normalize(np.float32(Img))
        Img = np.expand_dims(Img, (0, 1))  # Convert to (1, 1, H, W)
        ISource = torch.tensor(Img, dtype=torch.float32, device=device)

        # Add noise
        noise = torch.randn_like(ISource) * (opt.test_noiseL / 255.0)
        INoisy = ISource + noise

        # Forward pass (inference)
        with torch.no_grad():
            Out = torch.clamp(INoisy - model(INoisy), 0.0, 1.0)

        # Compute PSNR
        psnr = batch_PSNR(Out, ISource, 1.0)
        psnr_test += psnr
        print(f"{f} PSNR: {psnr:.2f}")

    # Print average PSNR
    psnr_test /= len(files_source)
    print(f"\nAverage PSNR on test data: {psnr_test:.2f}")

if __name__ == "__main__":
    main()
