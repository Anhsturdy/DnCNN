import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import batch_PSNR, weights_init_kaiming  # Ensure these functions exist

# Set CUDA device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Argument Parser
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help="Run prepare_data or not")
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help="Path of log files")
parser.add_argument("--mode", type=str, default="S", help="With known noise level (S) or blind training (B)")
parser.add_argument("--noiseL", type=float, default=25, help="Noise level; ignored when mode=B")
parser.add_argument("--val_noiseL", type=float, default=25, help="Noise level used on validation set")
opt = parser.parse_args()

def main():
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset...\n")
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)

    print(f"# of training samples: {len(dataset_train)}\n")

    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    net = net.to(device)

    criterion = nn.MSELoss(reduction="sum").to(device)  # Updated `size_average=False` → `reduction='sum'`
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B = [0, 55]  # Ignored when opt.mode == 'S'

    for epoch in range(opt.epochs):
        # Adjust learning rate
        current_lr = opt.lr if epoch < opt.milestone else opt.lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f"Learning rate: {current_lr:.6f}")

        # Training loop
        net.train()
        for i, data in enumerate(loader_train):
            img_train = data.to(device)

            # Add noise
            if opt.mode == "S":
                noise = torch.randn_like(img_train) * (opt.noiseL / 255.0)
            else:  # Blind training mode
                noise = torch.zeros_like(img_train)
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size(0))
                for n in range(noise.size(0)):
                    noise[n] = torch.randn_like(noise[n]) * (stdN[n] / 255.0)

            imgn_train = img_train + noise
            noise = noise.to(device)

            # Forward pass
            optimizer.zero_grad()
            out_train = net(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size(0) * 2)
            loss.backward()
            optimizer.step()

            # Compute PSNR
            with torch.no_grad():
                out_train_clamped = torch.clamp(imgn_train - net(imgn_train), 0.0, 1.0)
                psnr_train = batch_PSNR(out_train_clamped, img_train, 1.0)

            print(f"[Epoch {epoch+1}][{i+1}/{len(loader_train)}] Loss: {loss.item():.4f} PSNR_train: {psnr_train:.4f}")

            # Log training progress
            if step % 10 == 0:
                writer.add_scalar("Loss", loss.item(), step)
                writer.add_scalar("PSNR on training data", psnr_train, step)
            step += 1

        # Validation phase
        net.eval()
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = dataset_val[k].unsqueeze(0).to(device)
                noise = torch.randn_like(img_val) * (opt.val_noiseL / 255.0)
                imgn_val = img_val + noise

                out_val = torch.clamp(imgn_val - net(imgn_val), 0.0, 1.0)
                psnr_val += batch_PSNR(out_val, img_val, 1.0)

        psnr_val /= len(dataset_val)
        print(f"\n[Epoch {epoch+1}] PSNR_val: {psnr_val:.4f}")
        writer.add_scalar("PSNR on validation data", psnr_val, epoch)

        # Save images for visualization
        with torch.no_grad():
            out_train_clamped = torch.clamp(imgn_train - net(imgn_train), 0.0, 1.0)
            Img = utils.make_grid(img_train, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(imgn_train, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train_clamped, nrow=8, normalize=True, scale_each=True)

            writer.add_image("Clean Image", Img, epoch)
            writer.add_image("Noisy Image", Imgn, epoch)
            writer.add_image("Reconstructed Image", Irecon, epoch)

        # Save model
        model_path = os.path.join(opt.outf, f"net_epoch_{epoch+1}.pth")
        torch.save(net.state_dict(), model_path)
        print(f"Model saved: {model_path}")

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size= 50, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size= 40, stride=10, aug_times=2)
    main()
