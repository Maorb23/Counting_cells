import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch.nn.functional as F
from preprocess_cells import CellCountingDataset
import torchvision.models as models

def mse_with_count_regularization(pred_map, target_map, alpha=1, beta=0.001):
    pixel_mse = F.mse_loss(pred_map, target_map)

    pred_count = torch.sum(pred_map, dim=(1, 2, 3))
    true_count = torch.sum(target_map, dim=(1, 2, 3))
    count_mse = F.mse_loss(pred_count, true_count)

    return alpha * pixel_mse + beta * count_mse

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, x1_channels, x2_channels, out_channels, mode='bicubic', alg=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.reduce = nn.Conv2d(x1_channels, out_channels, kernel_size=1)


        self.conv = DoubleConv(out_channels + x2_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)




class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetCellCounter(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()

        # Load ResNet-34 backbone
        resnet = models.resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(
            resnet.conv1,  # out: 64
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1  # out: 64
        self.layer2 = resnet.layer2  # out: 128
        self.layer3 = resnet.layer3  # out: 256
        self.layer4 = resnet.layer4  # out: 512

        # Decoder
        factor = 2
        self.up1 = Up(512, 256, 256)  # x1=512, x2=256 → output=256
        self.up2 = Up(256, 128, 128)  # x1=256, x2=128 → output=128
        self.up3 = Up(128, 64, 64)    # x1=128, x2=64  → output=64
        self.up4 = Up(64, 64, 64)     # x1=64,  x2=64  → output=64


        self.outc = OutConv(64, 1)

    def forward(self, x):
        #print("Input:", x.shape)  # [B, 3, H, W]

        x0 = self.layer0(x)       # 64
        #print("x0 (after conv1/bn1/relu):", x0.shape)

        x1 = self.pool(x0)        # 64
        #print("x1 (after maxpool):", x1.shape)

        x2 = self.layer1(x1)      # 64
        #print("x2 (layer1):", x2.shape)

        x3 = self.layer2(x2)      # 128
        #print("x3 (layer2):", x3.shape)

        x4 = self.layer3(x3)      # 256
        #print("x4 (layer3):", x4.shape)

        x5 = self.layer4(x4)      # 512
        #print("x5 (layer4):", x5.shape)

        # Decoder path
        x = self.up1(x5, x4)
        #print("After up1:", x.shape)

        x = self.up2(x, x3)
        #print("After up2:", x.shape)

        x = self.up3(x, x2)
        #print("After up3:", x.shape)

        x = self.up4(x, x1)
        #print("After up4:", x.shape)

        density_map = F.relu(self.outc(x))
        #print("Final density map:", density_map.shape)
        density_map = F.interpolate(density_map, size=(256, 256), mode='bicubic', align_corners=False)
        scale_factor = (density_map.shape[2] * density_map.shape[3]) / (x.shape[2] * x.shape[3])
        density_map = density_map / scale_factor


        count = torch.sum(density_map, dim=(1, 2, 3))
        return density_map, count
    


class UNetCellCounterEffNet(nn.Module):
    def __init__(self, backbone="efficientnet-b4", pretrained=True):
        super().__init__()
        # 1) Build a Unet with EfficientNet‑B4 encoder
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            decoder_channels=(256, 128, 64, 64,64),   # you can adjust if you like
            activation=None
        )

    def forward(self, x):
        # returns [B,1,H,W]
        density_map = torch.relu(self.unet(x))
        count = density_map.sum(dim=(1, 2, 3))
        return density_map, count



def train_unet(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=50):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    mae_list = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0

        for inputs,_, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            density_maps, _ = model(inputs)

            # Calculate loss (L2 / MSE loss)
            loss = criterion(density_maps, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        mae = 0  # Mean Absolute Error for count

        with torch.no_grad():
            for batch_idx, (inputs, labels, targets) in enumerate(val_loader):
              inputs, labels, targets = inputs.to(device), labels.to(device),targets.to(device)
              density_maps, predicted_counts = model(inputs)

              loss = criterion(density_maps, targets)
              val_loss += loss.item()

              true_counts = torch.sum(labels, dim=(1, 2, 3))
              mae += torch.abs(predicted_counts - true_counts).sum().item()
              """
              if batch_idx % 10 == 0:
                # Visualize just the first sample in the batch
                print(mae)
                print(f"\n True count: {true_counts[0].item():.1f}, Predicted count: {predicted_counts[0].item():.1f}")

                plt.subplot(1, 2, 1)
                plt.imshow(targets[0].squeeze().cpu(), cmap='jet')
                plt.title(f"GT Density (Sum: {true_counts[0].item():.1f})")

                plt.subplot(1, 2, 2)
                plt.imshow(density_maps[0].squeeze().cpu(), cmap='jet')
                plt.title(f"Predicted (Sum: {predicted_counts[0].item():.1f})")
                plt.colorbar()
                plt.show()
              """

        # Then outside the loop:
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_mae = mae / len(val_loader.dataset)
        mae_list.append(avg_mae)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, MAE: {avg_mae:.2f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_unet_cell_counter.pth')
            print(f'Model saved with Val Loss: {best_val_loss:.4f}')

    return train_losses, val_losses, mae_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet for Cell Counting')
    parser.add_argument('--train_loader', type=str, default='data/processed/train_loader.pth', help='Path to train loader')
    parser.add_argument('--val_loader', type=str, default='data/processed/val_loader.pth', help='Path to validation loader')
    parser.add_argument('--manual', action='store_true', help='Manually specify the device')
    parser.add_argument('--smp', action='store_true', help='Use segmentation models pytorch')
    parser.add_argument('--summary', action='store_true', help='Print model summary')
    args = parser.parse_args()
    train_loader = torch.load(args.train_loader, weights_only=False)
    val_loader = torch.load(args.val_loader, weights_only=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.manual:
        model = UNetCellCounter().to(device)
    elif args.smp:
        model = UNetCellCounterEffNet(backbone="efficientnet-b4", pretrained=True).to(device)
    else:
        raise ValueError("Please specify a model type")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()  # L2 loss for density map regression
    if args.summary:    
        from torchsummary import summary
        summary(model, input_size = (3,256,256))



    train_losses, val_losses, mae_list = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=mse_with_count_regularization,
        device=device,
        num_epochs=30
    )
