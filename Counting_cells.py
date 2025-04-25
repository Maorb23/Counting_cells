#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Read images
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pillow
from PIL import Image


# In[3]:


get_ipython().system('pip install segmentation-models-pytorch')


# In[2]:


import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# In[3]:


# upload folder from drive
drive_folder_path = '/content/drive/MyDrive/counting_cells_data'
train_image_folder = '/kaggle/input/counting-cells/counting_cells_data/train_images'
train_label_folder = '/kaggle/input/counting-cells/counting_cells_data/train_labels'
val_image_folder = '/kaggle/input/counting-cells/counting_cells_data/val_images'
val_label_folder = '/kaggle/input/counting-cells/counting_cells_data/val_labels'

train_image_paths = [os.path.join(train_image_folder, fname) for fname in sorted(os.listdir(train_image_folder))]
train_label_paths = [os.path.join(train_label_folder, fname) for fname in sorted(os.listdir(train_label_folder))]
val_image_paths = [os.path.join(val_image_folder, fname) for fname in sorted(os.listdir(val_image_folder))]
val_label_paths = [os.path.join(val_label_folder, fname) for fname in sorted(os.listdir(val_label_folder))]


# In[6]:


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

class QuarterShuffle:
    def __init__(self, p=0.5):
        self.p = p

    def shuffle(self, img):
        """
        Shuffle the quarters of the image with probability p.
        Works on torch.Tensor [C, H, W] or PIL.Image.Image.
        Returns a torch.Tensor [C, H, W].
        """
        if random.random() > self.p:
            return TF.to_tensor(img) if isinstance(img, Image.Image) else img

        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)  # Convert to tensor [C, H, W]

        C, H, W = img.shape
        h_half, w_half = H // 2, W // 2

        # Extract quarters
        UL = img[:, :h_half, :w_half].clone()
        UR = img[:, :h_half, w_half:].clone()
        LL = img[:, h_half:, :w_half].clone()
        LR = img[:, h_half:, w_half:].clone()

        quarters = [UL, UR, LL, LR]
        idx = list(range(4))
        random.shuffle(idx)  # Random permutation

        # Rebuild image from shuffled quarters
        top = torch.cat([quarters[idx[0]], quarters[idx[1]]], dim=2)
        bottom = torch.cat([quarters[idx[2]], quarters[idx[3]]], dim=2)
        shuffled = torch.cat([top, bottom], dim=1)

        return shuffled


class CellCountingDataset(Dataset):
    def __init__(self, image_paths, label_paths, mode='train', img_transform=None, gt = False, qt = True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode  # 'train' or 'val'
        self.gt = gt
        self.qt = qt

        # Set transforms
        self.img_transform = img_transform or self._get_img_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load image and dot mask ---
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        dot_mask = cv2.imread(self.label_paths[idx])
        density_map = self._generate_density_map(dot_mask)
        dot_mask = cv2.cvtColor(dot_mask, cv2.COLOR_BGR2RGB)
        dot_mask = Image.fromarray(dot_mask)

        density_map = Image.fromarray(density_map.astype(np.float32), mode='F')
        density_map = transforms.ToTensor()(density_map)
        # --- Apply geometric transforms manually ---
        if self.mode == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                density_map = TF.hflip(density_map)

            if random.random() > 0.5:
                image = TF.vflip(image)
                density_map = TF.vflip(density_map)

            if self.qt:
                shuffler = QuarterShuffle(p=0.5)
                image = shuffler.shuffle(image)
                density_map = shuffler.shuffle(density_map)


        # --- Convert ---
        image = self.img_transform(image)
        true_count = density_map.sum()
        #print(f"[Sample {idx}] Dot count: {int(true_count)}")

        return image, transforms.ToTensor()(dot_mask), density_map

    def _generate_density_map(self, dot_image, sigma=2):
        # Ensure it's in RGB (OpenCV loads BGR by default)
        dot_image = cv2.cvtColor(dot_image, cv2.COLOR_BGR2RGB)

        # Threshold for "red" pixels
        red_mask = (dot_image[:, :, 0] > 150) & (dot_image[:, :, 1] < 80) & (dot_image[:, :, 2] < 80)

        # Build density map
        density = np.zeros(red_mask.shape, dtype=np.float32)
        density[red_mask] = 1.0
        density = gaussian_filter(density, sigma=sigma)

        count = red_mask.sum()
        if density.sum() > 0:
            density *= count / density.sum()
        #print(f"Detected dots: {red_mask.sum()}, Density map sum after normalization: {density.sum()}")

        return density

    def _get_img_transform(self):
        if self.mode == 'train':
            ops = []
            if self.gt:
                ops.extend([
                    transforms.ColorJitter(...),
                    transforms.RandomApply([...]),
                    transforms.RandomAutocontrast(),
                ])
            ops.append(transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225]))
            return transforms.Compose(ops)
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])



    def get_all_density_maps(self):
        maps = []
        for i in range(len(self)):
            _, _, density = self[i]
            maps.append(density)
        return torch.stack(maps)

    def get_all_counts(self):
        maps = self.get_all_density_maps()
        return maps.sum(dim=(1, 2, 3))  # Returns [N] count per image

    def get_loader(self, batch_size=8, shuffle=True, num_workers=2):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def visualize_sample(self, idx):
        """
        Visualize image, raw label (dot mask), and generated density map for one sample.
        """
        image, dot_mask, density_map = self[idx]

        # Convert image: [3, H, W] → [H, W, 3] and denormalize
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        # Convert dot_mask to NumPy
        dot_mask_np = dot_mask.permute(1, 2, 0).cpu().numpy()

        # Convert density map: [1, H, W] → [H, W]
        density_np = density_map.squeeze().cpu().numpy()

        # Plot all 3
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(image_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(dot_mask_np)
        axes[1].set_title("Raw Label (Dot Mask)")
        axes[1].axis("off")

        im = axes[2].imshow(density_np, cmap='jet')
        axes[2].set_title(f"Density Map (Sum: {density_np.sum():.1f})")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2])
        fig.savefig(f"sample_{idx}.png")
        plt.tight_layout()
        plt.show()


# In[7]:


train_set = CellCountingDataset(train_image_paths, train_label_paths, mode='train')
val_set = CellCountingDataset(val_image_paths, val_label_paths, mode='val')

train_loader = train_set.get_loader(batch_size=8)
val_loader = val_set.get_loader(batch_size=8)

#train_density_maps = train_set.get_all_density_maps()
val_counts = val_set.get_all_counts()


# In[10]:


val_set.visualize_sample(0)


# In[11]:


train_set.visualize_sample(0)


# In[15]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) × 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, alg=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if alg:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure x1 and x2 have same dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetCellCounter(nn.Module):
    def __init__(self, n_channels=3, alg=False):
        super(UNetCellCounter, self).__init__()
        self.n_channels = n_channels
        self.bilinear = alg

        # Initial feature size and scaling factor
        features = 64

        # Encoder (Downsampling path)
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)       # 64 -> 128
        self.down2 = Down(features * 2, features * 4)   # 128 -> 256
        self.down3 = Down(features * 4, features * 8)   # 256 -> 512

        # Bottleneck
        factor = 2 if alg else 1
        self.down4 = Down(features * 8, features * 16 // factor)  # 512 -> 1024/2=512

        # Decoder (Upsampling path)
        self.up1 = Up(features * 16, features * 8 // factor, alg)  # 1024 -> 512/2=256
        self.up2 = Up(features * 8, features * 4 // factor, alg)   # 512 -> 256/2=128
        self.up3 = Up(features * 4, features * 2 // factor, alg)   # 256 -> 128/2=64
        self.up4 = Up(features * 2, features, alg)                 # 128 -> 64

        # Final output layer - density map
        self.outc = OutConv(features, 1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)        # 64 channels
        x2 = self.down1(x1)     # 128 channels
        x3 = self.down2(x2)     # 256 channels
        x4 = self.down3(x3)     # 512 channels
        x5 = self.down4(x4)     # 1024/512 channels (bottleneck)

        # Decoder path with skip connections
        x = self.up1(x5, x4)    # 512/256 channels
        x = self.up2(x, x3)     # 256/128 channels
        x = self.up3(x, x2)     # 128/64 channels
        x = self.up4(x, x1)     # 64 channels

        # Final density map
        density_map = self.outc(x)  # 1 channel

        # Calculate cell count by summing the density map
        count = torch.sum(density_map, dim=(1, 2, 3))

        return density_map, count
def mse_with_count_regularization(pred_map, target_map, alpha=1, beta=0.001):
    pixel_mse = F.mse_loss(pred_map, target_map)

    pred_count = torch.sum(pred_map, dim=(1, 2, 3))
    true_count = torch.sum(target_map, dim=(1, 2, 3))
    count_mse = F.mse_loss(pred_count, true_count)

    return alpha * pixel_mse + beta * count_mse


# Training function with L2 loss
def train_unet(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=50):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

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
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, MAE: {avg_mae:.2f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_unet_cell_counter.pth')
            print(f'Model saved with Val Loss: {best_val_loss:.4f}')

    return train_losses, val_losses


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter(n_channels=3, alg=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()  # L2 loss for density map regression


# In[10]:


train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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



# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression


# In[15]:


from torchsummary import summary
summary(model, input_size = (3,256,256))


# In[10]:


import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

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



# In[11]:


from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetCellCounterEffNet().to(device)

summary(model, input_size=(3, 256, 256), device=str(device))



# ## Expeiment 1: Adam, lr 1e-4, weight_decay 1e-4, sigma - 1, bicubic with pretrained weights resnet 34

# In[196]:


train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# ## Expeiment 1: Adam, lr 1e-4, weight_decay 1e-4, sigma - 0.5, bicubic with pretrained weights resnet 34

# In[211]:


train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# ## Expeiment 1: Adam, lr 1e-4, weight_decay 1e-4, sigma - 0.8, bicubic with pretrained weights resnet 34

# In[13]:


train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# Expeiment 4: Adam, lr 1e-4, weight_decay 1e-4, sigma - 1, bicubic with pretrained weights resnet 34, rotation to image and desity maps (same angle)

# In[43]:


train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# Expeiment 4: Adam, lr 1e-4, weight_decay 1e-4, sigma - 1, bicubic with pretrained weights resnet 34, rotation to image and desity maps (same angle)

# In[60]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# In[61]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# In[63]:


import random
import matplotlib.pyplot as plt

model.eval()

# Get 5 random indices from the val set
random_indices = random.sample(range(len(val_set)), 5)

with torch.no_grad():
    for idx in random_indices:
        image, _, gt_map = val_set[idx]

        # Move to device and batchify
        image_batch = image.unsqueeze(0).to(device)
        gt_map = gt_map.unsqueeze(0).to(device)

        # Forward pass
        pred_map, pred_count = model(image_batch)
        true_count = gt_map.sum().item()

        # Convert for plotting
        input_img = image.permute(1, 2, 0).numpy()
        input_img = np.clip(input_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        gt_map_np = gt_map.squeeze().cpu().numpy()
        pred_map_np = pred_map.squeeze().cpu().numpy()

        # Plotting
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(input_img)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_map_np, cmap='jet')
        plt.title(f"Ground Truth (Sum: {true_count:.1f})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_map_np, cmap='jet')
        plt.title(f"Prediction (Sum: {pred_count.item():.1f})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# # `batch size = 8`

# Expeiment 6: Adam, lr 1e-4, weight_decay 1e-4, sigma - 1.2, bicubic with pretrained weights resnet 34, rotation to image and desity maps (same angle)

# In[64]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# Expeiment 4: Adam, lr 1e-4, weight_decay 1e-4, sigma - 0.9, bicubic with pretrained weights resnet 34, rotation to image and desity maps (same angle) with 1/4 transformation

# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=35
)


# Expeiment 4: Adam, lr 1e-4, weight_decay 1e-4, sigma - 1, bicubic with pretrained weights resnet 34, rotation to image and desity maps (same angle) with 1/4 transformation

# In[28]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=30
)


# Expeiment 4: Adam, lr 1e-4, weight_decay 1e-4, sigma - 2, bicubic with pretrained weights resnet 34, rotation to image and desity maps (same angle) with 1/4 transformation

# In[31]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounter().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=35
)


# Expeiment 4: Adam, lr 1e-3, weight_decay 1e-4, sigma - 2, bicubic with pretrained weights efficientnet using smp, rotation to image and desity maps (same angle) with 1/4 transformation

# In[22]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounterEffNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=50
)


# Expeiment 4: Adam, lr 1e-3, weight_decay 5e-5, sigma - 2, bicubic with pretrained weights efficientnet using smp, rotation to image and desity maps (same angle) with 1/4 transformation

# In[23]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetCellCounterEffNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
criterion = nn.MSELoss()  # L2 loss for density map regression
train_losses, val_losses = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=mse_with_count_regularization,
    device=device,
    num_epochs=50
)

