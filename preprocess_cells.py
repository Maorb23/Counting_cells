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
import argparse
import os

import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
class QuarterShuffle:
    def __init__(self, p=0.5):
        self.p = p

    def shuffle(self, img):
        
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
"""
class QuarterShuffle:
    def __init__(self, p=0.5):
        self.p = p

    def shuffle(self, img, idx=None):
        """
        Shuffle quarters of the image (optionally using a fixed idx).
        """
        if random.random() > self.p and idx is None:
            if isinstance(img, Image.Image):
                img = TF.to_tensor(img)
            return img, [0,1,2,3]  # Always return a tuple (tensor, identity_idx)


        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)

        C, H, W = img.shape
        h_half, w_half = H // 2, W // 2

        UL = img[:, :h_half, :w_half].clone()
        UR = img[:, :h_half, w_half:].clone()
        LL = img[:, h_half:, :w_half].clone()
        LR = img[:, h_half:, w_half:].clone()

        quarters = [UL, UR, LL, LR]

        if idx is None:
            if random.random() > self.p:
                # No shuffle, return identity
                idx = [0, 1, 2, 3]
            else:
                idx = list(range(4))
                random.shuffle(idx)

        top = torch.cat([quarters[idx[0]], quarters[idx[1]]], dim=2)
        bottom = torch.cat([quarters[idx[2]], quarters[idx[3]]], dim=2)
        shuffled = torch.cat([top, bottom], dim=1)

        return shuffled, idx


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
        #print(f"[DEBUG] Loading image: {self.image_paths[idx]}")
        #print(f"[DEBUG] Image shape: {image.shape}")
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
                """
                shuffler = QuarterShuffle(p=0.5)
                image = shuffler.shuffle(image)
                density_map = shuffler.shuffle(density_map)
                """
                shuffler = QuarterShuffle(p=0.5)
                image, idx = shuffler.shuffle(image)
                density_map, _ = shuffler.shuffle(density_map, idx=idx)


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
            if not self.qt:
                ops = [transforms.ToTensor()]
            if self.gt:
                ops.extend([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                          num_workers=num_workers, persistent_workers=True)

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
        fig.savefig(f"sample.png")
        plt.tight_layout()
        plt.show()
        return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cell Counting Dataset')
    parser.add_argument('--train_images', type=str, default='data/train_images', help='Path to the train imagse folder')
    parser.add_argument('--train_labels', type=str, default='data/train_labels', help='Path to the train labels folder')
    parser.add_argument('--val_images', type=str, default='data/val_images', help='Path to the val images folder')
    parser.add_argument('--val_labels', type=str, default='data/val_labels', help='Path to the val labels folder')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for DataLoader')


    args = parser.parse_args()
    train_image_paths = [os.path.join(args.train_images, f) for f in os.listdir(args.train_images) if f.endswith('.png')]
    train_label_paths = [os.path.join(args.train_labels, f) for f in os.listdir(args.train_labels) if f.endswith('.png')]
    val_image_paths = [os.path.join(args.val_images, f) for f in os.listdir(args.val_images) if f.endswith('.png')]
    val_label_paths = [os.path.join(args.val_labels, f) for f in os.listdir(args.val_labels) if f.endswith('.png')]


    train_dataset = CellCountingDataset(train_image_paths, train_label_paths, mode='train')
    val_dataset = CellCountingDataset(val_image_paths, val_label_paths, mode='val')
    train_loader = train_dataset.get_loader(batch_size=args.batch_size, shuffle=True)
    val_loader = val_dataset.get_loader(batch_size=args.batch_size, shuffle=False)
    # save train val loaders

    torch.save(train_loader, 'data/processed/train_loader.pth')
    torch.save(val_loader, 'data/processed/val_loader.pth')
    # Visualize a sample from the training dataset
    train_dataset.visualize_sample(1)
