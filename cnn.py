import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class GridDataset(Dataset):
    """
    Набор данных Yolo формата для обучения модели CNN detector:
      - data_dir содержит две папки: images и labels
      - images - изображения jpg/png
      - labels - для каждого image.jpg должен быть image.txt с аннотациями: [class_id x_min y_min w h] (нормированные координаты)
    
    Возвращает тензор изображения и тензор - сетку с метками (grid_size, grid_size, num_classes)
    """
    def __init__(self, data_dir: str, img_size: int, grid_size: int, num_classes: int, transform=None):
        self.data_dir = Path(data_dir)
        self.image_dir = Path(os.path.join(self.data_dir, "images"))
        self.label_dir = Path(os.path.join(self.data_dir, "labels"))
        self.images = sorted([p for p in self.image_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[1.0])
        ])

    def __len__(self):
        return len(self.images)

    def _read_labels(self, img_path: Path):
        """Read bounding box labels from a text file.
        returns tensor of shape (grid_size, grid_size, num_classes) with class probabilities for each cell
        """
        label_file_name = img_path.stem + ".txt"
        label_path = os.path.join(self.label_dir, label_file_name)

        if not os.path.exists(label_path):
            print
            return torch.zeros((self.grid_size, self.grid_size, self.num_classes), dtype=torch.float32)

        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_ = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    x_min = x_center - float(parts[3]) / 2
                    y_min = y_center - float(parts[4]) / 2
                    x_max = x_center + float(parts[3]) / 2
                    y_max = y_center + float(parts[4]) / 2
                    labels.append([cls_, x_min, y_min, x_max, y_max])

        # Create a grid of class probabilities
        grid = torch.zeros((self.grid_size, self.grid_size, self.num_classes), dtype=torch.float32)
        for cls_, x_min, y_min, x_max, y_max in labels:
            # Map bounding box coordinates to grid cells
            x_min_cell = int(round(x_min * self.grid_size))
            y_min_cell = int(round(y_min * self.grid_size))
            x_max_cell = int(round(x_max * self.grid_size))
            y_max_cell = int(round(y_max * self.grid_size))
            grid[y_min_cell:y_max_cell, x_min_cell:x_max_cell, cls_] = 1.0

        return grid

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L")
        img_t = self.transform(img)
        labels = self._read_labels(img_path)
        return img_t, labels, str(img_path.name)


class CnnDetector(nn.Module):
    """
    Простая CNN для извлечения признаков и классификации/регрессии каждой клетки в изображении.
    """
    def __init__(self, grid_size=64, num_classes=2):
        self.num_classes = num_classes
        self.grid_size = grid_size
        super(CnnDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=8, padding=0),  
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # extract features
        x = self.features(x)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, C)
        # classify
        x = self.classifier(x)
        return x


class VggDetector(nn.Module):   
    """
    VggDetector based detector, 
    output shape: (batch_size, grid_H, grid_W, num_classes)
    """
    def __init__(self, grid_size=64, num_classes=2):
        super(VggDetector, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Define a simple MobileNetV2-like architecture
        self.vggnet = nn.Sequential(
            self.convnet_block(1, 8, stride=2),   # 256x256
            self.convnet_block(8, 16, stride=2),  # 128x128
            self.convnet_block(16, 32, stride=2), # 64x64 final layer
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def convnet_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.vggnet(x)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, num_classes)
        x = self.classifier(x)
        return x


class CnnLoss(nn.Module):
    """
    Простой Cross Entropy Loss для классификации
    """
    def __init__(self):
        super(CnnLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


def test_random_input():
    print("Testing CNN Detector with random input...")
    img = torch.randn((1, 1, 512, 512))
    model = VggDetector(num_classes=2)
    out = model(img)
    print(out.shape)                            # ожидается (1, H', W', num_classes)
    loss_fn = CnnLoss()
    targets = torch.tensor([64*[ 64*[[0,1]]]], dtype=torch.float)  # пример целей
    loss = loss_fn(out, targets)
    print(loss.item())
    print("Test completed.")


def test_dataloader():
    print("Testing RCNN Dataset and DataLoader...")
    dataset = GridDataset(data_dir="data/frames_ir", img_size=512, grid_size=64, num_classes=2)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for imgs, labels, names in dataloader:
        print(f"Batch images shape: {imgs.shape}")  # ожидается (batch_size, 3, 512, 512)
        print(f"Batch labels shape: {labels.shape}")  # ожидается (batch_size, grid_size, grid_size, num_classes)
        print('Sum of labels in first batch element:', sum(labels[2].view(-1)).item())  # пример подсчёта меток в первом элементе батча
        break
    print("DataLoader test completed.")


if __name__ == "__main__":
    test_random_input()
    test_dataloader()
