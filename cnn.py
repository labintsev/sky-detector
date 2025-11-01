from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CnnDataset(Dataset):
    """
    Набор данных для обучения модели CNN detector:
      - images в image_dir (jpg/png)
      - для каждого image.jpg должен быть image.txt с аннотациями: class x_min y_min w, h (нормированные координаты)
    """
    def __init__(self, image_dir: str, img_size: int, grid_size: int, num_classes: int, transform=None):
        self.image_dir = Path(image_dir)
        self.images = sorted([p for p in self.image_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def _read_labels(self, img_path: Path):
        """Read bounding box labels from a text file.
        returns tensor of shape (grid_size, grid_size, num_classes) with class probabilities for each cell
        """
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
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
        p = self.images[idx]
        img = Image.open(p).convert("L")
        img_t = self.transform(img)
        labels = self._read_labels(p)
        return img_t, labels, str(p.name)


class CnnDetector(nn.Module):
    """
    Простая CNN для извлечения признаков и классификации/регрессии каждой клетки в изображении.
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        super(CnnDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=8, padding=0),  
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # extract features
        x = self.features(x)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, C)
        # classify
        x = self.classifier(x)
        return x


class CnnLoss(nn.Module):
    """
    Простой Cross Entropy Loss для классификации
    """
    def __init__(self):
        super(CnnLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


def test_random_input():
    print("Testing CNN Detector with random input...")
    img = torch.randn((1, 1, 512, 512))
    model = CnnDetector(num_classes=2)
    out = model(img)
    print(out.shape)  # ожидается (1, H', W', num_classes)
    loss_fn = CnnLoss()
    targets = torch.tensor([64*[ 64*[[0,1]]]], dtype=torch.float)  # пример целей
    loss = loss_fn(out, targets)
    print(loss.item())
    print("Test completed.")


def test_dataloader():
    print("Testing RCNN Dataset and DataLoader...")
    dataset = CnnDataset(image_dir="data/frames_ir", img_size=512, grid_size=64, num_classes=2)
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
