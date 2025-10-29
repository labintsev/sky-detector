from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class YoloDataset(Dataset):
    """
    Набор данных для обучения модели:
      - images в image_dir (jpg/png)
      - для каждого image.jpg должен быть image.txt с аннотациями YOLO: class x_center y_center w h (норм.)
    """
    def __init__(self, image_dir: str, img_size=448, transform=None):
        self.image_dir = Path(image_dir)
        self.images = sorted([p for p in self.image_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def _read_labels(self, img_path: Path):
        label_path = img_path.with_suffix(".txt")
        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                        boxes.append([cls, x, y, w, h])
        return boxes

    def __getitem__(self, idx):
        p = self.images[idx]
        img = Image.open(p).convert("RGB")
        img_t = self.transform(img)
        labels = self._read_labels(p)
        # вернём список боксов (может быть пустым)
        target = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0,5), dtype=torch.float32)
        return img_t, target, str(p.name)
 

class SimpleYolo(nn.Module):
    """
    Простая сверточная YOLO-подобная сеть.
    Выдаёт тензор batch x S x S x (B * (4 + C)) где:
     S - размер сетки (SxS ячеек)
     B - количество боксов на ячейку 
     C - количество классов
     4 - (x_center, y_center, w, h), нормализованные координаты в пределах ячейки
    """
    def __init__(self, S=7, B=2, C=1):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        # простая сверточная часть
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((self.S, self.S)),
        )
        out_dim = self.S * self.S * (self.B * (4 + self.C))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, out_dim)
        )

    def forward(self, x):
        bs = x.shape[0]
        f = self.features(x)
        out = self.head(f)
        out = out.view(bs, self.S, self.S, self.B * (4 + self.C))
        return out


class YoloLoss(nn.Module):
    """ 
    Функция потерь Loss = coordinates_loss +  classification_cross_entropy_loss
    """
    def __init__(self, S, B, C, lambda_coord=2, lambda_ce=1):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_ce = lambda_ce
        self.mse = nn.MSELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, preds, targets):
        device = preds.device
        batch = preds.shape[0]
        loss = 0.0
        for b in range(batch):  
            pred = preds[b]  # (S, S, D)
            target = targets[b]  # (S, S, D)
            for i in range(self.S):
                for j in range(self.S):
                    for bb in range(self.B):
                        p_offset = bb * (4 + self.C)
                        t_offset = bb * (4 + self.C)
                        pred_box = pred[i, j, p_offset:p_offset+4]
                        target_box = target[i, j, t_offset:t_offset+4]
                        # координаты
                        loss += self.lambda_coord * self.mse(pred_box, target_box)
                    # классификация (по первому боксу)
                    if self.C > 0:
                        pred_cls = pred[i, j, self.B * 4:self.B * 4 + self.C]
                        target_cls = target[i, j, self.B * 4:self.B * 4 + self.C]
                        if target_cls.sum() > 0:
                            target_label = torch.argmax(target_cls).unsqueeze(0).to(device)
                            loss += self.lambda_ce * self.ce(pred_cls.unsqueeze(0), target_label)
        
        return loss / batch
