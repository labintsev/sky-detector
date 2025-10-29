import os
import glob
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class YoloDataset(Dataset):
    """
    Ожидает:
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
        # вернём список коробок (может быть пустым)
        target = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0,5), dtype=torch.float32)
        return img_t, target, str(p.name)


class SimpleYolo(nn.Module):
    """
    Простая сверточная YOLO-подобная сеть.
    Выдаёт тензор S x S x (B*5 + C) в виде (batch, S, S, D)
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
        out_dim = self.S * self.S * (self.B * 5 + self.C)
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
        out = out.view(bs, self.S, self.S, self.B * 5 + self.C)
        return out


def encode_target(target_boxes: torch.Tensor, S: int, B: int, C: int, device):
    """
    target_boxes: (N, 5) [class, x, y, w, h] normalized coords for one image
    Возвращает тензор (S, S, B*5 + C) с заполнением:
     - Для первой B-ячейки записываем bbox+conf
     - Здесь упрощённый encoding: в каждой ячейке храним только одну GT (первую попавшуюся)
    """
    target = torch.zeros((S, S, B * 5 + C), dtype=torch.float32, device=device)
    cell_size = 1.0 / S
    for box in target_boxes:
        cls, x, y, w, h = box.tolist()
        if not (0 <= x <= 1 and 0 <= y <= 1):
            continue
        c_x = int(x / cell_size); c_y = int(y / cell_size)
        if c_x >= S: c_x = S - 1
        if c_y >= S: c_y = S - 1
        # relative coords inside cell
        rel_x = x * S - c_x
        rel_y = y * S - c_y
        # записываем только в первый B (упрощение)
        target[c_y, c_x, 0:5] = torch.tensor([rel_x, rel_y, w, h, 1.0], device=device)
        # class one-hot (C classes)
        if C > 0:
            cls_idx = int(cls)
            if 0 <= cls_idx < C:
                target[c_y, c_x, B * 5 + cls_idx] = 1.0
    return target


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1, lambda_coord=10, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, preds, targets):
        # preds: (batch, S, S, B*5 + C)
        # targets: (batch, S, S, B*5 + C)
        device = preds.device
        batch = preds.shape[0]
        # object mask: where target confidence == 1 (we use first box conf pos)
        obj_mask = targets[..., 4] > 0  # (batch, S, S)
        noobj_mask = ~obj_mask

        loss = 0.0
        # coord loss (x,y,w,h) for responsible boxes (we assume first box)
        pred_boxes = preds[..., 0:4]
        target_boxes = targets[..., 0:4]
        loss_coord = self.mse(pred_boxes[obj_mask], target_boxes[obj_mask]) if obj_mask.any() else torch.tensor(0.0, device=device)
        # confidence loss
        pred_conf = preds[..., 4]
        target_conf = targets[..., 4]
        loss_obj = self.mse(pred_conf[obj_mask], target_conf[obj_mask]) if obj_mask.any() else torch.tensor(0.0, device=device)
        loss_noobj = self.mse(pred_conf[noobj_mask], target_conf[noobj_mask]) if noobj_mask.any() else torch.tensor(0.0, device=device)
        # class loss
        if self.C > 0:
            pred_cls = preds[..., self.B*5:]
            target_cls = targets[..., self.B*5:]
            loss_cls = self.mse(pred_cls[obj_mask], target_cls[obj_mask]) if obj_mask.any() else torch.tensor(0.0, device=device)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        loss = self.lambda_coord * loss_coord + loss_obj + self.lambda_noobj * loss_noobj + loss_cls
        return loss / max(1, batch)


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    names = [b[2] for b in batch]
    return imgs, targets, names


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = YoloDataset(args.images, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    model = SimpleYolo(S=args.S, B=args.B, C=args.C).to(device)
    criterion = YoloLoss(S=args.S, B=args.B, C=args.C)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets_list, names in dl:
            imgs = imgs.to(device)
            preds = model(imgs)  # (batch, S, S, D)
            # encode targets per image into (S,S,D)
            batch_targets = torch.stack([encode_target(t, args.S, args.B, args.C, device) for t in targets_list], dim=0)
            loss = criterion(preds, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(dl)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg:.4f}")
        if epoch % args.save_every == 0:
            ckpt = Path(args.out_dir) / f"yolo_epoch{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
            print("Saved", ckpt)


def visualize_predictions(img: torch.Tensor, preds: torch.Tensor, S: int, B: int, C: int):
    """
    img: (3, H, W) tensor
    preds: (S, S, B*5 + C) tensor
    Отображает предсказанные боксы на изображении (для отладки)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img_np = img.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    cell_size = 1.0 / S
    for i in range(S):
        for j in range(S):
            for b in range(B):
                conf = preds[i, j, b * 5 + 4].item()
                if conf > 0.1:  # порог уверенности
                    x = (j + preds[i, j, b * 5 + 0].item()) * cell_size * img_np.shape[1]
                    y = (i + preds[i, j, b * 5 + 1].item()) * cell_size * img_np.shape[0]
                    w = preds[i, j, b * 5 + 2].item() * img_np.shape[1]
                    h = preds[i, j, b * 5 + 3].item() * img_np.shape[0]
                    print(f"Cell ({i},{j}) Box {b}  x: {x:.2f}  y: {y:.2f}  w: {w:.2f}  h: {h:.2f}, conf: {conf:.2f}")
                    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
    plt.show()


def test(checkpoint_path: str, images_dir: str, S: int, B: int, C: int, img_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleYolo(S=S, B=B, C=C).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Iterate over all images in the specified directory
    for img_name in os.listdir(images_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(img_t)[0]  # (S, S, D)
            print(f"Predictions for {img_name}:")
            visualize_predictions(img_t[0], preds, S, B, C)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="data/frames_ir/imgs", help="папка с изображениями")
    p.add_argument("--out-dir", type=str, default="checkpoints", help="куда сохранять модели")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--S", type=int, default=7)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--C", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--test", action="store_true", help="тестировать модель")
    args = p.parse_args()
    if args.test:
        # пример тестирования
        test_checkpoint = os.path.join(args.out_dir, "yolo_epoch30.pt")
        test(test_checkpoint, args.images, args.S, args.B, args.C, args.img_size)
    else:   
        os.makedirs(args.out_dir, exist_ok=True)
        train(args)
