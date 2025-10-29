import os
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from yolo import YoloDataset, SimpleYolo, YoloLoss


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    names = [b[2] for b in batch]
    return imgs, targets, names


def encode_target(target_boxes: torch.Tensor, S: int, B: int, C: int, device) -> torch.Tensor:
    """
    Кодирует размеченные боксы в формат, подходящий для обучения модели.
    Переводит список боксов в тензор (S, S, B*(4 + C))
    target_boxes: (N, 5) [class, x, y, w, h] размеченные боксы для одного изображения
    S: размер сетки
    Возвращает тензор (S, S, B*(4 + C)) 
    """
    target = torch.zeros((S, S, B * (4 + C)), dtype=torch.float32, device=device)
    cell_size = 1.0 / S
    for box in target_boxes:
        cls_, x, y, w, h = box.tolist()
        if not (0 <= x <= 1 and 0 <= y <= 1):
            continue
        c_x = int(x / cell_size); c_y = int(y / cell_size)
        if c_x >= S: c_x = S - 1
        if c_y >= S: c_y = S - 1
        # relative coords inside cell
        rel_x = x * S - c_x
        rel_y = y * S - c_y
        # записываем во все боксы B
        for b in range(B):
            target[c_y, c_x, b * (4 + C) + 0] = rel_x
            target[c_y, c_x, b * (4 + C) + 1] = rel_y
            target[c_y, c_x, b * (4 + C) + 2] = w
            target[c_y, c_x, b * (4 + C) + 3] = h
            # class one-hot (C classes)
            if C > 0:
                target[c_y, c_x, b * (4 + C) + 4 + int(cls_)] = 1.0
    return target


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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="data/frames_ir", help="папка с изображениями")
    p.add_argument("--out-dir", type=str, default="checkpoints", help="куда сохранять модели")
    p.add_argument("--epochs", type=int, default=20, help="количество эпох")
    p.add_argument("--batch", type=int, default=8, help="размер батча")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--img-size", type=int, default=512, help="размер входного изображения")
    p.add_argument("--S", type=int, default=8, help="размер сетки SxS")
    p.add_argument("--B", type=int, default=1, help="количество боксов на ячейку")
    p.add_argument("--C", type=int, default=2, help="количество классов")
    p.add_argument("--save-every", type=int, default=10, help="сохранять модель каждые N эпох")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
