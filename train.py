import os
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cnn import CnnDataset, CnnDetector, CnnLoss


def train_cnn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CnnDataset(args.images, img_size=args.img_size, grid_size=args.S, num_classes=args.C)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    model = CnnDetector(num_classes=args.C).to(device)
    criterion = CnnLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets_list, names in dl:
            imgs = imgs.to(device)
            targets = targets_list.to(device)  # (batch, S, S, C)
            batch_size = imgs.size(0)
            preds = model(imgs)  # (batch, H', W', C)
            preds = preds.view(batch_size, args.S, args.S, args.C)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(dl)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg:.4f}")
        if epoch % args.save_every == 0:
            ckpt = Path(args.out_dir) / f"cnn_epoch{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
            print("Saved", ckpt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="data/frames_ir", help="папка с изображениями")
    p.add_argument("--out-dir", type=str, default="checkpoints", help="куда сохранять модели")
    p.add_argument("--epochs", type=int, default=20, help="количество эпох")
    p.add_argument("--batch", type=int, default=8, help="размер батча")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--img-size", type=int, default=512, help="размер входного изображения")
    p.add_argument("--S", type=int, default=64, help="размер сетки SxS")
    p.add_argument("--B", type=int, default=1, help="количество боксов на ячейку")
    p.add_argument("--C", type=int, default=2, help="количество классов")
    p.add_argument("--save-every", type=int, default=10, help="сохранять модель каждые N эпох")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_cnn(args)
