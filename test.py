import argparse
import os
from PIL import Image
import torch
from torchvision import transforms

from yolo import SimpleYolo
from cnn import CnnDetector

def visualize_yolo_predictions(img: torch.Tensor, preds: torch.Tensor, S: int, B: int, C: int):
    """
    img: (3, H, W) tensor
    preds: (S, S, B*(4 + C)) tensor
    B: количество боксов на ячейку [x_center, y_center, w, h, ]
    Отображает предсказанные боксы на изображении (для отладки)
    Схема бокса patches.Rectangle
                +------------------+
                |                  |
              height               |
                |                  |
               (xy)---- width -----+
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
                if conf > 0.7:  # порог уверенности
                    x = (j + preds[i, j, b * 5 + 0].item()) * cell_size * img_np.shape[1] 
                    y = (i + preds[i, j, b * 5 + 1].item()) * cell_size * img_np.shape[0] 
                    w = preds[i, j, b * 5 + 2].item() * img_np.shape[1]
                    h = preds[i, j, b * 5 + 3].item() * img_np.shape[0]

                    print(f"Cell ({i},{j}) Box {b}  x: {x:.2f}  y: {y:.2f}  w: {w:.2f}  h: {h:.2f}, conf: {conf:.2f}")
                    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')    
    plt.show()


def test_yolo(checkpoint_path: str, images_dir: str, S: int, B: int, C: int, img_size: int):
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
            visualize_yolo_predictions(img_t[0], preds, S, B, C)



def visualize_cnn_predictions(img: torch.Tensor, predictions: torch.Tensor):
    """ img: (3, H, W) tensor
        predictions: (grid_size, grid_size, num_classes) tensor
        """
    import matplotlib.pyplot as plt
    import numpy as np

    img_np = img.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    grid_size = predictions.shape[0]
    cell_h = img_np.shape[0] / grid_size
    cell_w = img_np.shape[1] / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            for cls_ in range(predictions.shape[2]):
                prob = predictions[i, j, cls_].item()
                if prob > 0.5:  # порог вероятности
                    x = j * cell_w
                    y = i * cell_h
                    color = 'blue' if cls_ == 0 else 'red'
                    rect = plt.Rectangle((x, y), cell_w, cell_h, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')    
    plt.show()


def test_cnn(checkpoint_path: str, images_dir: str, img_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnDetector().to(device)
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
                preds = model(img_t)

            print(f" Image shape: {img_t.shape} Predictions shape: {preds.shape}")
            visualize_cnn_predictions(img_t[0], preds[0].cpu())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="data/frames_ir", help="папка с изображениями")
    p.add_argument("--out-dir", type=str, default="checkpoints", help="куда сохранять модели")
    p.add_argument("--img-size", type=int, default=512, help="размер входного изображения")
    p.add_argument("--S", type=int, default=16, help="размер сетки SxS")
    p.add_argument("--B", type=int, default=1, help="количество боксов на ячейку")
    p.add_argument("--C", type=int, default=2, help="количество классов")
    args = p.parse_args()

    # пример тестирования
    test_checkpoint = os.path.join(args.out_dir, "cnn_epoch20.pt")
    test_cnn(test_checkpoint, args.images, args.img_size)