import argparse
import os
import torch
from torchvision import transforms

from cnn import CnnDetector, CnnDataset


def visualize_cnn_predictions(img: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor, img_name: str):
    """ img: (3, H, W) tensor
        predictions: (grid_size, grid_size, num_classes) tensor
        """
    import matplotlib.pyplot as plt
    import numpy as np

    img_np = img.permute(1, 2, 0).cpu().numpy()
    # Visualization of predictions and ground truth boxes
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_np)
    ax[0].set_title("CNN Predictions")
    ax[1].imshow(img_np)
    ax[1].set_title(f"Ground Truth {img_name}")

    grid_size = predictions.shape[0]
    cell_h = img_np.shape[0] / grid_size
    cell_w = img_np.shape[1] / grid_size
    
    # Draw predicted boxes
    for i in range(grid_size):
        for j in range(grid_size):
            for cls_ in range(predictions.shape[2]):
                prob = predictions[i, j, cls_].item()
                if prob > 0.5:  # порог вероятности
                    x = j * cell_w
                    y = i * cell_h
                    color = 'blue' if cls_ == 0 else 'red'
                    rect = plt.Rectangle((x-cell_w/2, y-cell_h/2), cell_w, cell_h, linewidth=2, edgecolor=color, facecolor='none')
                    ax[0].add_patch(rect)
    # Draw ground truth boxes
    for i in range(grid_size):
        for j in range(grid_size):
            for cls_ in range(labels.shape[2]):
                prob = labels[i, j, cls_].item()
                if prob > 0.5:
                    x = j * cell_w
                    y = i * cell_h
                    color = 'blue' if cls_ == 0 else 'red'
                    rect = plt.Rectangle((x-cell_w/2, y-cell_h/2), cell_w, cell_h, linewidth=2, edgecolor=color, facecolor='none')
                    ax[1].add_patch(rect)
    
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

    dataset = CnnDataset(images_dir, img_size=img_size, grid_size=64, num_classes=2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    precision_list, recall_list = [], []

    for img_t, labels, img_name in dataloader:
        img_t = img_t.to(device)
        with torch.no_grad():
            preds = model(img_t)
        # compute precision and recall here
        preds = preds[0].cpu()
        labels = labels[0].cpu()
        threshold = 0.5
        tp = ((preds > threshold) & (labels > threshold)).sum().item()
        fp = ((preds > threshold) & (labels <= threshold)).sum().item()
        fn = ((preds <= threshold) & (labels > threshold)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)

        # visualize_cnn_predictions(img_t[0], preds, labels, img_name[0])

    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    print(f"Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}")


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
    test_checkpoint = os.path.join(args.out_dir, "cnn_epoch100.pt")
    test_cnn(test_checkpoint, args.images, args.img_size)