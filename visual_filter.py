
import torch
from cnn import CnnDetector


def visualize_cnn_filters(model: CnnDetector):
    import matplotlib.pyplot as plt

    first_conv = model.features[0]  # предполагается, что первый слой - это Conv2d
    weights = first_conv.weight.data.cpu().numpy()  # (out_channels, in_channels, kH, kW)
    num_filters = weights.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i in range(num_filters):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        filt = weights[i, 0, :, :]  # assuming single channel input
        ax.imshow(filt, cmap='gray')
        ax.axis('off')
    plt.savefig('filters.png')

if __name__ == "__main__":
    model = CnnDetector(num_classes=2)
    model.load_state_dict(torch.load("checkpoints/cnn_epoch20.pt", map_location="cpu")["model"])
    visualize_cnn_filters(model)
