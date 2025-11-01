def visualize_loss_curve(log_file: str):
    """Visualize the training loss curve from the log file.
    Line example in log file:
    Epoch 1/100  loss=0.6953
    """
    import matplotlib.pyplot as plt

    epochs = []
    losses = []

    with open(log_file, "r") as f:
        for line in f:
            if "loss=" in line:
                parts = line.strip().split()
                epoch_part = parts[1]  # e.g., "1/100"
                loss_part = parts[2]   # e.g., "loss=0.6953"
                epoch = int(epoch_part.split('/')[0])
                loss = float(loss_part.split('=')[1])
                epochs.append(epoch)
                losses.append(loss)

    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig('data/article/cnn/loss_curve.png')


if __name__ == "__main__":
    visualize_loss_curve("data/article/cnn/train.log")
