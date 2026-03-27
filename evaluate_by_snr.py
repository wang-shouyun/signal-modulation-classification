import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt

from dataset import RadioMLDataset
from models.cnn_model import CNNModel


def evaluate_by_snr():
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = os.path.join("data", "RML2016.10a_dict.pkl")

    selected_mods = ["BPSK", "QPSK", "8PSK", "QAM16"]
    selected_snrs = [10, 12, 14, 16, 18]

    dataset = RadioMLDataset(
        data_path=data_path,
        selected_mods=selected_mods,
        selected_snrs=selected_snrs
    )

    np.random.seed(42)
    sample_size = min(8000, len(dataset))
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    dataset = Subset(dataset, indices)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CNNModel(num_classes=len(selected_mods)).to(device)
    model.load_state_dict(torch.load("results/best_cnn_model.pth", map_location=device))
    model.eval()

    snr_correct = {}
    snr_total = {}

    with torch.no_grad():
        for x, y, snr in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            preds = preds.cpu().numpy()
            labels = y.cpu().numpy()
            snrs = snr.cpu().numpy()

            for pred, label, snr_value in zip(preds, labels, snrs):
                if snr_value not in snr_correct:
                    snr_correct[snr_value] = 0
                    snr_total[snr_value] = 0

                if pred == label:
                    snr_correct[snr_value] += 1
                snr_total[snr_value] += 1

    snr_list = sorted(snr_total.keys())
    acc_list = []

    print("\nAccuracy by SNR:")
    for snr_value in snr_list:
        acc = snr_correct[snr_value] / snr_total[snr_value]
        acc_list.append(acc)
        print(f"SNR = {snr_value:>2} dB : Accuracy = {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(snr_list, acc_list, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR")
    plt.grid(True)
    plt.savefig("results/accuracy_vs_snr.png")
    plt.close()

    print("\nFigure saved to results/accuracy_vs_snr.png")


if __name__ == "__main__":
    evaluate_by_snr()