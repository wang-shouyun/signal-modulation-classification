import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from dataset import RadioMLDataset
from models.cnn_model import CNNModel


def test():
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

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y, snr in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = np.mean(all_labels == all_preds)
    print(f"Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=selected_mods, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=selected_mods)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    test()