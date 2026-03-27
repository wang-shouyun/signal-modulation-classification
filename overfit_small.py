import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from dataset import RadioMLDataset
from models.cnn_model import CNNModel


def main():
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
    indices = np.random.choice(len(dataset), size=128, replace=False)
    small_dataset = Subset(dataset, indices)

    train_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)

    model = CNNModel(num_classes=len(selected_mods)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y, snr in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f}"
        )


if __name__ == "__main__":
    main()