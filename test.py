import torch
from torch.utils.data import DataLoader

from dataset import RandomSignalDataset
from models.cnn_model import CNNModel


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 5
    dataset = RandomSignalDataset(num_samples=200, signal_length=128, num_classes=num_classes)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNNModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("results/cnn_model.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    test()