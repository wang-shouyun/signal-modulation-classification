import os
import numpy as np
from torch.utils.data import Subset

from dataset import RadioMLDataset

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

labels = []
for i in range(len(small_dataset)):
    _, y, _ = small_dataset[i]
    labels.append(int(y))

labels = np.array(labels)
unique, counts = np.unique(labels, return_counts=True)

print("128条小样本中的类别分布：")
for u, c in zip(unique, counts):
    print(f"类别 {u}: {c} 条")