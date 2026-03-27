import os
import numpy as np
from torch.utils.data import Subset

from dataset import RadioMLDataset

data_path = os.path.join("data", "RML2016.10a_dict.pkl")
dataset = RadioMLDataset(data_path=data_path)

subset = Subset(dataset, range(10000))

labels = []
for i in range(len(subset)):
    _, y = subset[i]
    labels.append(int(y))

labels = np.array(labels)
unique, counts = np.unique(labels, return_counts=True)

print("前10000条样本的类别分布：")
for u, c in zip(unique, counts):
    print(f"类别 {u}: {c} 条")