import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class RadioMLDataset(Dataset):
    def __init__(self, data_path, selected_mods=None, selected_snrs=None):
        with open(data_path, "rb") as f:
            Xd = pickle.load(f, encoding="latin1")

        all_mods = sorted(list(set([key[0] for key in Xd.keys()])))
        all_snrs = sorted(list(set([key[1] for key in Xd.keys()])))

        if selected_mods is None:
            selected_mods = all_mods
        if selected_snrs is None:
            selected_snrs = all_snrs

        self.mods = selected_mods
        self.snrs = selected_snrs
        self.mod_to_idx = {mod: i for i, mod in enumerate(self.mods)}
        self.idx_to_mod = {i: mod for mod, i in self.mod_to_idx.items()}

        X_list = []
        y_list = []
        snr_list = []

        for mod in self.mods:
            for snr in self.snrs:
                if (mod, snr) in Xd:
                    samples = Xd[(mod, snr)]
                    labels = [self.mod_to_idx[mod]] * len(samples)
                    snrs_this_block = [snr] * len(samples)

                    X_list.append(samples)
                    y_list.extend(labels)
                    snr_list.extend(snrs_this_block)

        self.X = np.vstack(X_list).astype(np.float32)
        self.y = np.array(y_list, dtype=np.int64)
        self.snr = np.array(snr_list, dtype=np.int64)

        print("数据加载完成")
        print("X shape:", self.X.shape)
        print("类别数:", len(self.mod_to_idx))
        print("使用调制:", self.mods)
        print("使用SNR:", self.snrs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        power = np.mean(x ** 2)
        x = x / np.sqrt(power + 1e-8)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        snr = torch.tensor(self.snr[idx], dtype=torch.long)
        return x, y, snr