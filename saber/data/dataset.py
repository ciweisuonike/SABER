import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional
import pandas as pd


class BulkRNASeqDataset(Dataset):
    def __init__(
        self,
        X_bulk: pd.DataFrame,
        X_clean_bulk: Optional[pd.DataFrame] = None,
        P_df: Optional[pd.DataFrame] = None,
        device: str = "cpu",
    ):
        self.device = device

        self.X = np.maximum(X_bulk.values.astype(np.float32), 0)
        self.X_tensor = torch.FloatTensor(self.X).to(device)
        self.num_samples, self.gene_dim = self.X.shape

        self.X_clean_tensor = None
        if X_clean_bulk is not None:
            if list(X_clean_bulk.columns) != list(X_bulk.columns):
                raise ValueError("X_clean_bulk and X_bulk must have identical gene columns.")
            if len(X_clean_bulk) != self.num_samples:
                raise ValueError("X_clean_bulk and X_bulk must have identical sample counts.")
            self.X_clean = np.maximum(X_clean_bulk.values.astype(np.float32), 0)
            self.X_clean_tensor = torch.FloatTensor(self.X_clean).to(device)

        if P_df is not None:
            self.P = np.maximum(P_df.values.astype(np.float32), 0)
            self.num_cell_types = self.P.shape[1]
            self.celltype_names = list(P_df.columns)
        else:
            self.num_cell_types = 1
            self.celltype_names = ["Unknown"]
            self.P = np.zeros((self.num_samples, self.num_cell_types), dtype=np.float32)

        self.P_tensor = torch.FloatTensor(self.P).to(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.X_tensor[idx]
        proportions = self.P_tensor[idx]
        if self.X_clean_tensor is not None:
            x_clean = self.X_clean_tensor[idx]
            return x_clean, x, proportions
        return x, proportions
