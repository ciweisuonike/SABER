import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional
import pandas as pd
from .dataset import BulkRNASeqDataset


class MultiStageDataLoader:
    def __init__(
        self,
        X_source: pd.DataFrame,
        P_source: pd.DataFrame,
        X_target: pd.DataFrame,
        X_source_clean: Optional[pd.DataFrame] = None,
        batch_size: int = 128,
        train_ratio: float = 0.8,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        P_source_clean = P_source.copy()
        P_source_clean[P_source_clean < 0] = 0

        if X_source_clean is not None:
            if list(X_source_clean.columns) != list(X_source.columns):
                raise ValueError("X_source_clean and X_source must have identical gene columns.")
            if len(X_source_clean) != len(X_source):
                raise ValueError("X_source_clean and X_source must have identical sample counts.")

        num_source = len(X_source)
        if seed is None:
            source_indices = np.random.permutation(num_source)
        else:
            source_indices = np.random.default_rng(int(seed)).permutation(num_source)
        split_idx_src = int(num_source * train_ratio)

        src_train_idx = source_indices[:split_idx_src]
        src_val_idx = source_indices[split_idx_src:]

        X_src_train = X_source.iloc[src_train_idx]
        X_src_val = X_source.iloc[src_val_idx]
        X_src_clean_train = (
            X_source_clean.iloc[src_train_idx] if X_source_clean is not None else None
        )
        X_src_clean_val = (
            X_source_clean.iloc[src_val_idx] if X_source_clean is not None else None
        )

        P_src_train = P_source_clean.iloc[src_train_idx]
        P_src_val = P_source_clean.iloc[src_val_idx]

        X_tgt_train = X_target

        print(f"Source split: Train {len(P_src_train)}, Val {len(P_src_val)}")
        print(f"Target adaptation: {len(X_tgt_train)} samples")

        self.datasets = {
            "source_train": BulkRNASeqDataset(
                X_bulk=X_src_train,
                X_clean_bulk=X_src_clean_train,
                P_df=P_src_train,
                device=device,
            ),
            "source_val": BulkRNASeqDataset(
                X_bulk=X_src_val,
                X_clean_bulk=X_src_clean_val,
                P_df=P_src_val,
                device=device,
            ),
            "target_train": BulkRNASeqDataset(
                X_bulk=X_tgt_train,
                P_df=None,
                device=device,
            ),
        }

        self.dataloaders = {}
        for loader_idx, (name, dataset) in enumerate(self.datasets.items()):
            shuffle = name.endswith("train")
            generator = None
            if seed is not None and shuffle:
                generator = torch.Generator()
                generator.manual_seed(int(seed) + loader_idx)

            self.dataloaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator,
            )

    def get_dataloader(self, domain: str = "source", split: str = "train") -> DataLoader:
        key = f"{domain}_{split}"
        return self.dataloaders[key]
