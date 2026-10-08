import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if "saber" not in sys.modules:
    saber_pkg = types.ModuleType("saber")
    saber_pkg.__path__ = [str(ROOT / "saber")]
    sys.modules["saber"] = saber_pkg

from saber.data.dataloader import MultiStageDataLoader


def test_target_adaptation_uses_all_target_samples_without_val_split():
    genes = ["g0", "g1", "g2"]
    X_source = pd.DataFrame(
        np.arange(18, dtype=np.float32).reshape(6, 3),
        index=[f"src{i}" for i in range(6)],
        columns=genes,
    )
    X_source_clean = X_source + 1.0
    P_source = pd.DataFrame(
        np.tile([[0.7, 0.3]], (6, 1)),
        index=X_source.index,
        columns=["A", "B"],
    )
    X_target = pd.DataFrame(
        np.arange(15, dtype=np.float32).reshape(5, 3),
        index=[f"tgt{i}" for i in range(5)],
        columns=genes,
    )

    loader = MultiStageDataLoader(
        X_source=X_source,
        X_source_clean=X_source_clean,
        P_source=P_source,
        X_target=X_target,
        batch_size=2,
        train_ratio=0.5,
        device="cpu",
    )

    assert len(loader.datasets["source_train"]) == 3
    assert len(loader.datasets["source_val"]) == 3
    assert len(loader.datasets["target_train"]) == len(X_target)
    assert "target_val" not in loader.datasets
    assert loader.get_dataloader("target", "train") is loader.dataloaders["target_train"]


def test_seed_controls_source_split_and_train_shuffle():
    genes = ["g0", "g1"]
    X_source = pd.DataFrame(
        np.arange(16, dtype=np.float32).reshape(8, 2),
        index=[f"src{i}" for i in range(8)],
        columns=genes,
    )
    P_source = pd.DataFrame(
        np.arange(16, dtype=np.float32).reshape(8, 2),
        index=X_source.index,
        columns=["A", "B"],
    )
    X_target = pd.DataFrame(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        index=[f"tgt{i}" for i in range(4)],
        columns=genes,
    )

    def build(seed):
        return MultiStageDataLoader(
            X_source=X_source,
            P_source=P_source,
            X_target=X_target,
            batch_size=2,
            train_ratio=0.75,
            device="cpu",
            seed=seed,
        )

    first = build(seed=7)
    second = build(seed=7)
    third = build(seed=8)

    assert np.array_equal(first.datasets["source_train"].X, second.datasets["source_train"].X)
    assert not np.array_equal(first.datasets["source_train"].X, third.datasets["source_train"].X)

    def train_order(loader):
        order = []
        for _x, p in loader.get_dataloader("source", "train"):
            order.extend(p[:, 0].numpy().tolist())
        return order

    assert train_order(first) == train_order(second)
