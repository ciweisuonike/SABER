import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if "saber" not in sys.modules:
    saber_pkg = types.ModuleType("saber")
    saber_pkg.__path__ = [str(ROOT / "saber")]
    sys.modules["saber"] = saber_pkg
_added_scanpy_stub = "scanpy" not in sys.modules
if _added_scanpy_stub:
    sys.modules["scanpy"] = types.ModuleType("scanpy")

import saber.pipeline.prop_train as prop_train_module

if _added_scanpy_stub:
    del sys.modules["scanpy"]


class MiniAdata:
    def __init__(self, labels):
        self.obs = pd.DataFrame(
            {"celltype": labels},
            index=[f"cell{i}" for i in range(len(labels))],
        )
        self.var_names = ["g0", "g1"]
        self.n_obs = len(self.obs)
        self.shape = (self.n_obs, len(self.var_names))

    def __getitem__(self, key):
        obs_key, _var_key = key
        if isinstance(obs_key, slice):
            obs = self.obs.iloc[obs_key].copy()
        else:
            obs = self.obs.iloc[np.asarray(obs_key)].copy()

        subset = MiniAdata([])
        subset.obs = obs
        subset.var_names = self.var_names
        subset.n_obs = len(obs)
        subset.shape = (subset.n_obs, len(subset.var_names))
        return subset

    def copy(self):
        copied = MiniAdata([])
        copied.obs = self.obs.copy()
        copied.var_names = list(self.var_names)
        copied.n_obs = self.n_obs
        copied.shape = self.shape
        return copied


def _config(n_cells):
    return {
        "reference_subset": {"n_cells": n_cells},
        "simulation": {"celltype_col": "celltype"},
    }


def test_reference_subset_null_keeps_input_unchanged():
    adata = MiniAdata(["A", "A", "B"])

    subset = prop_train_module._subset_reference_adata(adata, _config(None), seed=1)

    assert subset is adata
    assert subset.n_obs == 3


def test_reference_subset_samples_requested_cells_and_keeps_each_celltype():
    adata = MiniAdata(["A"] * 6 + ["B"] * 3 + ["C"])

    subset = prop_train_module._subset_reference_adata(adata, _config(5), seed=7)

    assert subset.n_obs == 5
    assert set(subset.obs["celltype"]) == {"A", "B", "C"}


def test_reference_subset_is_reproducible_with_same_seed():
    adata = MiniAdata(["A"] * 6 + ["B"] * 6 + ["C"] * 6)

    first = prop_train_module._subset_reference_adata(adata, _config(9), seed=123)
    second = prop_train_module._subset_reference_adata(adata, _config(9), seed=123)

    assert first.obs.index.tolist() == second.obs.index.tolist()


def test_reference_subset_rejects_too_few_cells_for_celltypes():
    adata = MiniAdata(["A", "B", "C"])

    with pytest.raises(ValueError, match="at least the number of cell types"):
        prop_train_module._subset_reference_adata(adata, _config(2), seed=1)
