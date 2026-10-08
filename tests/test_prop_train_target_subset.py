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
_added_scanpy_stub = "scanpy" not in sys.modules
if _added_scanpy_stub:
    sys.modules["scanpy"] = types.ModuleType("scanpy")

import saber.pipeline.prop_train as prop_train_module

if _added_scanpy_stub:
    del sys.modules["scanpy"]


class FakeAdata:
    def __init__(self, var_names):
        self.var_names = list(var_names)

    def __getitem__(self, key):
        _obs_key, var_key = key
        if isinstance(var_key, slice):
            return FakeAdata(self.var_names[var_key])
        return FakeAdata(list(var_key))

    def copy(self):
        return FakeAdata(self.var_names)


class FakeModel:
    def to(self, device):
        self.device = device
        return self


class FakeLoader:
    def __init__(
        self,
        X_source,
        P_source,
        X_target,
        X_source_clean=None,
        batch_size=32,
        train_ratio=0.8,
        device="cpu",
        seed=None,
    ):
        self.X_target = X_target


def test_select_target_training_bulk_uses_all_for_null_and_large_values():
    bulk = pd.DataFrame(
        np.arange(12, dtype=float).reshape(4, 3),
        index=[f"target{i}" for i in range(4)],
        columns=["g0", "g1", "g2"],
    )

    assert prop_train_module._select_target_training_bulk(bulk, None).index.tolist() == bulk.index.tolist()
    assert prop_train_module._select_target_training_bulk(bulk, 0).index.tolist() == bulk.index.tolist()
    assert prop_train_module._select_target_training_bulk(bulk, 10).index.tolist() == bulk.index.tolist()


def test_prop_train_target_subset_feeds_gene_selection_and_mmd_but_returns_full_bulk(monkeypatch):
    captured = {}
    genes = ["g0", "g1", "g2"]
    bulk = pd.DataFrame(
        np.arange(15, dtype=float).reshape(5, 3),
        index=[f"target{i}" for i in range(5)],
        columns=genes,
    )

    def fake_select_genes_for_saber(adata, bulk, sim_config, return_info=False):
        captured["gene_selection_index"] = bulk.index.tolist()
        captured["gene_selection_seed"] = sim_config["random_state"]
        return ["g0", "g1"], {"gene_confidence": [1.0, 1.0]}

    def fake_run_simulation(adata, sim_config):
        captured["sim_var_names"] = adata.var_names
        captured["simulation_seed"] = sim_config["random_state"]
        return {}

    def fake_preprocess_simulated_data(sim_data):
        return {
            "X_obs": pd.DataFrame([[1.0, 2.0]], columns=["g0", "g1"]),
            "X_clean": pd.DataFrame([[1.0, 2.0]], columns=["g0", "g1"]),
            "P": pd.DataFrame([[1.0]], columns=["A"]),
            "gene_names": ["g0", "g1"],
            "enable_twoview": True,
        }

    def fake_transform_bulk_dataframe_to_analysis_scale(df, transform_info=None, return_info=False):
        captured["mmd_index"] = df.index.tolist()
        captured["transform_info_reused"] = transform_info
        if return_info:
            return df.copy(), transform_info
        return df.copy()

    def fake_multi_stage_loader(**kwargs):
        captured["loader_index"] = kwargs["X_target"].index.tolist()
        captured["loader_seed"] = kwargs["seed"]
        return FakeLoader(**kwargs)

    monkeypatch.setattr(
        prop_train_module,
        "resolve_bulk_transform_info",
        lambda values: {"summary": "full-bulk-transform", "n_rows": values.shape[0]},
    )
    monkeypatch.setattr(prop_train_module, "select_genes_for_saber", fake_select_genes_for_saber)
    monkeypatch.setattr(prop_train_module, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(prop_train_module, "preprocess_simulated_data", fake_preprocess_simulated_data)
    monkeypatch.setattr(
        prop_train_module,
        "transform_bulk_dataframe_to_analysis_scale",
        fake_transform_bulk_dataframe_to_analysis_scale,
    )
    monkeypatch.setattr(prop_train_module, "MultiStageDataLoader", fake_multi_stage_loader)
    monkeypatch.setattr(prop_train_module, "SABER", lambda config: FakeModel())
    monkeypatch.setattr(prop_train_module, "SABERLoss", lambda: object())
    monkeypatch.setattr(
        prop_train_module,
        "set_global_seed",
        lambda seed, deterministic=False: captured.update(
            {"global_seed": seed, "deterministic": deterministic}
        ),
    )
    monkeypatch.setattr(
        prop_train_module,
        "run_training",
        lambda model, criterion, multi_loader, config, device: (model, {"pretrain": [], "finetune": []}),
    )

    _model, _history, bulk_align, returned_genes, _celltypes, model_config = (
        prop_train_module.run_prop_train_from_scRNAseq(
            adata=FakeAdata(genes),
            bulk=bulk,
            config={
                "reproducibility": {
                    "seed": 123,
                    "deterministic": True,
                },
                "simulation": {
                    "n_target_sample_to_train": 2,
                },
                "dataloader": {
                    "batch_size": 2,
                    "train_ratio": 0.8,
                },
                "model": {},
                "training": {},
                "loss": {},
            },
            device="cpu",
        )
    )

    assert captured["gene_selection_index"] == ["target0", "target1"]
    assert captured["global_seed"] == 123
    assert captured["deterministic"] is True
    assert captured["gene_selection_seed"] == 123
    assert captured["simulation_seed"] == 123
    assert captured["mmd_index"] == ["target0", "target1"]
    assert captured["loader_index"] == ["target0", "target1"]
    assert captured["loader_seed"] == 123
    assert captured["transform_info_reused"]["n_rows"] == 5
    assert bulk_align.index.tolist() == bulk.index.tolist()
    assert bulk_align.columns.tolist() == ["g0", "g1"]
    assert returned_genes == ["g0", "g1"]
    assert model_config["target_bulk_transform_info"]["n_rows"] == 5
