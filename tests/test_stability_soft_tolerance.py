import importlib
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

import saber.simulate.gene_selection as gene_selection_module

if _added_scanpy_stub:
    del sys.modules["scanpy"]


def test_source_target_stability_soft_threshold_penalizes_only_excess(monkeypatch):
    source_pre = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=float,
    )
    target_pre = pd.DataFrame(
        [[2.5, 3.5, 5.0]],
        columns=["g0", "g1", "g2"],
    )

    def fake_transform(values, transform_info=None, return_info=False):
        values = np.asarray(values, dtype=float)
        if return_info:
            return values, {"mock_transform": True}
        return values

    monkeypatch.setattr(
        gene_selection_module,
        "_make_source_pseudobulk",
        lambda **_kwargs: source_pre,
    )
    monkeypatch.setattr(
        gene_selection_module,
        "transform_bulk_matrix_to_analysis_scale",
        fake_transform,
    )
    monkeypatch.setattr(
        gene_selection_module,
        "rank_percentile_rows",
        lambda values: np.asarray(values, dtype=float),
    )

    weights, mask, stats = gene_selection_module._compute_stability_weights(
        x_norm=np.ones((2, 3), dtype=float),
        labels=np.asarray(["A", "A"]),
        celltypes=np.asarray(["A"]),
        bulk=target_pre,
        cfg={
            "stability_gamma": 2.0,
            "stability_soft_threshold": 0.5,
            "source_target_shift_threshold": 2.0,
        },
    )

    assert mask.tolist() == [True, True, False]
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(np.exp(-2.0))
    assert weights[2] == pytest.approx(0.0)
    assert stats["source_target_shift"] == 1


def test_reference_stability_soft_threshold_penalizes_only_excess():
    adata = type("Adata", (), {})()
    adata.obs = pd.DataFrame(
        {
            "celltype": ["A"] * 20,
            "reference": ["r1"] * 10 + ["r2"] * 10,
        }
    )
    base = np.arange(10, dtype=float)
    base = (base - base.mean()) / base.std(ddof=1)
    reference_one = np.column_stack([base, base, base])
    reference_two = np.column_stack(
        [
            base + 1.0,
            base + 3.0,
            base + 6.0,
        ]
    )

    weights, mask, stats = gene_selection_module._compute_reference_stability_weights(
        adata=adata,
        x_norm=np.vstack([reference_one, reference_two]),
        labels=adata.obs["celltype"].to_numpy(),
        celltypes=np.asarray(["A"]),
        cfg={
            "reference_col": "reference",
            "reference_min_cells_per_group": 10,
            "reference_hard_filter_quantile": 0.75,
            "reference_stability_gamma": 2.0,
            "reference_stability_soft_threshold": 0.5,
        },
    )

    assert stats["enabled"] is True
    assert mask.tolist() == [True, True, False]
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(np.exp(-2.0))
    assert weights[2] == pytest.approx(0.0)


def test_run_simulation_strips_soft_tolerance_selection_keys(monkeypatch):
    pipeline_simulate = importlib.import_module("saber.pipeline.simulate")
    captured = {}
    expression = pd.DataFrame([[0.0]], columns=["g0"], index=["sample_1"])
    proportions = pd.DataFrame([[1.0]], columns=["A"], index=["sample_1"])
    batches = pd.DataFrame({"batch": [0]}, index=["sample_1"])

    def fake_simulate_bulk_with_batch_effect(adata, **kwargs):
        captured.update(kwargs)
        return (
            expression,
            expression,
            expression,
            expression,
            proportions,
            batches,
            np.asarray(["A"]),
        )

    monkeypatch.setattr(
        pipeline_simulate,
        "simulate_bulk_with_batch_effect",
        fake_simulate_bulk_with_batch_effect,
    )
    pipeline_simulate.run_simulation(
        adata=object(),
        sim_config={
            "reference_col": "reference",
            "n_target_sample_to_train": 1,
            "stability_soft_threshold": 0.5,
            "reference_stability_soft_threshold": 1.0,
        },
    )

    assert captured["reference_col"] == "reference"
    assert "n_target_sample_to_train" not in captured
    assert "stability_soft_threshold" not in captured
    assert "reference_stability_soft_threshold" not in captured
