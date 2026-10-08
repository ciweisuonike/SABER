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

pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")

import saber.simulate.gene_selection as gene_selection_module


def _adata_with_obs(obs):
    n_cells = len(next(iter(obs.values())))
    return ad.AnnData(
        X=np.ones((n_cells, 3), dtype=float),
        obs=pd.DataFrame(obs, index=[f"cell{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )


def _compute_reference_weights(adata, x_norm, **config):
    labels = adata.obs["celltype"].astype(str).to_numpy()
    return gene_selection_module._compute_reference_stability_weights(
        adata=adata,
        x_norm=np.asarray(x_norm, dtype=float),
        labels=labels,
        celltypes=np.unique(labels),
        cfg={
            "reference_col": "reference",
            "reference_min_cells_per_group": 10,
            **config,
        },
    )


@pytest.mark.parametrize(
    "obs",
    [
        {"celltype": ["A"] * 20},
        {"celltype": ["A"] * 20, "reference": ["r1"] * 20},
    ],
)
def test_reference_stability_skips_missing_or_single_reference(obs):
    adata = _adata_with_obs(obs)
    weights, mask, stats = _compute_reference_weights(
        adata,
        x_norm=np.ones((20, 3), dtype=float),
    )

    assert stats["enabled"] is False
    assert np.allclose(weights, 1.0)
    assert np.all(mask)


def test_reference_stability_skips_when_disabled():
    adata = _adata_with_obs(
        {
            "celltype": ["A"] * 20,
            "reference": ["r1"] * 10 + ["r2"] * 10,
        }
    )
    weights, mask, stats = _compute_reference_weights(
        adata,
        x_norm=np.ones((20, 3), dtype=float),
        enable_reference_stability_filter=False,
    )

    assert stats["enabled"] is False
    assert np.allclose(weights, 1.0)
    assert np.all(mask)


def test_single_reference_does_not_activate_reference_parameter_validation():
    adata = _adata_with_obs(
        {
            "celltype": ["A"] * 20,
            "reference": ["r1"] * 20,
        }
    )
    weights, mask, stats = _compute_reference_weights(
        adata,
        x_norm=np.ones((20, 3), dtype=float),
        reference_celltype_quantile=-1.0,
        reference_stability_gamma=-1.0,
    )

    assert stats["enabled"] is False
    assert np.allclose(weights, 1.0)
    assert np.all(mask)


def test_reference_stability_penalizes_reference_shift_within_celltype():
    adata = _adata_with_obs(
        {
            "celltype": ["A"] * 20,
            "reference": ["r1"] * 10 + ["r2"] * 10,
        }
    )
    within_group = np.arange(10, dtype=float) / 10.0
    reference_one = np.column_stack(
        [within_group + 1.0, within_group + 2.0, within_group + 3.0]
    )
    reference_two = np.column_stack(
        [within_group + 6.0, within_group + 2.0, within_group + 3.0]
    )
    weights, mask, stats = _compute_reference_weights(
        adata,
        x_norm=np.vstack([reference_one, reference_two]),
        reference_hard_filter_quantile=0.5,
    )

    assert stats["enabled"] is True
    assert stats["valid_references"] == 2
    assert stats["comparable_celltypes"] == 1
    assert mask.tolist() == [False, True, True]
    assert weights[0] == 0.0
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)


def test_reference_stability_skips_underpopulated_groups():
    adata = _adata_with_obs(
        {
            "celltype": ["A"] * 6,
            "reference": ["r1"] * 3 + ["r2"] * 3,
        }
    )
    weights, mask, stats = _compute_reference_weights(
        adata,
        x_norm=np.ones((6, 3), dtype=float),
    )

    assert stats["enabled"] is False
    assert "no cell type" in stats["reason"]
    assert np.allclose(weights, 1.0)
    assert np.all(mask)


def test_reference_stability_keeps_ties_at_hard_filter_threshold():
    adata = _adata_with_obs(
        {
            "celltype": ["A"] * 20,
            "reference": ["r1"] * 10 + ["r2"] * 10,
        }
    )
    within_group = np.arange(10, dtype=float)[:, None]
    reference = np.hstack([within_group, within_group, within_group])
    x_norm = np.vstack([reference, reference])
    weights, mask, stats = _compute_reference_weights(adata, x_norm=x_norm)

    assert stats["enabled"] is True
    assert np.all(mask)
    assert np.allclose(weights, 1.0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("reference_celltype_quantile", -0.1),
        ("reference_hard_filter_quantile", 1.1),
        ("reference_stability_gamma", -1.0),
    ],
)
def test_reference_stability_rejects_invalid_config(key, value):
    adata = _adata_with_obs(
        {
            "celltype": ["A"] * 20,
            "reference": ["r1"] * 10 + ["r2"] * 10,
        }
    )

    with pytest.raises(ValueError):
        _compute_reference_weights(
            adata,
            x_norm=np.ones((20, 3), dtype=float),
            **{key: value},
        )


def test_run_simulation_strips_reference_selection_keys_but_keeps_reference_col(monkeypatch):
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
            "enable_reference_stability_filter": True,
            "reference_min_cells_per_group": 10,
            "reference_celltype_quantile": 0.75,
            "reference_hard_filter_quantile": 0.95,
            "reference_stability_gamma": 1.0,
        },
    )

    assert captured["reference_col"] == "reference"
    assert "enable_reference_stability_filter" not in captured
    assert "reference_min_cells_per_group" not in captured
    assert "reference_celltype_quantile" not in captured
    assert "reference_hard_filter_quantile" not in captured
    assert "reference_stability_gamma" not in captured
