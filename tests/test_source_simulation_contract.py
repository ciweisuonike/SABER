import inspect
import random
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

from saber.data.dataloader import MultiStageDataLoader
from saber.pipeline.simulate import run_simulation
import saber.simulate.bulk as bulk_module
import saber.simulate.gene_selection as gene_selection_module
from saber.simulate.gene_selection import select_genes_for_saber


def _small_adata(seed=1):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(6)]
    obs = pd.DataFrame(
        {"celltype": ["A", "A", "A", "B", "B", "B"]},
        index=[f"cell{i}" for i in range(6)],
    )
    adata = ad.AnnData(
        X=rng.poisson(lam=5, size=(6, 6)).astype(float),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    return adata, genes


def _three_celltype_adata(seed=21):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(6)]
    obs = pd.DataFrame(
        {"celltype": ["A", "A", "A", "B", "B", "B", "C", "C", "C"]},
        index=[f"cell{i}" for i in range(9)],
    )
    adata = ad.AnnData(
        X=rng.poisson(lam=5, size=(9, 6)).astype(float),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    return adata, genes


def _base_sim_config(seed=11):
    return {
        "K": 6,
        "N": 20,
        "n_batches": 2,
        "celltype_col": "celltype",
        "layer": "counts",
        "random_state": seed,
        "n_conditions": 2,
        "affected_gene_range": (1, 3),
        "n_batch_factors": 1,
        "enable_missing_celltypes": False,
        "enable_dominant_celltypes": False,
        "n_pseudo_banks": 1,
    }


def _reference_encoded_adata(reference_values):
    obs = pd.DataFrame(
        {
            "celltype": ["A"] * len(reference_values),
            "reference": reference_values,
        },
        index=[f"cell{i}" for i in range(len(reference_values))],
    )
    return ad.AnnData(
        X=np.arange(1, len(reference_values) + 1, dtype=float)[:, None],
        obs=obs,
        var=pd.DataFrame(index=["g0"]),
    )


def test_reference_schedule_samples_without_replacement_when_references_cover_banks():
    pools = [{"name": f"r{i}"} for i in range(5)]
    schedule = bulk_module._build_balanced_reference_schedule(
        pools,
        n_banks=3,
        rng=random.Random(7),
    )

    names = [pool["name"] for pool in schedule]
    assert len(names) == 3
    assert len(set(names)) == 3


def test_reference_schedule_uses_balanced_shuffled_rounds_when_banks_are_more():
    pools = [{"name": f"r{i}"} for i in range(3)]
    schedule = bulk_module._build_balanced_reference_schedule(
        pools,
        n_banks=8,
        rng=random.Random(9),
    )

    names = [pool["name"] for pool in schedule]
    expected_names = {"r0", "r1", "r2"}
    assert set(names[:3]) == expected_names
    assert set(names[3:6]) == expected_names
    assert len(set(names[6:])) == 2
    counts = pd.Series(names).value_counts()
    assert counts.max() - counts.min() <= 1


def test_reference_schedule_is_reproducible_for_the_same_seed():
    pools = [{"name": f"r{i}"} for i in range(4)]

    schedule_a = bulk_module._build_balanced_reference_schedule(
        pools, n_banks=7, rng=random.Random(13)
    )
    schedule_b = bulk_module._build_balanced_reference_schedule(
        pools, n_banks=7, rng=random.Random(13)
    )

    assert [pool["name"] for pool in schedule_a] == [
        pool["name"] for pool in schedule_b
    ]


def test_reference_encoded_simulation_uses_each_selected_reference_once():
    adata = _reference_encoded_adata(["r0", "r1", "r2", "r3"])
    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=23),
            "K": 3,
            "N": 1,
            "n_conditions": 1,
            "n_pseudo_banks": 3,
            "reference_col": "reference",
            "enable_twoview": False,
        },
    )

    selected_counts = result["X_clean_counts"]["g0"].to_numpy()
    assert len(np.unique(selected_counts)) == 3


def test_more_requested_banks_than_samples_creates_no_empty_banks(monkeypatch):
    adata = _reference_encoded_adata(["r0", "r1", "r2", "r3"])
    calls = {"refaug": 0}

    def record_reference_style(subE, **_kwargs):
        calls["refaug"] += 1
        return np.asarray(subE, dtype=np.float64)

    monkeypatch.setattr(bulk_module, "_apply_reference_style", record_reference_style)
    run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=29),
            "K": 3,
            "N": 1,
            "n_conditions": 1,
            "n_pseudo_banks": 10,
            "reference_col": "reference",
            "enable_twoview": True,
            "prob_ref_aug": 1.0,
            "prob_batch_inj": 0.0,
            "gaussian_noise_level": 0.0,
            "affected_gene_range": (1, 2),
        },
    )

    assert calls["refaug"] == 3


def test_invalid_or_single_reference_column_keeps_global_pool_fallback():
    base = _reference_encoded_adata(["r0", "r1", "r2"])
    configs = []

    missing = base.copy()
    missing.obs = missing.obs.drop(columns=["reference"])
    configs.append((missing, "missing"))

    empty = base.copy()
    empty.obs["reference"] = ["", " ", None]
    configs.append((empty, "reference"))

    single = base.copy()
    single.obs["reference"] = "r0"
    configs.append((single, "reference"))

    outputs = []
    for adata, reference_col in configs:
        result = run_simulation(
            adata,
            sim_config={
                **_base_sim_config(seed=31),
                "K": 6,
                "N": 5,
                "n_conditions": 1,
                "n_pseudo_banks": 3,
                "reference_col": reference_col,
                "enable_twoview": False,
            },
        )
        outputs.append(result["X_clean_counts"].to_numpy())

    assert np.array_equal(outputs[0], outputs[1])
    assert np.array_equal(outputs[0], outputs[2])


def test_missing_celltype_in_reference_still_uses_global_celltype_pool():
    obs = pd.DataFrame(
        {
            "celltype": ["A", "A", "B", "B"],
            "reference": ["r0", "r1", "r1", "r1"],
        },
        index=[f"cell{i}" for i in range(4)],
    )
    adata = ad.AnnData(
        X=np.array([[10, 0], [11, 0], [0, 20], [0, 21]], dtype=float),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )

    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=37),
            "K": 2,
            "N": 1000,
            "n_conditions": 1,
            "n_pseudo_banks": 2,
            "reference_col": "reference",
            "enable_twoview": False,
        },
    )

    assert np.all(result["P"]["B"].to_numpy() > 0)
    assert np.all(result["X_clean_counts"]["g1"].to_numpy() > 0)


def test_reference_and_batch_probabilities_keep_half_defaults():
    signature = inspect.signature(bulk_module.simulate_bulk_with_batch_effect)
    assert signature.parameters["prob_ref_aug"].default == 0.5
    assert signature.parameters["prob_batch_inj"].default == 0.5


def test_run_simulation_produces_clean_source_without_target_calibration():
    adata, genes = _small_adata(seed=1)

    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=11),
            "K": 5,
            "enable_twoview": False,
        },
    )

    assert {"X_obs", "X_clean", "X_clean_counts", "P", "B", "celltypes"}.issubset(result)
    assert result["X_obs"].shape == (5, 6)
    assert result["X_clean"].shape == (5, 6)
    assert result["P"].shape[0] == 5
    assert list(result["X_obs"].columns) == genes
    assert list(result["X_clean"].columns) == genes
    assert np.all(result["X_clean_counts"].values >= 0)


def test_reference_augmentation_changes_observed_not_clean_teacher():
    adata, _genes = _small_adata(seed=6)
    base_config = {
        **_base_sim_config(seed=17),
        "n_conditions": 1,
        "prob_ref_aug": 1.0,
        "prob_batch_inj": 0.0,
    }

    no_refaug = run_simulation(
        adata,
        sim_config={**base_config, "enable_twoview": False},
    )
    with_refaug = run_simulation(
        adata,
        sim_config={**base_config, "enable_twoview": True},
    )

    assert np.allclose(
        no_refaug["X_clean_counts"].values,
        with_refaug["X_clean_counts"].values,
    )
    assert np.allclose(no_refaug["P"].values, with_refaug["P"].values)
    assert np.allclose(
        no_refaug["X_proc"].values,
        no_refaug["X_clean_counts"].values,
    )
    assert not np.allclose(
        with_refaug["X_proc"].values,
        with_refaug["X_clean_counts"].values,
    )


def test_reference_augmentation_is_cell_level_before_bulk_summing(monkeypatch):
    adata, _genes = _small_adata(seed=7)

    def add_one_reference_style(subE, **_kwargs):
        return np.asarray(subE, dtype=np.float64) + 1.0

    monkeypatch.setattr(bulk_module, "_apply_reference_style", add_one_reference_style)
    config = {
        **_base_sim_config(seed=19),
        "K": 4,
        "N": 20,
        "n_conditions": 1,
        "enable_twoview": True,
        "prob_ref_aug": 1.0,
        "prob_batch_inj": 0.0,
    }

    result = run_simulation(adata, sim_config=config)
    diff = result["X_proc"].values - result["X_clean_counts"].values

    assert np.allclose(diff, float(config["N"]))


def test_run_simulation_twoview_disabled_uses_clean_single_view():
    adata, _genes = _small_adata(seed=2)

    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=12),
            "K": 5,
            "enable_twoview": False,
        },
    )

    assert result["enable_twoview"] is False
    assert np.allclose(result["X_obs"].values, result["X_clean"].values)

    loader = MultiStageDataLoader(
        X_source=result["X_obs"],
        X_source_clean=None,
        P_source=result["P"],
        X_target=result["X_obs"],
        batch_size=2,
        train_ratio=0.8,
        device="cpu",
    )
    source_batch = next(iter(loader.get_dataloader("source", "train")))
    assert len(source_batch) == 2


def test_confidence_gene_dropout_only_changes_observed_branch(monkeypatch):
    adata, genes = _small_adata(seed=3)

    def drop_every_gene(
        X,
        gene_confidence,
        dropout_max,
        dropout_kappa,
        dropout_retention,
        rng,
    ):
        return np.zeros_like(X)

    monkeypatch.setattr(
        bulk_module,
        "_apply_confidence_gene_dropout",
        drop_every_gene,
    )
    base_config = {
        **_base_sim_config(seed=13),
        "K": 5,
        "enable_gene_selection": True,
        "enable_twoview": True,
        "prob_ref_aug": 0.0,
        "prob_batch_inj": 0.0,
        "enable_confidence_gene_dropout": False,
        "confidence_gene_dropout_max": 0.3,
        "gene_confidence": [0.0] * len(genes),
    }

    without_dropout = run_simulation(adata, sim_config=base_config)
    with_dropout = run_simulation(
        adata,
        sim_config={**base_config, "enable_confidence_gene_dropout": True},
    )

    assert np.allclose(
        without_dropout["X_clean_counts"].values,
        with_dropout["X_clean_counts"].values,
    )
    assert np.allclose(without_dropout["X_clean"].values, with_dropout["X_clean"].values)
    assert np.allclose(without_dropout["P"].values, with_dropout["P"].values)
    assert np.allclose(with_dropout["X_proc"].values, 0.0)
    assert not np.allclose(without_dropout["X_proc"].values, with_dropout["X_proc"].values)


def test_gene_selection_disabled_forces_confidence_dropout_off(monkeypatch):
    adata, genes = _small_adata(seed=4)

    def fail_if_called(
        X,
        gene_confidence,
        dropout_max,
        dropout_kappa,
        dropout_retention,
        rng,
    ):
        raise AssertionError("confidence gene dropout should be disabled")

    monkeypatch.setattr(
        bulk_module,
        "_apply_confidence_gene_dropout",
        fail_if_called,
    )
    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=14),
            "K": 5,
            "enable_gene_selection": False,
            "enable_twoview": True,
            "prob_ref_aug": 0.0,
            "prob_batch_inj": 0.0,
            "enable_confidence_gene_dropout": True,
            "confidence_gene_dropout_max": 0.3,
            "gene_confidence": [0.0] * len(genes),
        },
    )

    assert np.allclose(result["X_proc"].values, result["X_clean_counts"].values)


def test_twoview_enabled_runs_batch_and_expression_dropout(monkeypatch):
    adata, _genes = _small_adata(seed=8)
    called = {"batch": False, "dropout": False}

    def mark_batch(X, *_args):
        called["batch"] = True
        return np.asarray(X, dtype=float) + 1.0

    def mark_dropout(X, *_args):
        called["dropout"] = True
        return np.asarray(X, dtype=float) + 2.0

    monkeypatch.setattr(bulk_module, "_apply_batch_effects", mark_batch)
    monkeypatch.setattr(bulk_module, "_apply_expression_based_dropout", mark_dropout)

    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=16),
            "K": 4,
            "enable_twoview": True,
            "prob_ref_aug": 0.0,
            "prob_batch_inj": 1.0,
            "gaussian_noise_level": 0.0,
        },
    )

    assert called == {"batch": True, "dropout": True}
    assert not np.allclose(result["X_proc"].values, result["X_clean_counts"].values)


def test_missing_celltype_samples_are_preserved():
    adata, _genes = _three_celltype_adata(seed=9)

    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=18),
            "K": 6,
            "N": 60,
            "n_conditions": 1,
            "enable_twoview": False,
            "missing_celltype_fraction": 1.0,
            "min_celltypes_per_sample": 2,
            "enable_missing_celltypes": True,
            "enable_dominant_celltypes": False,
        },
    )

    zero_counts = (result["P"].values == 0.0).sum(axis=1)
    assert np.all(zero_counts >= 1)
    assert np.allclose(result["P"].values.sum(axis=1), 1.0)


def test_dominant_celltype_samples_are_preserved():
    adata, _genes = _three_celltype_adata(seed=10)

    result = run_simulation(
        adata,
        sim_config={
            **_base_sim_config(seed=20),
            "K": 6,
            "N": 100,
            "n_conditions": 1,
            "enable_twoview": False,
            "missing_celltype_fraction": 0.0,
            "enable_missing_celltypes": False,
            "enable_dominant_celltypes": True,
            "dominant_sample_fraction": 1.0,
            "min_dominant_fraction": 0.7,
            "max_dominant_fraction": 0.7,
        },
    )

    assert np.allclose(result["P"].values.max(axis=1), 0.7)
    assert np.allclose(result["P"].values.sum(axis=1), 1.0)


def test_confidence_gene_dropout_shrinks_to_current_gene_means():
    class FixedRng:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def random(self, shape):
            assert shape == self.values.shape
            return self.values

    X = np.asarray(
        [
            [0.0, 10.0, 100.0],
            [10.0, 20.0, 200.0],
        ]
    )
    gene_confidence = np.asarray([0.0, 0.5, 1.0])
    rng_values = np.asarray(
        [
            [0.70, 0.30, 0.00],
            [0.81, 0.41, 0.99],
        ]
    )

    linear = bulk_module._apply_confidence_gene_dropout(
        X,
        gene_confidence,
        dropout_max=0.8,
        dropout_kappa=1.0,
        dropout_retention=0.5,
        rng=FixedRng(rng_values),
    )
    np.testing.assert_allclose(
        linear,
        np.asarray(
            [
            [2.5, 12.5, 100.0],
            [10.0, 20.0, 200.0],
            ]
        ),
    )

    nonlinear = bulk_module._apply_confidence_gene_dropout(
        X,
        gene_confidence,
        dropout_max=0.8,
        dropout_kappa=2.0,
        dropout_retention=0.5,
        rng=FixedRng(rng_values),
    )
    np.testing.assert_allclose(
        nonlinear,
        np.asarray(
            [
            [2.5, 10.0, 100.0],
            [10.0, 20.0, 200.0],
            ]
        ),
    )


def test_confidence_gene_dropout_retention_one_preserves_expression():
    class ZeroRng:
        def random(self, shape):
            return np.zeros(shape)

    X = np.asarray([[1.0, 3.0], [5.0, 7.0]])
    result = bulk_module._apply_confidence_gene_dropout(
        X,
        gene_confidence=np.zeros(2),
        dropout_max=0.9,
        dropout_kappa=2.0,
        dropout_retention=1.0,
        rng=ZeroRng(),
    )

    assert result == pytest.approx(X)

def test_gene_confidence_average_rank_matches_selected_gene_order():
    confidence = gene_selection_module._gene_confidence_from_selection_weights(
        ["g2", "g0", "g1"],
        {"g0": 2.0, "g1": 2.0, "g2": 1.0},
    )

    assert confidence == pytest.approx([0.0, 0.75, 0.75])


def test_gene_selection_disabled_does_not_require_celltype_labels():
    rng = np.random.default_rng(4)
    genes = [f"g{i}" for i in range(8)]
    adata = ad.AnnData(
        X=rng.poisson(lam=5, size=(6, 8)).astype(float),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(6)]),
        var=pd.DataFrame(index=genes),
    )
    bulk = pd.DataFrame(
        rng.poisson(lam=80, size=(3, 8)).astype(float),
        index=[f"target{i}" for i in range(3)],
        columns=genes,
    )

    selected = select_genes_for_saber(
        adata,
        bulk,
        {
            "top_genes": 3,
            "enable_gene_selection": False,
            "celltype_col": "missing_celltype",
        },
        return_info=False,
    )

    assert len(selected) == 3
    assert selected == sorted(selected)
    assert set(selected).issubset(set(genes))

    selected, info = select_genes_for_saber(
        adata,
        bulk,
        {
            "top_genes": 3,
            "enable_gene_selection": False,
            "celltype_col": "missing_celltype",
        },
        return_info=True,
    )
    assert info["gene_confidence"] == [1.0] * len(selected)
