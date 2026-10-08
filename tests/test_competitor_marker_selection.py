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


def _run_gene_selection(monkeypatch, expression, labels, genes):
    adata = ad.AnnData(
        X=expression.copy(),
        obs=pd.DataFrame(
            {"celltype": labels},
            index=[f"cell{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=genes),
    )
    bulk = pd.DataFrame(
        np.ones((2, len(genes))),
        index=["sample1", "sample2"],
        columns=genes,
    )
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    monkeypatch.setattr(
        gene_selection_module,
        "_normalized_log_expression",
        lambda current: expression[
            :, [gene_index[str(gene)] for gene in current.var_names]
        ].copy(),
    )

    return gene_selection_module.select_genes_for_saber(
        adata,
        bulk,
        {
            "top_genes": len(genes) - 1,
            "celltype_col": "celltype",
            "marker_genes_per_celltype": len(genes),
            "min_pct_in_celltype": 0.05,
            "min_logfc": 0.25,
            "enable_target_stability_filter": False,
            "enable_reference_stability_filter": False,
        },
        return_info=True,
    )


def test_competitor_filter_keeps_identity_markers_only(monkeypatch):
    genes = [
        "shared_ab",
        "a_identity",
        "b_identity",
        "c_identity",
        "equal_detection",
        "background",
    ]
    expression = np.asarray(
        [
            [10, 10, 0, 0, 10, 1],
            [10, 10, 0, 0, 10, 1],
            [9, 0, 10, 0, 1, 1],
            [9, 0, 10, 0, 1, 1],
            [0, 0, 0, 10, 0, 1],
            [0, 0, 0, 10, 0, 1],
        ],
        dtype=float,
    )

    selected, info = _run_gene_selection(
        monkeypatch,
        expression,
        np.asarray(["A", "A", "B", "B", "C", "C"]),
        genes,
    )

    assert set(info["marker_genes"]) == {
        "a_identity",
        "b_identity",
        "c_identity",
    }
    assert "shared_ab" not in info["marker_genes"]
    assert "equal_detection" not in info["marker_genes"]
    assert len(selected) == len(genes) - 1
    assert selected == sorted(selected)


def test_competitor_logfc_uses_inclusive_existing_threshold(monkeypatch):
    eps = 1e-8
    competitor_mean = 4.0
    at_threshold = (competitor_mean + eps) * np.exp(0.25) - eps
    at_threshold = np.nextafter(at_threshold, np.inf)
    genes = [
        "below",
        "at_threshold",
        "above",
        "b_identity",
        "c_identity",
        "background",
    ]
    expression = np.asarray(
        [
            [
                (competitor_mean + eps) * np.exp(0.249) - eps,
                at_threshold,
                (competitor_mean + eps) * np.exp(0.251) - eps,
                0,
                0,
                1,
            ],
            [
                (competitor_mean + eps) * np.exp(0.249) - eps,
                at_threshold,
                (competitor_mean + eps) * np.exp(0.251) - eps,
                0,
                0,
                1,
            ],
            [2 * competitor_mean, 2 * competitor_mean, 2 * competitor_mean, 10, 0, 1],
            [0, 0, 0, 10, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 10, 1],
        ],
        dtype=float,
    )

    _, info = _run_gene_selection(
        monkeypatch,
        expression,
        np.asarray(["A", "A", "B", "B", "C", "C"]),
        genes,
    )

    marker_genes = set(info["marker_genes"])
    assert "below" not in marker_genes
    assert "at_threshold" in marker_genes
    assert "above" in marker_genes
