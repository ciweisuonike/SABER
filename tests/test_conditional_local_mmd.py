import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if "saber" not in sys.modules:
    saber_pkg = types.ModuleType("saber")
    saber_pkg.__path__ = [str(ROOT / "saber")]
    sys.modules["saber"] = saber_pkg

from saber.models.loss import SABERLoss


def test_source_consistency_loss_matches_component_formula():
    criterion = SABERLoss()
    p_clean = torch.softmax(torch.randn(5, 3), dim=1)
    p_obs = torch.softmax(torch.randn(5, 3), dim=1)
    z_clean = torch.randn(5, 4)
    z_obs = torch.randn(5, 4)
    lambda_latent = 0.25

    combined = criterion.get_source_consistency_loss(
        p_clean,
        p_obs,
        z_clean,
        z_obs,
        lambda_latent,
    )
    expected = (
        criterion.get_directional_kl_loss(p_clean, p_obs)
        + lambda_latent * criterion.get_latent_consistency_loss(z_clean, z_obs)
    )

    assert torch.allclose(combined, expected)


def test_conditional_local_mmd_is_finite_and_nonnegative():
    criterion = SABERLoss()
    z_src = torch.randn(4, 3, requires_grad=True)
    z_tgt = torch.randn(2, 3, requires_grad=True)
    p_src = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.8, 0.2],
        ]
    )
    p_tgt = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    loss = criterion.get_conditional_local_mmd_loss(
        z_src,
        z_tgt,
        p_src,
        p_tgt,
        tau=10.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_conditional_local_mmd_weights_prefer_composition_neighbors():
    criterion = SABERLoss()
    p_src = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    p_tgt = torch.tensor(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ]
    )

    weights = criterion.get_conditional_local_mmd_weights(p_src, p_tgt, tau=20.0)

    assert weights[0, 0] > weights[1, 0]
    assert weights[1, 1] > weights[0, 1]
    assert torch.allclose(weights.sum(dim=0), torch.ones(2), atol=1e-6)


def test_conditional_local_mmd_detaches_target_pseudo_labels():
    criterion = SABERLoss()
    z_src = torch.randn(4, 3, requires_grad=True)
    z_tgt = torch.randn(2, 3, requires_grad=True)
    p_src = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.8, 0.2],
        ]
    )
    p_tgt = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        requires_grad=True,
    )

    loss = criterion.get_conditional_local_mmd_loss(
        z_src,
        z_tgt,
        p_src,
        p_tgt,
        tau=10.0,
    )
    loss.backward()

    assert z_src.grad is not None
    assert z_tgt.grad is not None
    assert z_src.grad.abs().sum() > 0
    assert z_tgt.grad.abs().sum() > 0
    assert p_tgt.grad is None
