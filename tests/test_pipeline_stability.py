import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if "saber" not in sys.modules:
    saber_pkg = types.ModuleType("saber")
    saber_pkg.__path__ = [str(ROOT / "saber")]
    sys.modules["saber"] = saber_pkg

from saber.models.saber import SABER

preprocess_spec = importlib.util.spec_from_file_location(
    "saber_pipeline_preprocess_for_test",
    ROOT / "saber" / "pipeline" / "preprocess.py",
)
preprocess_module = importlib.util.module_from_spec(preprocess_spec)
preprocess_spec.loader.exec_module(preprocess_module)
preprocess_simulated_data = preprocess_module.preprocess_simulated_data

train_spec = importlib.util.spec_from_file_location(
    "saber_pipeline_train_for_stability_test",
    ROOT / "saber" / "pipeline" / "train.py",
)
train_module = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train_module)
run_training = train_module.run_training


def _sim_data(x_obs=None, x_clean=None, x_clean_counts=None, p=None):
    x_obs = x_obs if x_obs is not None else pd.DataFrame([[1.0, 2.0]], columns=["g0", "g1"])
    x_clean = x_clean if x_clean is not None else x_obs.copy()
    x_clean_counts = x_clean_counts if x_clean_counts is not None else x_obs.copy()
    p = p if p is not None else pd.DataFrame([[1.0]], columns=["A"])
    return {
        "X_obs": x_obs,
        "X_clean": x_clean,
        "X_clean_counts": x_clean_counts,
        "P": p,
        "B": pd.DataFrame({"batch": [0]}, index=x_obs.index),
        "celltypes": ["A"],
    }


def _model():
    return SABER(
        {
            "num_cell_types": 2,
            "gene_dim": 3,
            "prop_dim": 4,
            "encoder_hidden_dim": 5,
            "proportion_head_hidden_dim": 6,
            "dropout_rate": 0.0,
        }
    )


class EmptyLoader:
    def get_dataloader(self, domain, split):
        return []


def test_preprocess_raises_value_error_for_sample_mismatch():
    bad_p = pd.DataFrame([[1.0], [1.0]], columns=["A"])

    with pytest.raises(ValueError, match="Sample number mismatch"):
        preprocess_simulated_data(_sim_data(p=bad_p))


def test_preprocess_raises_value_error_for_gene_order_mismatch():
    bad_clean = pd.DataFrame([[1.0, 2.0]], columns=["g1", "g0"])

    with pytest.raises(ValueError, match="Gene order mismatch"):
        preprocess_simulated_data(_sim_data(x_clean=bad_clean))


def test_run_training_uses_default_learning_rates_when_config_block_is_missing():
    model, history = run_training(
        model=_model(),
        criterion=object(),
        multi_loader=EmptyLoader(),
        config={
            "enable_alignment": False,
            "epochs": {"pretrain": 0, "finetune": 0},
        },
        device="cpu",
    )

    assert model is not None
    assert history == {"pretrain": [], "finetune": []}


def test_run_training_rejects_empty_target_loader_before_finetune_loop():
    with pytest.raises(ValueError, match="Target adaptation requires"):
        run_training(
            model=_model(),
            criterion=object(),
            multi_loader=EmptyLoader(),
            config={
                "enable_alignment": True,
                "finetune_update_scope": "encoder_last_linear",
                "epochs": {"pretrain": 0, "finetune": 0},
                "learning_rates": {"pretrain": 1e-3, "finetune": 1e-4},
            },
            device="cpu",
        )


def test_run_training_restores_best_pretrain_checkpoint_before_finetune(monkeypatch):
    class NonEmptyTargetLoader:
        def get_dataloader(self, domain, split):
            return [object()] if domain == "target" else []

    model = _model()
    pretrain_epoch = 0
    validation_losses = iter([0.3, 0.1, 0.2, 0.4])

    def fake_pretrain_epoch(model, criterion, source_loader, optimizer, config, device):
        nonlocal pretrain_epoch
        pretrain_epoch += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(pretrain_epoch))
        return float(pretrain_epoch), 0.0, 0.0

    def fake_eval_epoch(model, criterion, val_loader, config, device):
        loss = next(validation_losses)
        return loss, loss

    def fake_finetune_epoch(
        model,
        criterion,
        source_loader,
        target_loader,
        optimizer,
        config,
        device,
    ):
        assert all(
            torch.all(parameter == 2.0)
            for parameter in model.parameters()
        )
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(train_module, "_train_pretrain_epoch", fake_pretrain_epoch)
    monkeypatch.setattr(train_module, "_eval_epoch", fake_eval_epoch)
    monkeypatch.setattr(train_module, "_train_finetune_epoch", fake_finetune_epoch)

    model, history = run_training(
        model=model,
        criterion=object(),
        multi_loader=NonEmptyTargetLoader(),
        config={
            "enable_alignment": True,
            "finetune_update_scope": "encoder_last_linear",
            "epochs": {"pretrain": 3, "finetune": 1},
            "learning_rates": {"pretrain": 1e-3, "finetune": 1e-4},
        },
        device="cpu",
    )

    assert all(torch.all(parameter == 2.0) for parameter in model.parameters())
    assert history == {
        "pretrain": [(1.0, 0.3), (2.0, 0.1), (3.0, 0.2)],
        "finetune": [(0.0, 0.4)],
    }
