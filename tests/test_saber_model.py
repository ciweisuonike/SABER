import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if "saber" not in sys.modules:
    saber_pkg = types.ModuleType("saber")
    saber_pkg.__path__ = [str(ROOT / "saber")]
    sys.modules["saber"] = saber_pkg

from saber.models.saber import SABER

train_spec = importlib.util.spec_from_file_location(
    "saber_pipeline_train_for_test",
    ROOT / "saber" / "pipeline" / "train.py",
)
train_module = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train_module)
run_training = train_module.run_training


def _model():
    return SABER(
        {
            "num_cell_types": 3,
            "gene_dim": 5,
            "prop_dim": 4,
            "encoder_hidden_dim": 6,
            "proportion_head_hidden_dim": 7,
            "dropout_rate": 0.0,
            "checkpoint_metadata": "ignored",
        }
    )


def test_encoder_last_linear_finetune_only_unfreezes_last_encoder_layer():
    model = _model()

    returned_params = model.configure_encoder_last_linear_finetune()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    last_layer_params = list(model.encoder.network[-1].parameters())

    assert {id(param) for param in returned_params} == {
        id(param) for param in last_layer_params
    }
    assert {id(param) for param in trainable_params} == {
        id(param) for param in last_layer_params
    }
    assert len(returned_params) == len(trainable_params)
    assert all(not param.requires_grad for param in model.proportion_head.parameters())


def test_saber_model_does_not_promote_checkpoint_metadata_to_attribute():
    assert not hasattr(_model(), "checkpoint_metadata")


def test_run_training_rejects_unsupported_finetune_scope():
    class EmptyLoader:
        def get_dataloader(self, domain, split):
            return []

    with pytest.raises(ValueError, match="Unsupported finetune_update_scope"):
        run_training(
            model=_model(),
            criterion=object(),
            multi_loader=EmptyLoader(),
            config={
                "enable_alignment": True,
                "finetune_update_scope": "encoder_full",
                "epochs": {"pretrain": 0, "finetune": 0},
                "learning_rates": {"pretrain": 1e-3, "finetune": 1e-4},
            },
            device="cpu",
        )
