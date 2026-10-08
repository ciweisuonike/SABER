import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if "saber" not in sys.modules:
    saber_pkg = types.ModuleType("saber")
    saber_pkg.__path__ = [str(ROOT / "saber")]
    sys.modules["saber"] = saber_pkg

import saber.cli as cli_module


class FakeModel:
    def state_dict(self):
        return {"weight": 1}


def test_cli_help_exposes_prop_commands(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["saber", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "prop" in output
    assert "prop-predict" in output


@pytest.mark.parametrize("command", ["prop", "prop-predict"])
def test_prop_command_help_uses_prop_dir_not_out(monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", ["saber", command, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--prop-dir" in output
    removed_output_arg = "--" + "out"
    assert removed_output_arg not in output


def _install_fake_scanpy(monkeypatch):
    scanpy_stub = types.ModuleType("scanpy")
    scanpy_stub.read_h5ad = lambda path: object()
    monkeypatch.setitem(sys.modules, "scanpy", scanpy_stub)


def _install_fake_pipeline(
    monkeypatch,
    prop_train_runner=None,
    predict_bulk_with_model=None,
    run_predict=None,
):
    pipeline_pkg = types.ModuleType("saber.pipeline")
    pipeline_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "saber.pipeline", pipeline_pkg)

    if prop_train_runner is not None:
        prop_train_module = types.ModuleType("saber.pipeline.prop_train")
        prop_train_module.run_prop_train_from_scRNAseq = prop_train_runner
        monkeypatch.setitem(sys.modules, "saber.pipeline.prop_train", prop_train_module)

    if predict_bulk_with_model is not None or run_predict is not None:
        predict_module = types.ModuleType("saber.pipeline.predict")
        if predict_bulk_with_model is not None:
            predict_module.predict_bulk_with_model = predict_bulk_with_model
        if run_predict is not None:
            predict_module.run_predict = run_predict
        monkeypatch.setitem(sys.modules, "saber.pipeline.predict", predict_module)


def _fake_prop_train(captured):
    def fake_run_prop_train_from_scRNAseq(adata, config, bulk, device):
        captured["train_device"] = device
        captured["train_bulk_index"] = bulk.index.tolist()
        return (
            FakeModel(),
            {"pretrain": [], "finetune": []},
            pd.DataFrame([[1.0, 2.0]], index=["sample0"], columns=["g0", "g1"]),
            ["g0", "g1"],
            ["celltype_a"],
            {"model": {}},
        )

    return fake_run_prop_train_from_scRNAseq


def test_prop_without_model_dir_only_writes_proportions(monkeypatch, tmp_path):
    captured = {}
    config_path = tmp_path / "config.yaml"
    bulk_path = tmp_path / "bulk.csv"
    adata_path = tmp_path / "adata.h5ad"
    prop_dir = tmp_path / "prop"

    config_path.write_text(yaml.safe_dump({"simulation": {}}))
    bulk_path.write_text(",g0,g1\nsample0,1,2\n")
    adata_path.write_text("")

    def fake_predict_bulk_with_model(model, genes, celltypes, bulk_file, device):
        captured["predict_model"] = model
        captured["predict_genes"] = genes
        captured["predict_celltypes"] = celltypes
        captured["predict_bulk_file"] = bulk_file
        captured["predict_device"] = device
        return pd.DataFrame([[1.0]], index=["sample0"], columns=["celltype_a"])

    _install_fake_scanpy(monkeypatch)
    _install_fake_pipeline(
        monkeypatch,
        prop_train_runner=_fake_prop_train(captured),
        predict_bulk_with_model=fake_predict_bulk_with_model,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saber",
            "prop",
            "--adata",
            str(adata_path),
            "--config",
            str(config_path),
            "--bulk",
            str(bulk_path),
            "--prop-dir",
            str(prop_dir),
            "--device",
            "cpu",
            "--prefix",
            "SABER",
        ],
    )

    cli_module.main()

    assert captured["train_device"] == "cpu"
    assert captured["train_bulk_index"] == ["sample0"]
    assert isinstance(captured["predict_model"], FakeModel)
    assert captured["predict_genes"] == ["g0", "g1"]
    assert captured["predict_celltypes"] == ["celltype_a"]
    assert captured["predict_bulk_file"] == str(bulk_path)
    assert captured["predict_device"] == "cpu"
    assert not (tmp_path / "model.pt").exists()
    assert not (tmp_path / "bulk_align.csv").exists()
    assert (prop_dir / "SABER.csv").exists()


def test_prop_with_model_dir_saves_model_artifacts(monkeypatch, tmp_path):
    captured = {}
    config_path = tmp_path / "config.yaml"
    bulk_path = tmp_path / "bulk.csv"
    adata_path = tmp_path / "adata.h5ad"
    model_dir = tmp_path / "model"
    prop_dir = tmp_path / "prop"

    config_path.write_text(yaml.safe_dump({"simulation": {}}))
    bulk_path.write_text(",g0,g1\nsample0,1,2\n")
    adata_path.write_text("")

    def fake_predict_bulk_with_model(model, genes, celltypes, bulk_file, device):
        captured["predict_bulk_file"] = bulk_file
        return pd.DataFrame([[1.0]], index=["sample0"], columns=["celltype_a"])

    _install_fake_scanpy(monkeypatch)
    _install_fake_pipeline(
        monkeypatch,
        prop_train_runner=_fake_prop_train(captured),
        predict_bulk_with_model=fake_predict_bulk_with_model,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saber",
            "prop",
            "--adata",
            str(adata_path),
            "--config",
            str(config_path),
            "--bulk",
            str(bulk_path),
            "--prop-dir",
            str(prop_dir),
            "--model-dir",
            str(model_dir),
            "--device",
            "cpu",
            "--prefix",
            "SABER",
        ],
    )

    cli_module.main()

    assert captured["predict_bulk_file"] == str(bulk_path)
    assert (model_dir / "model.pt").exists()
    assert (model_dir / "bulk_align.csv").exists()
    assert (prop_dir / "SABER.csv").exists()


def test_prop_predict_dispatches_existing_prediction(monkeypatch, tmp_path):
    captured = {}
    prop_dir = tmp_path / "prop"

    def fake_run_predict(model_path, bulk_file, device):
        captured["model_path"] = model_path
        captured["bulk_file"] = bulk_file
        captured["device"] = device
        return pd.DataFrame([[0.25, 0.75]], index=["sample0"], columns=["a", "b"])

    _install_fake_pipeline(monkeypatch, run_predict=fake_run_predict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saber",
            "prop-predict",
            "--model",
            "model.pt",
            "--bulk",
            "bulk.csv",
            "--prop-dir",
            str(prop_dir),
            "--device",
            "cpu",
            "--prefix",
            "pred",
        ],
    )

    cli_module.main()

    assert captured == {
        "model_path": "model.pt",
        "bulk_file": "bulk.csv",
        "device": "cpu",
    }
    assert (prop_dir / "pred.csv").exists()
