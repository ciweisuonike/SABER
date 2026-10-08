import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ABLATION_DIFFERENCES = {
    "simulation.enable_gene_selection",
    "simulation.enable_twoview",
    "simulation.enable_confidence_gene_dropout",
    "training.enable_alignment",
    "loss.lambda_local_mmd",
    "loss.lambda_consistency_source",
    "loss.lambda_consistency_latent",
}


def _flatten(config, prefix=""):
    values = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten(value, path))
        else:
            values[path] = value
    return values


def _normalize(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _normalize(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    return value


def _function_defaults(relative_path, function_name, class_name=None):
    tree = ast.parse((ROOT / relative_path).read_text())
    nodes = tree.body
    if class_name is not None:
        class_node = next(
            node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = class_node.body
    function = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    arguments = function.args.args
    defaults = function.args.defaults
    return {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(arguments[len(arguments) - len(defaults) :], defaults)
    }


def _get_defaults(relative_path):
    source = (ROOT / relative_path).read_text()
    tree = ast.parse(source)
    defaults = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        try:
            default = ast.literal_eval(node.args[1])
        except (ValueError, TypeError):
            continue
        defaults.setdefault(node.args[0].value, set()).add(_normalize(default))
    return defaults


def _named_dict(relative_path, variable_name):
    tree = ast.parse((ROOT / relative_path).read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_configs_use_top_level_reproducibility_seed():
    for config_path in (ROOT / "configs").glob("config*.yaml"):
        config = yaml.safe_load(config_path.read_text())

        assert "reproducibility" in config, config_path.name
        assert config["reproducibility"]["seed"] == 42, config_path.name
        assert config["reproducibility"]["deterministic"] is False, config_path.name
        assert config["reference_subset"]["n_cells"] is None, config_path.name
        assert "random_state" not in config.get("simulation", {}), config_path.name


def test_ablation_configs_only_change_declared_switches():
    configs = {
        path.name: _flatten(yaml.safe_load(path.read_text()))
        for path in sorted((ROOT / "configs").glob("config*.yaml"))
    }
    assert len(configs) == 9
    canonical = configs["config.yaml"]
    for name, config in configs.items():
        assert set(config) == set(canonical), name
        differences = {key for key in canonical if config[key] != canonical[key]}
        assert differences <= ALLOWED_ABLATION_DIFFERENCES, (name, differences)
        assert "simulation.stability_shift_scale" not in config, name
        assert "simulation.stability_zero_scale" not in config, name


def test_code_defaults_match_canonical_config():
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    simulation = config["simulation"]

    bulk_defaults = _function_defaults(
        "saber/simulate/bulk.py", "simulate_bulk_with_batch_effect"
    )
    for key, value in bulk_defaults.items():
        if key in simulation:
            assert _normalize(value) == _normalize(simulation[key]), key

    selection_defaults = _named_dict(
        "saber/simulate/gene_selection.py", "GENE_SELECTION_DEFAULTS"
    )
    for key in (
        "top_genes",
        "enable_gene_selection",
        "marker_genes_per_celltype",
        "min_pct_in_celltype",
        "min_logfc",
        "enable_target_stability_filter",
        "source_zero_threshold",
        "target_zero_threshold",
        "target_min_variance_quantile",
        "source_target_shift_threshold",
        "target_high_expression_quantile",
        "stability_soft_threshold",
        "stability_gamma",
        "stability_pseudobulk_samples",
        "reference_col",
        "enable_reference_stability_filter",
        "reference_min_cells_per_group",
        "reference_celltype_quantile",
        "reference_hard_filter_quantile",
        "reference_stability_soft_threshold",
        "reference_stability_gamma",
    ):
        assert _normalize(selection_defaults[key]) == _normalize(simulation[key]), key

    get_expectations = {
        "saber/models/saber.py": {
            "prop_dim": config["model"]["prop_dim"],
            "encoder_hidden_dim": config["model"]["encoder_hidden_dim"],
            "dropout_rate": config["model"]["dropout_rate"],
            "proportion_head_hidden_dim": config["model"]["proportion_head_hidden_dim"],
        },
        "saber/pipeline/simulate.py": {
            "enable_gene_selection": simulation["enable_gene_selection"],
            "enable_twoview": simulation["enable_twoview"],
            "enable_confidence_gene_dropout": simulation["enable_confidence_gene_dropout"],
        },
        "saber/simulate/gene_selection.py": {
            "top_genes": simulation["top_genes"],
            "enable_gene_selection": simulation["enable_gene_selection"],
            "celltype_col": simulation["celltype_col"],
            "marker_genes_per_celltype": simulation["marker_genes_per_celltype"],
            "min_pct_in_celltype": simulation["min_pct_in_celltype"],
            "min_logfc": simulation["min_logfc"],
            "enable_target_stability_filter": simulation["enable_target_stability_filter"],
            "source_zero_threshold": simulation["source_zero_threshold"],
            "target_zero_threshold": simulation["target_zero_threshold"],
            "target_min_variance_quantile": simulation["target_min_variance_quantile"],
            "source_target_shift_threshold": simulation["source_target_shift_threshold"],
            "target_high_expression_quantile": simulation["target_high_expression_quantile"],
            "stability_soft_threshold": simulation["stability_soft_threshold"],
            "stability_gamma": simulation["stability_gamma"],
            "stability_pseudobulk_samples": simulation["stability_pseudobulk_samples"],
            "reference_col": simulation["reference_col"],
            "enable_reference_stability_filter": simulation[
                "enable_reference_stability_filter"
            ],
            "reference_min_cells_per_group": simulation["reference_min_cells_per_group"],
            "reference_celltype_quantile": simulation["reference_celltype_quantile"],
            "reference_hard_filter_quantile": simulation["reference_hard_filter_quantile"],
            "reference_stability_soft_threshold": simulation[
                "reference_stability_soft_threshold"
            ],
            "reference_stability_gamma": simulation["reference_stability_gamma"],
        },
        "saber/pipeline/prop_train.py": {
            "celltype_col": simulation["celltype_col"],
            "enable_gene_selection": simulation["enable_gene_selection"],
            "enable_twoview": simulation["enable_twoview"],
            "enable_confidence_gene_dropout": simulation["enable_confidence_gene_dropout"],
            "batch_size": config["dataloader"]["batch_size"],
            "train_ratio": config["dataloader"]["train_ratio"],
            "prop_dim": config["model"]["prop_dim"],
            "deterministic": config["reproducibility"]["deterministic"],
        },
        "saber/pipeline/train.py": {
            "lambda_prop": config["loss"]["lambda_prop"],
            "lambda_local_mmd": config["loss"]["lambda_local_mmd"],
            "lambda_consistency_source": config["loss"]["lambda_consistency_source"],
            "lambda_consistency_latent": config["loss"]["lambda_consistency_latent"],
            "local_mmd_tau": config["training"]["local_mmd_tau"],
            "enable_alignment": config["training"]["enable_alignment"],
            "finetune_update_scope": config["training"]["finetune_update_scope"],
            "pretrain": {
                config["training"]["epochs"]["pretrain"],
                config["training"]["learning_rates"]["pretrain"],
            },
            "finetune": {
                config["training"]["epochs"]["finetune"],
                config["training"]["learning_rates"]["finetune"],
            },
        },
    }
    for relative_path, expectations in get_expectations.items():
        defaults = _get_defaults(relative_path)
        for key, value in expectations.items():
            expected = value if isinstance(value, set) else {_normalize(value)}
            assert defaults[key] == expected, (relative_path, key)

    dataloader_defaults = _function_defaults(
        "saber/data/dataloader.py", "__init__", class_name="MultiStageDataLoader"
    )
    assert dataloader_defaults["batch_size"] == config["dataloader"]["batch_size"]
    assert dataloader_defaults["train_ratio"] == config["dataloader"]["train_ratio"]

    encoder_defaults = _function_defaults(
        "saber/models/saber.py", "__init__", class_name="ProportionEncoder"
    )
    head_defaults = _function_defaults(
        "saber/models/saber.py", "__init__", class_name="ProportionHead"
    )
    assert encoder_defaults["hidden_dim"] == config["model"]["encoder_hidden_dim"]
    assert encoder_defaults["dropout_rate"] == config["model"]["dropout_rate"]
    assert head_defaults["hidden_dim"] == config["model"]["proportion_head_hidden_dim"]
    assert head_defaults["dropout_rate"] == config["model"]["dropout_rate"]
