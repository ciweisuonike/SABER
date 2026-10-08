import copy
import itertools

import torch
from tqdm import tqdm


def _get_loss_weights(config):
    return {
        "lambda_prop": config.get("lambda_prop", 1.0),
        "lambda_local_mmd": config.get("lambda_local_mmd", 0.01),
        "lambda_consistency_source": config.get("lambda_consistency_source", 0.1),
        "lambda_consistency_latent": config.get("lambda_consistency_latent", 0.01),
    }


def _unpack_source_batch(batch_data, device):
    if len(batch_data) == 3:
        x_clean, x_obs, p_src = [b.to(device) for b in batch_data]
        return x_clean, x_obs, p_src, True
    if len(batch_data) == 2:
        x_obs, p_src = [b.to(device) for b in batch_data]
        return x_obs, x_obs, p_src, False
    raise ValueError(f"Unexpected source batch with {len(batch_data)} tensors.")


def _train_pretrain_epoch(model, criterion, source_loader, optimizer, config, device):
    model.train()
    total_loss, total_prop, total_source_cons = 0.0, 0.0, 0.0
    weights = _get_loss_weights(config)
    lambda_prop = weights["lambda_prop"]
    lambda_source_cons = weights["lambda_consistency_source"]
    lambda_latent = weights["lambda_consistency_latent"]
    for batch_data in tqdm(source_loader, desc="Training [Pretrain]", leave=False):
        x_clean, x_obs, p_src, has_clean_view = _unpack_source_batch(batch_data, device)

        p_obs, _scores_obs, z_obs = model(x_obs)
        prop_obs = criterion.get_task_loss(p_obs, p_src)

        source_cons = torch.zeros((), device=device)
        if has_clean_view:
            p_clean, _scores_clean, z_clean = model(x_clean)
            prop_clean = criterion.get_task_loss(p_clean, p_src)
            prop_loss = 0.5 * (prop_obs + prop_clean)
            if lambda_source_cons > 0:
                source_cons = criterion.get_source_consistency_loss(
                    p_clean,
                    p_obs,
                    z_clean,
                    z_obs,
                    lambda_latent,
                )
        else:
            prop_loss = prop_obs

        loss = (
            lambda_prop * prop_loss
            + lambda_source_cons * source_cons
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_prop += prop_loss.item()
        total_source_cons += source_cons.item()

    num_batches = max(len(source_loader), 1)
    return (
        total_loss / num_batches,
        total_prop / num_batches,
        total_source_cons / num_batches,
    )


def _train_finetune_epoch(model, criterion, source_loader, target_loader, optimizer, config, device):
    model.train()
    total_loss, total_prop, total_local_mmd = 0.0, 0.0, 0.0
    weights = _get_loss_weights(config)
    lambda_prop = weights["lambda_prop"]
    lambda_local_mmd = weights["lambda_local_mmd"]
    local_mmd_tau = float(config.get("local_mmd_tau", 10.0))

    target_iter = itertools.cycle(target_loader)

    for src_batch in tqdm(source_loader, desc="Training [Finetune/local MMD]", leave=False):
        tgt_batch = next(target_iter)

        _x_clean, x_src_obs, p_src, _has_clean_view = _unpack_source_batch(src_batch, device)
        x_tgt, _ = [b.to(device) for b in tgt_batch]

        p_tgt, _, z_tgt_prop = model(x_tgt)

        p_src_pred, _scores_src, z_src_prop = model(x_src_obs)
        prop_loss = criterion.get_task_loss(p_src_pred, p_src)
        task_loss = lambda_prop * prop_loss

        local_mmd_loss = criterion.get_conditional_local_mmd_loss(
            z_src_prop,
            z_tgt_prop,
            p_src,
            p_tgt,
            tau=local_mmd_tau,
        )

        loss = task_loss + lambda_local_mmd * local_mmd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_prop += prop_loss.item()
        total_local_mmd += local_mmd_loss.item()

    num_batches = max(len(source_loader), 1)
    return (
        total_loss / num_batches,
        total_prop / num_batches,
        total_local_mmd / num_batches,
    )


def _eval_epoch(model, criterion, val_loader, config, device):
    model.eval()
    total_loss, total_prop = 0.0, 0.0
    lambda_prop = _get_loss_weights(config)["lambda_prop"]

    with torch.no_grad():
        for batch_data in val_loader:
            _x_clean, x_obs, p_val, _has_clean_view = _unpack_source_batch(batch_data, device)
            p_pred, _, _ = model(x_obs)
            prop_loss = criterion.get_task_loss(p_pred, p_val)
            loss = lambda_prop * prop_loss

            total_loss += loss.item()
            total_prop += prop_loss.item()

    num_batches = max(len(val_loader), 1)
    return total_loss / num_batches, total_prop / num_batches


def run_training(model, criterion, multi_loader, config, device):
    history = {"pretrain": [], "finetune": []}
    enable_alignment = config.get("enable_alignment", True)
    learning_rates = config.get("learning_rates", {})

    print("\n" + "=" * 80)
    print("STAGE 1: Source Domain Proportion Encoder Pretraining")
    print("=" * 80)

    model.unfreeze_encoder()
    model.unfreeze_proportion_head()

    opt_pretrain = torch.optim.Adam(
        model.get_main_parameters(),
        lr=learning_rates.get("pretrain", 1e-3),
    )

    src_train_loader = multi_loader.get_dataloader("source", "train")
    src_val_loader = multi_loader.get_dataloader("source", "val")
    best_pretrain_val = float("inf")
    best_pretrain_epoch = None
    best_pretrain_state = None
    for epoch in range(config["epochs"].get("pretrain", 40)):
        train_l, tr_prop, tr_source_cons = _train_pretrain_epoch(
            model, criterion, src_train_loader, opt_pretrain, config, device
        )
        val_l, val_prop = _eval_epoch(model, criterion, src_val_loader, config, device)

        history["pretrain"].append((train_l, val_l))
        if val_l < best_pretrain_val:
            best_pretrain_val = val_l
            best_pretrain_epoch = epoch + 1
            best_pretrain_state = copy.deepcopy(model.state_dict())
        print(
            f"[Ep {epoch + 1:02d}] "
            f"TRAIN Total: {train_l:.4f} (Prop: {tr_prop:.4f}, "
            f"SrcCons: {tr_source_cons:.4f}) | "
            f"VAL Total: {val_l:.4f} (Prop: {val_prop:.4f})"
        )

    if best_pretrain_state is not None:
        model.load_state_dict(best_pretrain_state)
        del best_pretrain_state
        print(
            "[SABER] Restored best pretrain checkpoint from "
            f"epoch {best_pretrain_epoch} (validation loss={best_pretrain_val:.4f})."
        )

    print("\n" + "=" * 80)
    if not enable_alignment:
        print("STAGE 2 SKIPPED: target adaptation is disabled via config (enable_alignment: false)")
        print("=" * 80)
        return model, history

    print("STAGE 2: Target conditional local MMD Adaptation on Proportion Latents")
    print("=" * 80)

    finetune_scope = config.get("finetune_update_scope", "encoder_last_linear")
    if finetune_scope != "encoder_last_linear":
        raise ValueError(
            "Unsupported finetune_update_scope "
            f"'{finetune_scope}'. Supported scopes: encoder_last_linear."
        )
    trainable_params = model.configure_encoder_last_linear_finetune()
    trainable_count = sum(param.numel() for param in trainable_params)
    if trainable_count == 0:
        raise RuntimeError(
            f"No trainable parameters found for finetune_update_scope='{finetune_scope}'."
        )
    print(
        "[SABER] Finetune trainable scope: "
        f"{finetune_scope} ({trainable_count} parameters)"
    )

    tgt_train_loader = multi_loader.get_dataloader("target", "train")
    if len(tgt_train_loader) == 0:
        raise ValueError("Target adaptation requires at least one target sample.")

    opt_finetune = torch.optim.Adam(
        trainable_params,
        lr=learning_rates.get("finetune", 1e-4),
    )

    for epoch in range(config["epochs"].get("finetune", 10)):
        train_l, tr_prop, tr_local_mmd = _train_finetune_epoch(
            model, criterion, src_train_loader, tgt_train_loader, opt_finetune, config, device
        )
        val_l, val_prop = _eval_epoch(model, criterion, src_val_loader, config, device)

        history["finetune"].append((train_l, val_l))
        print(
            f"[Ep {epoch + 1:02d}] "
            f"TRAIN Total: {train_l:.4f} (Prop: {tr_prop:.4f}, "
            f"LocalMMD: {tr_local_mmd:.4f}) | "
            f"VAL Total: {val_l:.4f} (Prop: {val_prop:.4f})"
        )

    return model, history
