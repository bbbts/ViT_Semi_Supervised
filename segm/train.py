#!/usr/bin/env python3
# Path: train.py
import sys
from pathlib import Path
import yaml
import torch
import click
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import suppress
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params
from segm.engine import train_one_epoch, evaluate
from timm.utils import NativeScaler

IGNORE_LABEL = 255  # consistent ignore


@click.command(help="Train a segmentation model with full metric logging")
@click.option("--log-dir", type=str, required=True)
@click.option("--dataset", type=str, default="flame")
@click.option("--im-size", type=int, default=None)
@click.option("--crop-size", type=int, default=None)
@click.option("--window-size", type=int, default=None)
@click.option("--window-stride", type=int, default=None)
@click.option("--backbone", type=str, default="vit_small")
@click.option("--decoder", type=str, default="mask_transformer")
@click.option("--optimizer", type=str, default="sgd")
@click.option("--scheduler", type=str, default="polynomial")
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--dropout", type=float, default=0.0)
@click.option("--drop-path", type=float, default=0.1)
@click.option("--batch-size", type=int, default=None)
@click.option("--epochs", type=int, default=None)
@click.option("-lr", "--learning-rate", type=float, default=None)
@click.option("--normalization", type=str, default=None)
@click.option("--eval-freq", type=int, default=None)
@click.option("--amp/--no-amp", default=False)
@click.option("--resume/--no-resume", default=True)
@click.option("--labeled-ratio", type=float, default=1.0)
@click.option("--pseudo-threshold", type=float, default=0.7)
@click.option("--unsup-weight", type=float, default=0.5)
def main(log_dir, dataset, im_size, crop_size, window_size, window_stride,
         backbone, decoder, optimizer, scheduler, weight_decay,
         dropout, drop_path, batch_size, epochs, learning_rate,
         normalization, eval_freq, amp, resume, labeled_ratio,
         pseudo_threshold, unsup_weight):

    # ---- Setup ----
    ptu.set_gpu_mode(True)
    distributed.init_process()
    device = ptu.device

    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    decoder_cfg = cfg["decoder"].get(decoder, cfg["decoder"]["mask_transformer"])
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    im_size = im_size or dataset_cfg["im_size"]
    crop_size = crop_size or dataset_cfg.get("crop_size", im_size)
    window_size = window_size or dataset_cfg.get("window_size", im_size)
    window_stride = window_stride or dataset_cfg.get("window_stride", im_size)
    crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
    window_stride = (window_stride, window_stride) if isinstance(window_stride, int) else window_stride

    model_cfg.update({
        "image_size": crop_size,
        "backbone": backbone,
        "dropout": dropout,
        "drop_path_rate": drop_path,
    })
    if normalization:
        model_cfg["normalization"] = normalization

    # ---- Dataset ----
    world_batch_size = batch_size or dataset_cfg["batch_size"]
    num_epochs = epochs or dataset_cfg["epochs"]
    lr = learning_rate or dataset_cfg["learning_rate"]
    eval_freq = eval_freq or dataset_cfg.get("eval_freq", 1)
    batch_size = max(1, world_batch_size // max(1, ptu.world_size))

    transform_img = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    dataset_kwargs = dict(
        dataset=dataset,
        image_size=im_size,
        crop_size=crop_size[0],
        batch_size=batch_size,
        normalization=model_cfg.get("normalization", "vit"),
        split="train",
        num_workers=10,
        root=dataset_cfg.get("data_root", dataset_cfg.get("root", None)),
        labeled_ratio=labeled_ratio,
        ssl=(labeled_ratio < 1.0),
        transform_img=transform_img
    )

    print(f"Creating training dataset for {dataset}...")
    train_dataset = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs.update({"split":"validation", "batch_size":1, "ssl":False})
    val_dataset = create_dataset(val_kwargs)

    # ---- Collate function ----
    def semi_supervised_collate_fn(batch):
        images, masks, ids, is_labeled = [], [], [], []
        for sample in batch:
            images.append(sample["image"])
            m = sample.get("mask", None)
            ids.append(sample.get("id", None))
            is_labeled.append(sample.get("is_labeled", False))
            masks.append(m)

        images = torch.stack(images, dim=0)
        B, C, H, W = images.shape
        processed_masks = []
        for m in masks:
            if m is None:
                processed_masks.append(torch.full((H, W), IGNORE_LABEL, dtype=torch.long))
            else:
                if not isinstance(m, torch.Tensor):
                    m = torch.from_numpy(m)
                m = m.long()
                processed_masks.append(m)
        masks = torch.stack(processed_masks, dim=0)
        return {"image": images, "mask": masks, "id": ids, "is_labeled": is_labeled}

    def make_loader(ds, bs, shuffle=True):
        sampler = DistributedSampler(ds, shuffle=shuffle) if ptu.distributed else None
        return DataLoader(ds, batch_size=bs, shuffle=(sampler is None) and shuffle,
                          sampler=sampler, num_workers=10, pin_memory=True,
                          collate_fn=semi_supervised_collate_fn)

    train_loader = make_loader(train_dataset, batch_size, True)
    val_loader = make_loader(val_dataset, 1, False)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # ---- Model ----
    n_cls = train_dataset.n_cls
    model_cfg["n_cls"] = n_cls
    model = create_segmenter(model_cfg).to(device)

    # ---- Optimizer & Scheduler ----
    iter_max = len(train_loader) * num_epochs
    opt_args = type('', (), {})()
    params = dict(
        opt=optimizer, lr=lr, weight_decay=weight_decay, momentum=0.9,
        sched=scheduler, epochs=num_epochs, min_lr=1e-5, poly_power=0.9,
        poly_step_size=1, iter_max=iter_max, iter_warmup=0.0)
    for k, v in params.items(): setattr(opt_args, k, v)
    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)

    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # ---- Resume ----
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = log_dir / "checkpoint.pth"
    csv_path = log_dir / "training_metrics.csv"
    losses_path = log_dir / "losses.npz"

    # load previous losses if exist
    train_ce_hist = []
    train_dice_hist = []
    train_sup_hist = []
    train_unsup_hist = []
    train_total_hist = []
    val_hist = []

    if resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if loss_scaler and "loss_scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["loss_scaler"])

    if losses_path.exists():
        try:
            d = np.load(losses_path, allow_pickle=True)
            train_ce_hist = d.get("train_ce_hist", []).tolist()
            train_dice_hist = d.get("train_dice_hist", []).tolist()
            train_sup_hist = d.get("train_sup_hist", []).tolist()
            train_unsup_hist = d.get("train_unsup_hist", []).tolist()
            train_total_hist = d.get("train_total_hist", []).tolist()
            val_hist = d.get("val_hist", []).tolist()
        except Exception:
            pass

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # ---- Save config ----
    with open(log_dir / "variant.yml", "w") as f:
        yaml.dump({
            "dataset_kwargs": dataset_kwargs,
            "net_kwargs": model_cfg,
            "optimizer_kwargs": params,
            "amp": amp,
            "log_dir": str(log_dir),
            "inference_kwargs": {"window_size": window_size[0], "window_stride": window_stride[0]}
        }, f)

    print(f"Encoder params: {num_params(model.encoder)}, Decoder params: {num_params(model.decoder)}")

    # ---- Initialize CSV ----
    if not csv_path.exists():
        header = ["epoch", "CE_loss", "Dice_loss", "Sup_loss", "Unsup_loss", "Total_loss", "Val_loss",
                  "Pixel_Acc", "Mean_Acc", "Mean_IoU", "FWIoU"]
        header += [f"F1_class{i}" for i in range(n_cls)]
        header += [f"Precision_class{i}" for i in range(n_cls)]
        header += [f"Recall_class{i}" for i in range(n_cls)]
        header += [f"IoU_class{i}" for i in range(n_cls)]
        header += [f"PerClassDice_{i}" for i in range(n_cls)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # ---- Training loop ----
    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_logger = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler, epoch,
            amp_autocast, loss_scaler,
            semi_supervised=(labeled_ratio < 1.0),
            pseudo_thresh=pseudo_threshold,
            unsup_weight=unsup_weight
        )

        CE_loss = train_logger.get("CE_loss", float("nan"))
        Dice_loss = train_logger.get("Dice_loss", float("nan"))
        Sup_loss = train_logger.get("Sup_loss", float("nan"))
        Unsup_loss = train_logger.get("Unsup_loss", float("nan"))
        Total_loss = train_logger.get("Total_loss", float("nan"))

        train_ce_hist.append(CE_loss)
        train_dice_hist.append(Dice_loss)
        train_sup_hist.append(Sup_loss)
        train_unsup_hist.append(Unsup_loss)
        train_total_hist.append(Total_loss)

        # ---- Evaluate ----
        eval_logger = {}
        val_loss = float("nan")
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:
            val_seg_gt = {}
            for idx in range(len(val_loader.dataset)):
                item = val_loader.dataset[idx]
                mask = item.get("mask", None)
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask
                file_id = item.get("id", f"img_{idx}")
                file_id = Path(file_id).stem
                val_seg_gt[file_id] = mask_np

            val_loss, eval_logger = evaluate(model, val_loader, val_seg_gt,
                                             window_size, window_stride, amp_autocast,
                                             print_per_class=True)
            val_hist.append(val_loss)

        # ---- Save checkpoint & CSV row (only main rank) ----
        if ptu.dist_rank == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            if loss_scaler is not None:
                ckpt["loss_scaler"] = loss_scaler.state_dict()
            torch.save(ckpt, ckpt_path)

            def _safe_list(key):
                v = eval_logger.get(key, [])
                if v is None:
                    v = []
                v = list(v)
                if len(v) < n_cls:
                    v += [float("nan")] * (n_cls - len(v))
                return v[:n_cls]

            row = [
                epoch, CE_loss, Dice_loss, Sup_loss, Unsup_loss, Total_loss, val_loss,
                eval_logger.get("Pixel_Acc", float("nan")),
                eval_logger.get("Mean_Acc", float("nan")),
                eval_logger.get("Mean_IoU", float("nan")),
                eval_logger.get("FWIoU", float("nan")),
            ]

            for metric_name in ["F1_per_class", "Precision_per_class", "Recall_per_class",
                                "IoU_per_class", "PerClassDice"]:
                row += _safe_list(metric_name)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # ---- Save losses ----
            try:
                np.savez_compressed(
                    str(losses_path),
                    train_ce_hist=np.array(train_ce_hist, dtype=np.float32),
                    train_dice_hist=np.array(train_dice_hist, dtype=np.float32),
                    train_sup_hist=np.array(train_sup_hist, dtype=np.float32),
                    train_unsup_hist=np.array(train_unsup_hist, dtype=np.float32),
                    train_total_hist=np.array(train_total_hist, dtype=np.float32),
                    val_hist=np.array(val_hist, dtype=np.float32),
                )
            except Exception as e:
                print("Warning: could not save losses.npz:", e)

            # ---- Plot losses (linear scale with nice y-axis ticks) ----
            # ---- Plot losses (linear scale with nice y-axis ticks) ----
            try:
                epochs_x = np.arange(1, len(train_total_hist) + 1)
                plt.figure(figsize=(12, 6))
            
                loss_dict = {
                    "Train CE": train_ce_hist,
                    "Train Dice": train_dice_hist,
                    "Train Sup": train_sup_hist,
                    "Train Unsup": train_unsup_hist,
                    "Train Total": train_total_hist
                }
            
                for name, hist in loss_dict.items():
                    if len(hist) > 0:
                        plt.plot(epochs_x, hist, label=name, marker='o', markersize=6, linewidth=2)
            
                if len(val_hist) > 0:
                    plt.plot(np.arange(1, len(val_hist) + 1), val_hist, label="Val Loss",
                             marker='x', markersize=8, linewidth=2, linestyle='--', color='black')
            
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Losses")
                plt.legend()
                plt.grid(True)
            
                # dynamic y-axis limits
                all_losses = []
                for h in loss_dict.values():
                    all_losses.extend(h)
                if len(val_hist) > 0:
                    all_losses.extend(val_hist)
            
                min_y = max(0, min(all_losses) * 0.9)
                max_y = max(all_losses) * 1.1
                plt.ylim(min_y, max_y)
            
                plt.tight_layout()
                plt.savefig(log_dir / "training_losses.png")
                plt.close()
            except Exception as e:
                print("Warning: could not save loss plot:", e)

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(0)


if __name__ == "__main__":
    main()

