# Path: segm/engine.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import suppress

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data
import segm.utils.torch as ptu

IGNORE_LABEL = 255
CE = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)


def _safe_onehot(tensor, n_cls):
    """Return one-hot [B, C, H, W] for integer tensor [B, H, W]."""
    return F.one_hot(tensor.clamp(0, n_cls - 1), num_classes=n_cls).permute(0, 3, 1, 2).float()


def dice_per_class_from_preds(pred, target, n_cls, ignore_index=IGNORE_LABEL, eps=1e-10):
    """
    pred: [B, H, W] ints
    target: [B, H, W] ints
    returns: per-class dice averaged across batch (array length n_cls)
    """
    batch = pred.shape[0]
    dice_accum = np.zeros(n_cls, dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.float32)

    for b in range(batch):
        p = pred[b]
        t = target[b]
        valid_mask = (t != ignore_index)
        if valid_mask.sum() == 0:
            continue
        p = p[valid_mask]
        t = t[valid_mask]
        for c in range(n_cls):
            p_c = (p == c).astype(np.int32)
            t_c = (t == c).astype(np.int32)
            inter = (p_c & t_c).sum()
            denom = p_c.sum() + t_c.sum()
            dice = (2.0 * inter) / (denom + eps) if denom > 0 else 1.0
            dice_accum[c] += dice
            counts[c] += 1.0

    # avoid division by zero
    counts[counts == 0] = 1.0
    return (dice_accum / counts)


def compute_segmentation_metrics(preds_dict, gts_dict, n_cls, ignore_index=IGNORE_LABEL):
    """
    preds_dict: {id: 2D np array (H,W) predicted labels}
    gts_dict:   {id: 2D np array (H,W) gt labels}
    returns metrics dict with overall & per-class metrics (numpy arrays)
    """
    eps = 1e-10
    total_pixels = 0
    correct_pixels = 0

    tp = np.zeros(n_cls, dtype=np.float64)
    fp = np.zeros(n_cls, dtype=np.float64)
    fn = np.zeros(n_cls, dtype=np.float64)
    gt_count = np.zeros(n_cls, dtype=np.float64)

    for k, pred in preds_dict.items():
        if k not in gts_dict:
            continue
        gt = np.asarray(gts_dict[k], dtype=np.int64)
        pred = np.asarray(pred, dtype=np.int64)

        # if shapes mismatch, resize pred using nearest (use numpy)
        if pred.shape != gt.shape:
            # simple nearest resize via torch (keeps no extra deps)
            p_t = torch.from_numpy(pred).astype(torch.int64).unsqueeze(0).unsqueeze(0).float()
            p_resized = F.interpolate(p_t, size=gt.shape, mode='nearest').squeeze().long().numpy()
            pred = p_resized

        mask = (gt != ignore_index)
        pred = pred[mask]
        gt = gt[mask]

        total_pixels += gt.size
        correct_pixels += (pred == gt).sum()

        for c in range(n_cls):
            pred_c = (pred == c)
            gt_c = (gt == c)
            inter = np.logical_and(pred_c, gt_c).sum()
            tp[c] += inter
            fp[c] += pred_c.sum() - inter
            fn[c] += gt_c.sum() - inter
            gt_count[c] += gt_c.sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    pixel_acc = correct_pixels / (total_pixels + eps)
    per_class_acc = np.zeros(n_cls, dtype=np.float64)
    for c in range(n_cls):
        per_class_acc[c] = tp[c] / (gt_count[c] + eps)
    mean_acc = np.nanmean(per_class_acc)
    mean_iou = np.nanmean(iou)
    fw_iou = (gt_count * iou).sum() / (gt_count.sum() + eps)

    metrics = {
        "PixelAcc": float(pixel_acc),
        "MeanAcc": float(mean_acc),
        "MeanIoU": float(mean_iou),
        "FWIoU": float(fw_iou),
        "Precision_per_class": precision.tolist(),
        "Recall_per_class": recall.tolist(),
        "F1_per_class": f1.tolist(),
        "IoU_per_class": iou.tolist(),
        "GT_count_per_class": gt_count.tolist()
    }
    return metrics


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, epoch,
                    amp_autocast, loss_scaler, semi_supervised=False,
                    pseudo_thresh=0.7, unsup_weight=0.5):
    """
    Train one epoch. Returns dict:
      { 'CE_loss', 'Dice_loss', 'Sup_loss', 'Unsup_loss', 'Total_loss' }
    """
    model.train()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 50

    total_ce = total_dice = total_sup = total_unsup = total_total = 0.0
    num_batches = 0

    n_cls = getattr(train_loader.dataset, "n_cls", None)
    if n_cls is None:
        raise AttributeError("Dataset must define 'n_cls'")

    for batch_idx, batch in enumerate(logger.log_every(train_loader, print_freq, header)):
        imgs = batch["image"].to(ptu.device)
        masks = batch.get("mask", None)
        is_labeled = batch.get("is_labeled", None)

        optimizer.zero_grad()
        with amp_autocast():
            logits = model(imgs)  # [B, C, H, W]

            # prepare masks
            if masks is not None:
                masks = masks.long().to(ptu.device)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            # supervised loss (apply only on labeled pixels)
            sup_loss = torch.tensor(0.0, device=ptu.device)
            if masks is not None:
                # if is_labeled is provided, select labeled samples only
                if is_labeled is not None:
                    labeled_idx = [i for i, flag in enumerate(is_labeled) if bool(flag)]
                    if len(labeled_idx) > 0:
                        logits_l = logits[labeled_idx]
                        masks_l = masks[labeled_idx]
                        # compute CE ignoring IGNORE_LABEL
                        valid_any = (masks_l != IGNORE_LABEL).any()
                        if valid_any:
                            sup_loss = CE(logits_l, masks_l)
                else:
                    # all labeled
                    if (masks != IGNORE_LABEL).any():
                        sup_loss = CE(logits, masks)

            # unsupervised pseudo-label CE
            unsup_loss = torch.tensor(0.0, device=ptu.device)
            if semi_supervised:
                # find unlabeled indices
                unlabeled_idx = []
                if is_labeled is not None:
                    unlabeled_idx = [i for i, flag in enumerate(is_labeled) if not bool(flag)]
                else:
                    # treat masks filled entirely with IGNORE as unlabeled
                    unlabeled_idx = [i for i in range(logits.shape[0]) if masks is None or (masks[i] == IGNORE_LABEL).all()]

                if len(unlabeled_idx) > 0:
                    logits_u = logits[unlabeled_idx]  # [U, C, H, W]
                    with torch.no_grad():
                        probs = torch.softmax(logits_u, dim=1)  # [U, C, H, W]
                        conf, pseudo = probs.max(dim=1)  # conf: [U, H, W], pseudo: [U, H, W]
                        mask_conf = conf >= pseudo_thresh
                        # set low confidence to IGNORE_LABEL
                        pseudo_masked = pseudo.clone()
                        pseudo_masked[~mask_conf] = IGNORE_LABEL

                        # If we have GT masks for these (rare), also avoid overriding valid GT
                        if masks is not None:
                            mask_u = masks[unlabeled_idx]
                            valid_mask_u = mask_u != IGNORE_LABEL
                            pseudo_masked[valid_mask_u] = IGNORE_LABEL

                    # compute CE only on confident pixels
                    # CE from logits_u vs pseudo_masked, using ignore_index will ignore low-confidence pixels
                    if (pseudo_masked != IGNORE_LABEL).any():
                        unsup_loss = CE(logits_u, pseudo_masked)

            total_loss = sup_loss + unsup_weight * unsup_loss

            # compute dice on *labeled* pixels only (for reporting)
            dice_vals = np.zeros(n_cls, dtype=np.float32)
            if masks is not None:
                # gather predictions and labeled masks for labeled samples
                if is_labeled is not None:
                    labeled_idx = [i for i, flag in enumerate(is_labeled) if bool(flag)]
                    if len(labeled_idx) > 0:
                        pred = logits.argmax(1)[labeled_idx].cpu().numpy()
                        gt_mask = masks[labeled_idx].cpu().numpy()
                        dice_vals = dice_per_class_from_preds(pred, gt_mask, n_cls, ignore_index=IGNORE_LABEL)
                else:
                    pred = logits.argmax(1).cpu().numpy()
                    gt_mask = masks.cpu().numpy()
                    dice_vals = dice_per_class_from_preds(pred, gt_mask, n_cls, ignore_index=IGNORE_LABEL)

        # backward
        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer, parameters=model.parameters())
        else:
            total_loss.backward()
            optimizer.step()

        # scheduler step (try safe)
        with suppress(Exception):
            if hasattr(lr_scheduler, "step_update"):
                try:
                    lr_scheduler.step_update(num_updates=epoch)
                except Exception:
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

        # accumulate scalars
        total_ce += float(sup_loss.detach().cpu().item())
        total_sup += float(sup_loss.detach().cpu().item())
        total_unsup += float(unsup_loss.detach().cpu().item())
        total_total += float(total_loss.detach().cpu().item())
        total_dice += float(np.nanmean(dice_vals))
        num_batches += 1

    # average
    if num_batches == 0:
        return {"CE_loss": float("nan"), "Dice_loss": float("nan"),
                "Sup_loss": float("nan"), "Unsup_loss": float("nan"), "Total_loss": float("nan")}

    return {
        "CE_loss": total_ce / num_batches,
        "Dice_loss": total_dice / num_batches,
        "Sup_loss": total_sup / num_batches,
        "Unsup_loss": total_unsup / num_batches,
        "Total_loss": total_total / num_batches
    }


@torch.no_grad()
def evaluate(model, val_loader, val_seg_gt, window_size, window_stride, amp_autocast,
             print_per_class=False):
    """
    Evaluate model. Returns (avg_loss, metrics_dict)
    metrics_dict contains:
      Pixel_Acc, Mean_Acc, Mean_IoU, FWIoU,
      Precision_per_class, Recall_per_class, F1_per_class, IoU_per_class, PerClassDice
    """
    model_eval = model.module if hasattr(model, "module") else model
    n_cls = getattr(val_loader.dataset, "n_cls", None)
    if n_cls is None:
        raise AttributeError("Dataset must define 'n_cls'")

    criterion_eval = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    val_preds = {}
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        imgs = batch["image"].to(ptu.device)
        masks = batch.get("mask", None)
        ids = batch.get("id", None)

        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)

        with amp_autocast():
            logits = model_eval(imgs)
            # resize to match masks if present
            if masks is not None:
                masks = masks.long().to(ptu.device)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="nearest")
                loss = criterion_eval(logits, masks)
                total_loss += float(loss.item()) * imgs.size(0)
                total_samples += imgs.size(0)

            preds = logits.argmax(1).cpu().numpy()

        # store predictions keyed by id (or index)
        if ids is None:
            ids = [f"img_{i}" for i in range(preds.shape[0])]
        for i, fid in enumerate(ids):
            fid_key = os.path.splitext(fid)[0] if isinstance(fid, str) else fid
            val_preds[fid_key] = preds[i]

    avg_loss = total_loss / max(1, total_samples)

    # gather across processes (if distributed)
    val_preds = gather_data(val_preds)

    # filter ground truth keys to those present in val_preds
    val_gt_filtered = {k: np.asarray(val_seg_gt[k], dtype=np.int64) for k in val_preds.keys() if k in val_seg_gt}

    metrics = compute_segmentation_metrics(val_preds, val_gt_filtered, n_cls, ignore_index=IGNORE_LABEL)

    # Per-class dice (recompute using preds and gts)
    perclass_dice = []
    for k, pred in val_preds.items():
        if k in val_gt_filtered:
            perclass_dice.append(dice_per_class_from_preds(
                np.expand_dims(pred, 0), np.expand_dims(val_gt_filtered[k], 0), n_cls, ignore_index=IGNORE_LABEL
            ))
    if len(perclass_dice) > 0:
        perclass_dice = np.mean(np.stack(perclass_dice, axis=0), axis=0)
    else:
        perclass_dice = np.zeros(n_cls, dtype=np.float32)

    # Massage metrics dict to the format train.py expects
    out = {
        "Pixel_Acc": metrics["PixelAcc"],
        "Mean_Acc": metrics["MeanAcc"],
        "Mean_IoU": metrics["MeanIoU"],
        "FWIoU": metrics["FWIoU"],
        "Precision_per_class": metrics["Precision_per_class"],
        "Recall_per_class": metrics["Recall_per_class"],
        "F1_per_class": metrics["F1_per_class"],
        "IoU_per_class": metrics["IoU_per_class"],
        "PerClassDice": perclass_dice.tolist(),
    }

    if print_per_class:
        print("\n[Eval] Per-class IoU / F1 / Precision / Recall:")
        for c in range(n_cls):
            print(f"  Class {c}: IoU={out['IoU_per_class'][c]:.4f} F1={out['F1_per_class'][c]:.4f} "
                  f"Prec={out['Precision_per_class'][c]:.4f} Rec={out['Recall_per_class'][c]:.4f}")

    return avg_loss, out
