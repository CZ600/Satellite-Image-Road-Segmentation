import argparse
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Models.ConvNeXt_UPerNet_DGCN_MTL import ConvNeXt_UPerNet_DGCN_MTL
from Tools import DatasetUtility
from Tools import Losses
from Tools import util

tqdm.monitor_interval = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="ConvNeXt_UPerNet_DGCN_MTL",
        choices=["ConvNeXt_UPerNet_DGCN_MTL"],
        help="Model name.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="DeepGlobe",
        choices=["DeepGlobe", "MassachusettsRoads", "Spacenet"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--config",
        default="cfg.json",
        help="Path to config json.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from a checkpoint path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader workers.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Default is DGCN_<timestamp>.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def align_predictions_if_needed(dataset_name, predicted_road, predicted_orientation, road_labels, orient_labels):
    if dataset_name != "Spacenet":
        return predicted_road, predicted_orientation
    resized_road = []
    resized_orientation = []
    for index, tensor in enumerate(predicted_road):
        resized_road.append(
            F.interpolate(
                tensor,
                size=road_labels[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
    for index, tensor in enumerate(predicted_orientation):
        resized_orientation.append(
            F.interpolate(
                tensor,
                size=orient_labels[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
    return resized_road, resized_orientation


def unpack_metrics(hist, loss, aux_loss):
    metrics = util.segmentation_metrics_from_hist(hist)
    metrics["loss"] = loss
    metrics["aux_loss"] = aux_loss
    return metrics


def append_metrics_row(csv_path, epoch, split, metrics):
    row = (
        f"{epoch},{split},{metrics['loss']:.6f},{metrics['background_iou']:.6f},"
        f"{metrics['road_iou']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},"
        f"{metrics['miou']:.6f},{metrics['pixel_accuracy']:.6f},{metrics['aux_loss']:.6f}\n"
    )
    with open(csv_path, "a", encoding="utf-8") as file:
        file.write(row)


def log_to_tensorboard(writer, split, metrics, epoch):
    writer.add_scalar(f"{split}/loss", metrics["loss"], epoch)
    writer.add_scalar(f"{split}/background_iou", metrics["background_iou"], epoch)
    writer.add_scalar(f"{split}/road_iou", metrics["road_iou"], epoch)
    writer.add_scalar(f"{split}/precision", metrics["precision"], epoch)
    writer.add_scalar(f"{split}/recall", metrics["recall"], epoch)
    writer.add_scalar(f"{split}/miou", metrics["miou"], epoch)


def save_checkpoint(path, epoch, model, optimizer, cfg, metrics, run_name):
    state = util.checkpoint_state(epoch, model, optimizer, cfg, metrics)
    state["run_name"] = run_name
    torch.save(state, path)


def maybe_strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return util.getParllelNetworkStateDict(state_dict)


def run_epoch(
    model,
    loader,
    optimizer,
    segmentation_loss,
    orientation_loss,
    device,
    dataset_name,
    n_road_classes,
    is_train,
):
    model.train(mode=is_train)
    total_road_loss = 0.0
    total_angle_loss = 0.0
    hist = np.zeros((n_road_classes, n_road_classes), dtype=np.float64)
    batch_count = 0

    progress = tqdm(loader, ncols=120, leave=False, desc="train" if is_train else "val")
    for images, road_labels, orient_labels in progress:
        batch_count += 1
        images = images.float().to(device, non_blocking=True)
        road_labels = [label.to(device, non_blocking=True) for label in road_labels]
        orient_labels = [label.to(device, non_blocking=True) for label in orient_labels]

        with torch.set_grad_enabled(is_train):
            predictions = model(images)
            predicted_road = list(predictions[0])
            predicted_orientation = list(predictions[1])
            predicted_road, predicted_orientation = align_predictions_if_needed(
                dataset_name,
                predicted_road,
                predicted_orientation,
                road_labels,
                orient_labels,
            )

            road_loss = sum(
                segmentation_loss(predicted_road[index], road_labels[index])
                for index in range(len(predicted_road))
            )
            angle_loss = sum(
                orientation_loss(predicted_orientation[index], orient_labels[index])
                for index in range(len(predicted_orientation))
            )
            total_loss = road_loss + angle_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

        total_road_loss += road_loss.item()
        total_angle_loss += angle_loss.item()

        final_pred = predicted_road[-1].argmax(dim=1)
        final_target = road_labels[-1].long()
        hist += util.fast_hist(
            final_pred.view(final_pred.size(0), -1).detach().cpu().numpy(),
            final_target.view(final_target.size(0), -1).detach().cpu().numpy(),
            n_road_classes,
        )

        current_metrics = unpack_metrics(
            hist,
            (total_road_loss + total_angle_loss) / batch_count,
            total_angle_loss / batch_count,
        )
        progress.set_postfix(
            loss=f"{((total_road_loss + total_angle_loss) / batch_count):.4f}",
            miou=f"{current_metrics['miou']:.4f}",
            road_iou=f"{current_metrics['road_iou']:.4f}",
        )

    avg_road_loss = total_road_loss / len(loader)
    avg_angle_loss = total_angle_loss / len(loader)
    metrics = unpack_metrics(hist, avg_road_loss + avg_angle_loss, avg_angle_loss)
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This training entry requires a CUDA-capable GPU.")

    device = torch.device("cuda")
    seed = cfg["GlobalSeed"]
    set_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"DGCN_{timestamp}"
    checkpoint_dir = Path(cfg["training_settings"]["checkpoint_dir"])
    log_root = Path(cfg["training_settings"]["log_dir"])
    log_dir = log_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_json(log_dir / "config_snapshot.json", cfg)

    writer = SummaryWriter(log_dir=str(log_dir))
    metrics_csv = log_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text(
            "epoch,split,loss,background_iou,road_iou,precision,recall,miou,pixel_accuracy,aux_loss\n",
            encoding="utf-8",
        )

    model = ConvNeXt_UPerNet_DGCN_MTL()
    dataset_class = getattr(DatasetUtility, args.dataset)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["optimizer_settings"]["learning_rate"],
        momentum=0.9,
        weight_decay=cfg["optimizer_settings"]["learning_rate_decay"],
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=eval(cfg["optimizer_settings"]["learning_rate_drop_at_epoch"]),
        gamma=cfg["optimizer_settings"]["learning_rate_step"],
    )

    workers = args.workers
    if workers is None:
        workers = cfg["training_settings"].get("num_workers", 4)

    train_loader = data.DataLoader(
        dataset_class(cfg, args.model, args.dataset, "training_settings"),
        batch_size=cfg["training_settings"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    valid_loader = data.DataLoader(
        dataset_class(cfg, args.model, args.dataset, "validation_settings"),
        batch_size=cfg["validation_settings"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    n_road_classes = cfg["training_settings"]["roadclass"]
    n_orient_classes = cfg["training_settings"]["orientationclass"]
    segmentation_weights = torch.ones(n_road_classes, device=device)
    orientation_weights = torch.ones(n_orient_classes, device=device)
    segmentation_loss = Losses.mIoULoss(weight=segmentation_weights, n_classes=n_road_classes).to(device)
    orientation_loss = Losses.CrossEntropyLossImage(weight=orientation_weights, ignore_index=255).to(device)

    start_epoch = 1
    best_miou = -math.inf
    best_metrics = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint["state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(maybe_strip_module_prefix(state_dict))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_metrics = checkpoint.get("metrics")
        if best_metrics is not None:
            best_miou = best_metrics.get("miou", -math.inf)

    total_epochs = cfg["training_settings"]["epochs"]
    checkpoint_interval = cfg["training_settings"]["checkpoint_interval"]

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            segmentation_loss,
            orientation_loss,
            device,
            args.dataset,
            n_road_classes,
            is_train=True,
        )
        val_metrics = run_epoch(
            model,
            valid_loader,
            optimizer,
            segmentation_loss,
            orientation_loss,
            device,
            args.dataset,
            n_road_classes,
            is_train=False,
        )
        scheduler.step()

        log_to_tensorboard(writer, "train", train_metrics, epoch)
        log_to_tensorboard(writer, "val", val_metrics, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        append_metrics_row(metrics_csv, epoch, "train", train_metrics)
        append_metrics_row(metrics_csv, epoch, "val", val_metrics)

        latest_path = checkpoint_dir / f"{run_name}_latest.pth.tar"
        save_checkpoint(latest_path, epoch, model, optimizer, cfg, val_metrics, run_name)

        if epoch % checkpoint_interval == 0:
            epoch_path = checkpoint_dir / f"{run_name}_epoch{epoch:03d}.pth.tar"
            save_checkpoint(epoch_path, epoch, model, optimizer, cfg, val_metrics, run_name)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_metrics = deepcopy(val_metrics)
            best_path = checkpoint_dir / f"{run_name}_best.pth.tar"
            save_checkpoint(best_path, epoch, model, optimizer, cfg, val_metrics, run_name)

        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"train loss={train_metrics['loss']:.4f}, miou={train_metrics['miou']:.4f}, road_iou={train_metrics['road_iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, miou={val_metrics['miou']:.4f}, road_iou={val_metrics['road_iou']:.4f}, "
            f"precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}"
        )

    writer.close()
    if best_metrics is not None:
        print(
            f"Best val mIoU={best_metrics['miou']:.4f}, "
            f"road IoU={best_metrics['road_iou']:.4f}, "
            f"precision={best_metrics['precision']:.4f}, recall={best_metrics['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
