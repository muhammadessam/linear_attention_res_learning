import argparse
from calendar import c
import os
import time
import random
from typing import Tuple, Callable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from models.linear_feat_extractot import SpikformerCCSSA_QKFusion_ST
from datasets.npy_mulit_dataset import create_data_loaders

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for a, v, y in loader:
        functional.reset_net(model)
        a = a.to(device)  # [B,T,C,H,W]
        v = v.to(device)
        y = y.to(device)

        a = a.permute(1, 0, 2, 3, 4).contiguous()  # [T,B,C,H,W]
        v = v.permute(1, 0, 2, 3, 4).contiguous()

        logits, _ = model(a, v)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for a, v, y in train_loader:
        a = a.to(device)  # [B,T,C,H,W]
        v = v.to(device)
        y = y.to(device)

        a = a.permute(1, 0, 2, 3, 4).contiguous()  # [T,B,C,H,W]
        v = v.permute(1, 0, 2, 3, 4).contiguous()
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)

        use_amp = amp_enabled and (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(a, v)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = epoch_loss / max(1, total)
    train_acc = correct / max(1, total)
    return train_loss, train_acc


@torch.no_grad()
def recalibrate_bn_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 32,
) -> None:
    # Put model into train mode for BN updates but disable dropout
    model.train()
    drop_mods = []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            drop_mods.append(m)
            m.eval()

    seen = 0
    for a, v, _ in loader:
        a = a.to(device)
        v = v.to(device)
        a = a.permute(1, 0, 2, 3, 4).contiguous()
        v = v.permute(1, 0, 2, 3, 4).contiguous()
        functional.reset_net(model)
        _ = model(a, v)
        seen += 1
        if seen >= num_batches:
            break

    for m in drop_mods:
        m.train()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bi-directional QKFormer x CCSSA (ST-AVE) on CREMA NPY dataset")
    parser.add_argument("--data_root", type=str, default="/user/mohamed.saleh01/u18697/projects/dataset_final/crema/crema_t4_128_repeat_audio_unique")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_crema_unique")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--embed_dims", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bn_recal_batches", type=int, default=10, help="BN recalibration batches from train set before validation")
    parser.add_argument("--plot_every", type=int, default=10, help="Save plots every N epochs")
    parser.add_argument("--time-step", type=int, default=4, help="Number of time steps")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes for the dataset")
    return parser.parse_args()


def save_curves(hist: dict, save_dir: str, epoch: int) -> None:
    if not hist["epoch"]:
        return

    if not hasattr(save_curves, "fig"):
        save_curves.fig, save_curves.axes = plt.subplots(1, 2, figsize=(10, 4))

    fig, axes = save_curves.fig, save_curves.axes
    ax_loss, ax_acc = axes

    ax_loss.clear()
    ax_acc.clear()

    epochs = hist["epoch"]
    ax_loss.plot(epochs, hist["train_loss"], label="train")
    ax_loss.plot(epochs, hist["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, hist["train_acc"], label="train")
    ax_acc.plot(epochs, hist["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Acc")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    out_path = os.path.join(save_dir, "curves.png")
    fig.tight_layout()
    fig.savefig(out_path)


def write_log_csv(hist: dict, log_csv: str) -> None:
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,time_sec\n")
        for i in range(len(hist["epoch"])):
            f.write(
                f"{hist['epoch'][i]},{hist['train_loss'][i]:.6f},{hist['train_acc'][i]:.6f},{hist['val_loss'][i]:.6f},{hist['val_acc'][i]:.6f},{hist['time_sec'][i]:.2f}\n"
            )


def main():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(args.data_root, args)
   
    # Model (Bi-directional ST-AVE)
    model = SpikformerCCSSA_QKFusion_ST(
        num_classes=args.num_classes,
        step=args.time_step,
        in_channels_audio=1,
        in_channels_video=1,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        align_mode='interp',
        target_hw=None,
        apply_stages=2,
        add_rpe=True,
        alpha=args.alpha,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args={},
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and torch.cuda.is_available()))

    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, "qk_ccssa_st_crem_bi_best.pt")

    # Training log containers
    hist = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "time_sec": [],
    }

    log_csv = os.path.join(args.save_dir, "training_log.csv")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp_enabled=scaler.is_enabled() and args.amp,
        )

        # BN recalibration to stabilize eval
        recalibrate_bn_stats(model, train_loader, device, num_batches=args.bn_recal_batches)
        val_loss, val_acc = evaluate(model, val_loader, device)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | {dt:5.1f}s | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        # Log epoch
        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["time_sec"].append(dt)
        write_log_csv(hist, log_csv)

        # Save curves periodically
        if args.plot_every > 0 and (epoch % args.plot_every == 0):
            save_curves(hist, args.save_dir, epoch)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "args": vars(args),
                    "model_config": {
                        "num_classes": args.num_classes,
                        "step": args.time_step,
                        "embed_dims": args.embed_dims,
                        "num_heads": args.num_heads,
                        "mlp_ratio": args.mlp_ratio,
                        "alpha": args.alpha,
                    },
                },
                best_path,
            )
            print(f"  ↳ Saved best checkpoint to {best_path} (val_acc={best_val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}. Best ckpt: {best_path}")


if __name__ == "__main__":
    main()
