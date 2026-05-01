import os
import json
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import wandb

# CONFIG
TRAIN_FOLDER  = "final_split/train"
VAL_FOLDER    = "final_split/val"
INPUT_DIM     = 768
EPOCHS        = 50
PATIENCE      = 20
BATCH_SIZE    = 512
NUM_WORKERS   = 10
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
SEED          = 42
WANDB_PROJECT = "music-popularity"

LOSS_WEIGHTS = {
    "score_streams": 5.0,
    "score_likes"  : 5.0,
    "coherence"    : 1.0,
    "musicality"   : 1.0,
    "memorability" : 1.0,
    "clarity"      : 1.0,
    "naturalness"  : 1.0
}

POPULARITY_TASKS = ["score_streams", "score_likes"]
SONGEVAL_TASKS   = ["coherence", "musicality", "memorability", "clarity", "naturalness"]
ALL_TASKS        = POPULARITY_TASKS + SONGEVAL_TASKS


# DATASET
class MusicDataset(Dataset):
    def __init__(self, folder, mode, task, rank=0):
        self.mode  = mode
        self.task  = task
        self.tasks = POPULARITY_TASKS if task == "popularity" else ALL_TASKS

        files      = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        dfs        = []
        files_iter = tqdm(files, desc=f"Loading {os.path.basename(folder)} parquets", leave=True) if rank == 0 else files
        for f in files_iter:
            dfs.append(pq.read_table(f).to_pandas())
        df = pd.concat(dfs, ignore_index=True)

        if mode == "song":
            df = self._aggregate_to_song(df)

        self.embeddings    = np.stack(df["segment_embedding"].values).astype(np.float32)
        self.score_streams = df["score_streams"].values.astype(np.float32)
        self.score_likes   = df["score_likes"].values.astype(np.float32)

        if task == "full":
            songeval          = df["songeval_scores"].apply(
                lambda x: x if isinstance(x, dict) else json.loads(x)
            )
            self.coherence    = songeval.apply(lambda x: x["coherence"]).values.astype(np.float32)
            self.musicality   = songeval.apply(lambda x: x["musicality"]).values.astype(np.float32)
            self.memorability = songeval.apply(lambda x: x["memorability"]).values.astype(np.float32)
            self.clarity      = songeval.apply(lambda x: x["clarity"]).values.astype(np.float32)
            self.naturalness  = songeval.apply(lambda x: x["naturalness"]).values.astype(np.float32)

    def _aggregate_to_song(self, df):
        return df.groupby(["audio_id", "platform"]).agg(
            segment_embedding = ("segment_embedding", lambda x: np.stack(x.values).mean(axis=0)),
            score_streams     = ("score_streams",     "first"),
            score_likes       = ("score_likes",       "first"),
            songeval_scores   = ("songeval_scores",   "first")
        ).reset_index()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = {
            "embedding"     : torch.tensor(self.embeddings[idx]),
            "score_streams" : torch.tensor(self.score_streams[idx]),
            "score_likes"   : torch.tensor(self.score_likes[idx]),
        }
        if self.task == "full":
            sample["coherence"]    = torch.tensor(self.coherence[idx])
            sample["musicality"]   = torch.tensor(self.musicality[idx])
            sample["memorability"] = torch.tensor(self.memorability[idx])
            sample["clarity"]      = torch.tensor(self.clarity[idx])
            sample["naturalness"]  = torch.tensor(self.naturalness[idx])
        return sample



# MODEL

class SharedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class BranchBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, use_bn=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers += [nn.GELU(), nn.Dropout(dropout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TaskBranch(nn.Module):
    def __init__(self, scale, shift):
        super().__init__()
        self.branch = nn.Sequential(
            BranchBlock(256, 128, dropout=0.1, use_bn=True),
            BranchBlock(128, 64,  dropout=0.1, use_bn=True),
            nn.Linear(64, 1)
        )
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(self.branch(x)) * self.scale + self.shift


class apexMLP(nn.Module):
    def __init__(self, task, n_shared):
        super().__init__()
        self.task = task

        # Shared layers based on n_shared
        if n_shared == 2:
            self.shared = nn.Sequential(
                SharedBlock(768, 512, dropout=0.3),
                SharedBlock(512, 256, dropout=0.3)
            )
        elif n_shared == 3:
            self.shared = nn.Sequential(
                SharedBlock(768, 512, dropout=0.3),
                SharedBlock(512, 384, dropout=0.3),
                SharedBlock(384, 256, dropout=0.3)
            )
        else:
            raise ValueError(f"n_shared must be 2 or 3, got {n_shared}")

        self.branch_score_streams = TaskBranch(scale=100, shift=0)
        self.branch_score_likes   = TaskBranch(scale=100, shift=0)

        if task == "full":
            self.branch_coherence    = TaskBranch(scale=4, shift=1)
            self.branch_musicality   = TaskBranch(scale=4, shift=1)
            self.branch_memorability = TaskBranch(scale=4, shift=1)
            self.branch_clarity      = TaskBranch(scale=4, shift=1)
            self.branch_naturalness  = TaskBranch(scale=4, shift=1)

    def forward(self, x):
        shared = self.shared(x)
        out = {
            "score_streams": self.branch_score_streams(shared).squeeze(1),
            "score_likes"  : self.branch_score_likes(shared).squeeze(1),
        }
        if self.task == "full":
            out["coherence"]    = self.branch_coherence(shared).squeeze(1)
            out["musicality"]   = self.branch_musicality(shared).squeeze(1)
            out["memorability"] = self.branch_memorability(shared).squeeze(1)
            out["clarity"]      = self.branch_clarity(shared).squeeze(1)
            out["naturalness"]  = self.branch_naturalness(shared).squeeze(1)
        return out



# UNCERTAINTY LOSS MODULE

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.log_sigmas = nn.ParameterDict({
            t.replace("-", "_"): nn.Parameter(torch.zeros(1))
            for t in tasks
        })

    def forward(self, losses):
        total    = 0
        weighted = {}
        for t, loss in losses.items():
            key           = t.replace("-", "_")
            log_sigma     = self.log_sigmas[key]
            weighted_loss = (1 / (2 * torch.exp(log_sigma) ** 2)) * loss + log_sigma
            weighted[t]   = weighted_loss.item()
            total        += weighted_loss
        return total, weighted

    def get_weights(self):
        return {
            t.replace("_", "-"): float(1 / (torch.exp(self.log_sigmas[t]) ** 2))
            for t in self.log_sigmas
        }

    def get_sigmas(self):
        return {
            t.replace("_", "-"): float(torch.exp(self.log_sigmas[t]))
            for t in self.log_sigmas
        }



# COMPUTE LOSSES

def compute_raw_losses(preds, batch, task, device):
    mse    = nn.MSELoss()
    losses = {}
    losses["score_streams"] = mse(preds["score_streams"], batch["score_streams"].to(device))
    losses["score_likes"]   = mse(preds["score_likes"],   batch["score_likes"].to(device))
    if task == "full":
        losses["coherence"]    = mse(preds["coherence"],    batch["coherence"].to(device))
        losses["musicality"]   = mse(preds["musicality"],   batch["musicality"].to(device))
        losses["memorability"] = mse(preds["memorability"], batch["memorability"].to(device))
        losses["clarity"]      = mse(preds["clarity"],      batch["clarity"].to(device))
        losses["naturalness"]  = mse(preds["naturalness"],  batch["naturalness"].to(device))
    return losses


def combine_losses(raw_losses, loss_type, uncertainty_loss=None):
    if loss_type == "equal":
        total = sum(raw_losses.values())
        return total, {}

    elif loss_type == "weighted":
        total = sum(LOSS_WEIGHTS[t] * v for t, v in raw_losses.items())
        return total, {}

    elif loss_type == "uncertainty":
        return uncertainty_loss(raw_losses)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")



# TRAIN ONE EPOCH

def train_epoch(model, uncertainty_loss, loader, optimizer, scaler, task, loss_type, device, rank, epoch):
    model.train()
    if uncertainty_loss is not None:
        uncertainty_loss.train()

    total_loss  = 0
    task_losses = {t: 0 for t in (POPULARITY_TASKS if task == "popularity" else ALL_TASKS)}
    num_batches = 0

    pbar = tqdm(
        loader,
        desc          = f"Epoch {epoch:03d} | Train",
        leave         = False,
        unit          = "batch",
        dynamic_ncols = True
    ) if rank == 0 else loader

    for batch in pbar:
        optimizer.zero_grad()

        with autocast():
            preds      = model(batch["embedding"].to(device))
            raw_losses = compute_raw_losses(preds, batch, task, device)
            loss, _    = combine_losses(raw_losses, loss_type, uncertainty_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        params = list(model.parameters())
        if uncertainty_loss is not None:
            params += list(uncertainty_loss.parameters())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        num_batches += 1
        for t, v in raw_losses.items():
            task_losses[t] += v.item()

        if rank == 0:
            pbar.set_postfix({
                "loss"   : f"{loss.item():.4f}",
                "streams": f"{raw_losses['score_streams'].item():.4f}",
                "likes"  : f"{raw_losses['score_likes'].item():.4f}",
            })

    return total_loss / num_batches, {t: v / num_batches for t, v in task_losses.items()}



# VALIDATE

def validate(model, uncertainty_loss, loader, task, loss_type, device, rank, epoch):
    model.eval()
    if uncertainty_loss is not None:
        uncertainty_loss.eval()

    total_loss  = 0
    task_losses = {t: 0 for t in (POPULARITY_TASKS if task == "popularity" else ALL_TASKS)}
    num_batches = 0

    pbar = tqdm(
        loader,
        desc          = f"Epoch {epoch:03d} | Val  ",
        leave         = False,
        unit          = "batch",
        dynamic_ncols = True
    ) if rank == 0 else loader

    with torch.no_grad():
        for batch in pbar:
            with autocast():
                preds      = model(batch["embedding"].to(device))
                raw_losses = compute_raw_losses(preds, batch, task, device)
                loss, _    = combine_losses(raw_losses, loss_type, uncertainty_loss)

            total_loss  += loss.item()
            num_batches += 1
            for t, v in raw_losses.items():
                task_losses[t] += v.item()

            if rank == 0:
                pbar.set_postfix({
                    "loss"   : f"{loss.item():.4f}",
                    "streams": f"{raw_losses['score_streams'].item():.4f}",
                    "likes"  : f"{raw_losses['score_likes'].item():.4f}",
                })

    return total_loss / num_batches, {t: v / num_batches for t, v in task_losses.items()}



# PRINT EPOCH SUMMARY

def print_epoch_summary(rank, epoch, train_loss, train_tasks, val_loss, val_tasks, lr, best_val, patience_count, loss_type, uncertainty_loss=None):
    if rank != 0:
        return
    print(f"\n{'─'*70}")
    print(f"  Epoch {epoch:03d}/{EPOCHS} | LR: {lr:.6f} | Loss: {loss_type}")
    print(f"{'─'*70}")

    if loss_type == "uncertainty" and uncertainty_loss is not None:
        sigmas  = uncertainty_loss.module.get_sigmas()
        weights = uncertainty_loss.module.get_weights()
        print(f"  {'Task':<20} {'σ':>8} {'W':>8} {'Train':>10} {'Val':>10}")
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
        print(f"  {'TOTAL':<20} {'':>8} {'':>8} {train_loss:>10.4f} {val_loss:>10.4f}")
        for t in train_tasks:
            print(f"  {t:<20} {sigmas.get(t,1.0):>8.4f} {weights.get(t,1.0):>8.4f} {train_tasks[t]:>10.4f} {val_tasks[t]:>10.4f}")
    else:
        print(f"  {'Task':<20} {'Train':>10} {'Val':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10}")
        print(f"  {'TOTAL':<20} {train_loss:>10.4f} {val_loss:>10.4f}")
        for t in train_tasks:
            print(f"  {t:<20} {train_tasks[t]:>10.4f} {val_tasks[t]:>10.4f}")

    print(f"{'─'*70}")
    print(f"  Best Val Loss : {best_val:.4f} | Patience: {patience_count}/{PATIENCE}")
    print(f"{'─'*70}\n")



# MAIN WORKER

def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend     = "nccl",
        init_method = "env://",
        world_size  = world_size,
        rank        = rank
    )
    torch.manual_seed(SEED)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tasks = POPULARITY_TASKS if args.task == "popularity" else ALL_TASKS

    # Checkpoint and log paths
    run_id            = f"loss-{args.loss}_shared-{args.shared}_mode-{args.mode}_task-{args.task}"
    checkpoint_folder = f"checkpoints/{run_id}"
    log_file          = f"logs/{run_id}.log"
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if rank == 0:
        logging.basicConfig(
            level    = logging.INFO,
            format   = "%(asctime)s [%(levelname)s] %(message)s",
            handlers = [
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    logger = logging.getLogger(__name__)

    if rank == 0:
        wandb.init(
            project = WANDB_PROJECT,
            name    = f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config  = {
                "mode"          : args.mode,
                "task"          : args.task,
                "loss"          : args.loss,
                "shared"        : args.shared,
                "epochs"        : EPOCHS,
                "batch_size"    : BATCH_SIZE * world_size,
                "lr"            : LR,
                "weight_decay"  : WEIGHT_DECAY,
                "grad_clip"     : GRAD_CLIP,
                "patience"      : PATIENCE,
                "num_gpus"      : world_size,
                "dropout_shared": 0.3,
                "dropout_branch": 0.1,
                "optimizer"     : "AdamW",
                "scheduler"     : "CosineAnnealing",
            }
        )
        logger.info("=" * 60)
        logger.info(f"Run: {run_id}")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

    # Datasets
    train_dataset = MusicDataset(TRAIN_FOLDER, args.mode, args.task, rank=rank)
    val_dataset   = MusicDataset(VAL_FOLDER,   args.mode, args.task, rank=rank)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size         = BATCH_SIZE,
        sampler            = train_sampler,
        num_workers        = NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size         = BATCH_SIZE,
        sampler            = val_sampler,
        num_workers        = NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2
    )

    # Model
    model = apexMLP(task=args.task, n_shared=args.shared).to(device)
    model = DDP(model, device_ids=[rank])

    # Uncertainty loss (only for uncertainty loss type)
    uncertainty_loss = None
    if args.loss == "uncertainty":
        uncertainty_loss = UncertaintyWeightedLoss(tasks=tasks).to(device)
        uncertainty_loss = DDP(uncertainty_loss, device_ids=[rank])

    # Optimizer
    params = list(model.parameters())
    if uncertainty_loss is not None:
        params += list(uncertainty_loss.parameters())

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    best_val_loss  = float("inf")
    patience_count = 0

    epoch_bar = tqdm(
        range(1, EPOCHS + 1),
        desc     = "Epochs",
        position = 0,
        leave    = True,
        unit     = "epoch"
    ) if rank == 0 else range(1, EPOCHS + 1)

    for epoch in epoch_bar:
        train_sampler.set_epoch(epoch)

        train_loss, train_tasks = train_epoch(
            model, uncertainty_loss, train_loader,
            optimizer, scaler, args.task, args.loss, device, rank, epoch
        )
        val_loss, val_tasks = validate(
            model, uncertainty_loss, val_loader,
            args.task, args.loss, device, rank, epoch
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            epoch_bar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val"  : f"{val_loss:.4f}",
                "best" : f"{best_val_loss:.4f}"
            })

            print_epoch_summary(
                rank, epoch,
                train_loss, train_tasks,
                val_loss, val_tasks,
                current_lr, best_val_loss, patience_count,
                args.loss, uncertainty_loss
            )

            # WandB logging
            log_dict = {
                "epoch"            : epoch,
                "train/total_loss" : train_loss,
                "val/total_loss"   : val_loss,
                "train/lr"         : current_lr,
            }
            for t, v in train_tasks.items():
                log_dict[f"train/{t}"] = v
            for t, v in val_tasks.items():
                log_dict[f"val/{t}"] = v
            if args.loss == "uncertainty" and uncertainty_loss is not None:
                for t, v in uncertainty_loss.module.get_sigmas().items():
                    log_dict[f"uncertainty/sigma_{t}"] = v
                for t, v in uncertainty_loss.module.get_weights().items():
                    log_dict[f"uncertainty/weight_{t}"] = v
            wandb.log(log_dict, step=epoch)

            logger.info(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0

                checkpoint = {
                    "epoch"          : epoch,
                    "model_state"    : model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss"       : val_loss,
                    "val_task_losses": val_tasks,
                    "train_loss"     : train_loss,
                    "mode"           : args.mode,
                    "task"           : args.task,
                    "loss"           : args.loss,
                    "shared"         : args.shared,
                }
                if args.loss == "uncertainty" and uncertainty_loss is not None:
                    checkpoint["uncertainty_loss_state"] = uncertainty_loss.module.state_dict()
                    checkpoint["final_sigmas"]           = uncertainty_loss.module.get_sigmas()
                    checkpoint["final_weights"]          = uncertainty_loss.module.get_weights()

                ckpt_path = os.path.join(checkpoint_folder, "best_model.pt")
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Epoch {epoch:03d} | Best model saved — Val Loss: {val_loss:.4f}")
                wandb.log({"best/val_loss": val_loss, "best/epoch": epoch}, step=epoch)

                artifact = wandb.Artifact(name=run_id, type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

            else:
                patience_count += 1
                logger.info(f"Epoch {epoch:03d} | No improvement — Patience: {patience_count}/{PATIENCE}")
                if patience_count >= PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    wandb.log({"early_stop_epoch": epoch})
                    break

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()



# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX Training Script")
    parser.add_argument("--mode",   type=str, required=True, choices=["segment", "song"],
                        help="segment: each segment independent | song: average segments per song")
    parser.add_argument("--task",   type=str, required=True, choices=["popularity", "full"],
                        help="popularity: 2 branches | full: 7 branches")
    parser.add_argument("--loss",   type=str, required=True, choices=["equal", "weighted", "uncertainty"],
                        help="equal: equal sum | weighted: manual weights | uncertainty: learned weights")
    parser.add_argument("--shared", type=int, required=True, choices=[2, 3],
                        help="2: 768→512→256 | 3: 768→512→384→256")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    assert world_size > 0, "No GPUs found!"

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        main_worker,
        args   = (world_size, args),
        nprocs = world_size,
        join   = True
    )