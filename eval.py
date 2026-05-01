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
from scipy import stats

import torch
import torch.nn as nn


# CONFIG

NUM_EXAMPLES = 10


POPULARITY_TASKS = ["score_streams", "score_likes"]
SONGEVAL_TASKS   = ["coherence", "musicality", "memorability", "clarity", "naturalness"]
ALL_TASKS        = POPULARITY_TASKS + SONGEVAL_TASKS



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



# LOAD TEST DATA

def load_test_data(test_folder, mode, task):
    files = sorted(glob.glob(os.path.join(test_folder, "*.parquet")))
    dfs   = []
    for f in tqdm(files, desc="Loading test parquets"):
        dfs.append(pq.read_table(f).to_pandas())
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} segments | {df['audio_id'].nunique():,} unique songs")

    if mode == "song":
        logger.info("Aggregating segments to song level...")
        df = aggregate_to_song(df)
        logger.info(f"Aggregated to {len(df):,} songs")

    return df


def aggregate_to_song(df):
    return df.groupby(["audio_id", "platform"]).agg(
        segment_embedding = ("segment_embedding", lambda x: np.stack(x.values).mean(axis=0)),
        score_streams     = ("score_streams",     "first"),
        score_likes       = ("score_likes",       "first"),
        songeval_scores   = ("songeval_scores",   "first")
    ).reset_index()



# RUN INFERENCE

def run_inference(model, df, task, device, batch_size=512):
    model.eval()
    embeddings = np.stack(df["segment_embedding"].values).astype(np.float32)
    all_preds  = {t: [] for t in (POPULARITY_TASKS if task == "popularity" else ALL_TASKS)}

    with torch.no_grad():
        for start in tqdm(range(0, len(embeddings), batch_size), desc="Inference"):
            batch = torch.tensor(embeddings[start:start + batch_size]).to(device)
            preds = model(batch)
            for t in all_preds:
                all_preds[t].extend(preds[t].cpu().numpy().tolist())

    return {t: np.array(v) for t, v in all_preds.items()}



# AGGREGATE SEGMENT PREDICTIONS TO SONG LEVEL

def aggregate_segment_predictions(df, preds):
    df = df.copy()
    for t, v in preds.items():
        df[f"pred_{t}"] = v

    return df.groupby(["audio_id", "platform"]).agg(
        score_streams   = ("score_streams",   "first"),
        score_likes     = ("score_likes",     "first"),
        songeval_scores = ("songeval_scores", "first"),
        **{f"pred_{t}": (f"pred_{t}", "mean") for t in preds.keys()}
    ).reset_index()



# GET GROUND TRUTH

def get_ground_truth(df, task):
    gt = {
        "score_streams": df["score_streams"].values.astype(np.float32),
        "score_likes"  : df["score_likes"].values.astype(np.float32),
    }
    if task == "full":
        songeval = df["songeval_scores"].apply(
            lambda x: x if isinstance(x, dict) else json.loads(x)
        )
        gt["coherence"]    = songeval.apply(lambda x: x["coherence"]).values.astype(np.float32)
        gt["musicality"]   = songeval.apply(lambda x: x["musicality"]).values.astype(np.float32)
        gt["memorability"] = songeval.apply(lambda x: x["memorability"]).values.astype(np.float32)
        gt["clarity"]      = songeval.apply(lambda x: x["clarity"]).values.astype(np.float32)
        gt["naturalness"]  = songeval.apply(lambda x: x["naturalness"]).values.astype(np.float32)
    return gt



# COMPUTE METRICS

def compute_metrics(actual, predicted):
    return {
        "mse"     : float(np.mean((actual - predicted) ** 2)),
        "mae"     : float(np.mean(np.abs(actual - predicted))),
        "pearson" : float(stats.pearsonr(actual, predicted)[0]),
        "spearman": float(stats.spearmanr(actual, predicted)[0])
    }



# GET EXAMPLES

def get_examples(df, actual, predicted, n=10):
    temp = pd.DataFrame({
        "audio_id" : df["audio_id"].values,
        "platform" : df["platform"].values,
        "actual"   : actual,
        "predicted": predicted
    })
    sorted_df = temp.sort_values("actual", ascending=False)
    top_n     = sorted_df.head(n)
    bottom_n  = sorted_df.tail(n).sort_values("actual", ascending=True)

    def to_records(d):
        return [
            {
                "audio_id" : row["audio_id"],
                "platform" : row["platform"],
                "actual"   : round(float(row["actual"]), 4),
                "predicted": round(float(row["predicted"]), 4),
                "error"    : round(float(abs(row["actual"] - row["predicted"])), 4)
            }
            for _, row in d.iterrows()
        ]

    return {
        "top_10_highest": to_records(top_n),
        "top_10_lowest" : to_records(bottom_n)
    }



# PRINT METRICS

def print_metrics(metrics, mode, task):
    tasks = POPULARITY_TASKS if task == "popularity" else ALL_TASKS
    print(f"\n{'-'*60}")
    print(f"  EVALUATION RESULTS | mode={mode} | task={task}")
    print(f"{'─'*60}")
    print(f"  {'Task':<20} {'MSE':>10} {'MAE':>10} {'Pearson':>10} {'Spearman':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for t in tasks:
        m = metrics[t]
        print(f"  {t:<20} {m['mse']:>10.4f} {m['mae']:>10.4f} {m['pearson']:>10.4f} {m['spearman']:>10.4f}")
    print(f"{'-'*60}\n")



# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX Evaluation Script")
    parser.add_argument("--checkpoint",   type=str, required=True,  help="Path to checkpoint file")
    parser.add_argument("--test_folder",  type=str, required=True,  help="Path to test parquet folder")
    parser.add_argument("--results_folder", type=str, default="eval_results", help="Path to save results")
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.results_folder, "eval.log")
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load checkpoint — auto detect all settings
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    mode     = checkpoint["mode"]
    task     = checkpoint["task"]
    n_shared = checkpoint.get("shared", 2)  # default 2 for old checkpoints
    loss     = checkpoint.get("loss", "equal")

    logger.info(f"Auto-detected — mode: {mode} | task: {task} | shared: {n_shared} | loss: {loss}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = apexMLP(task=task, n_shared=n_shared)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from epoch {checkpoint['epoch']} | Val Loss: {checkpoint['val_loss']:.4f}")

    # Load test data
    df    = load_test_data(args.test_folder, mode, task)
    preds = run_inference(model, df, task, device)

    # Aggregate if segment mode
    if mode == "segment":
        logger.info("Aggregating segment predictions to song level...")
        song_df = aggregate_segment_predictions(df, preds)
        eval_df = song_df
        tasks   = POPULARITY_TASKS if task == "popularity" else ALL_TASKS
        preds   = {t: song_df[f"pred_{t}"].values for t in tasks}
    else:
        eval_df = df

    # Ground truth
    gt    = get_ground_truth(eval_df, task)
    tasks = POPULARITY_TASKS if task == "popularity" else ALL_TASKS

    # Compute metrics
    metrics = {}
    for t in tasks:
        metrics[t] = compute_metrics(gt[t], preds[t])
        logger.info(f"{t}: MSE={metrics[t]['mse']:.4f} | MAE={metrics[t]['mae']:.4f} | Pearson={metrics[t]['pearson']:.4f} | Spearman={metrics[t]['spearman']:.4f}")

    print_metrics(metrics, mode, task)

    # Examples
    examples = {}
    for t in tasks:
        examples[t] = get_examples(eval_df, gt[t], preds[t], n=NUM_EXAMPLES)

    # Print examples
    for t in tasks:
        print(f"\n── {t} — Top 10 Highest ────────────────────────────")
        print(f"  {'audio_id':<40} {'platform':<8} {'actual':>8} {'predicted':>10} {'error':>8}")
        print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
        for ex in examples[t]["top_10_highest"]:
            print(f"  {ex['audio_id']:<40} {ex['platform']:<8} {ex['actual']:>8.4f} {ex['predicted']:>10.4f} {ex['error']:>8.4f}")

        print(f"\n── {t} — Top 10 Lowest ─────────────────────────────")
        print(f"  {'audio_id':<40} {'platform':<8} {'actual':>8} {'predicted':>10} {'error':>8}")
        print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
        for ex in examples[t]["top_10_lowest"]:
            print(f"  {ex['audio_id']:<40} {ex['platform']:<8} {ex['actual']:>8.4f} {ex['predicted']:>10.4f} {ex['error']:>8.4f}")

    # Save results
    results = {
        "checkpoint"    : args.checkpoint,
        "mode"          : mode,
        "task"          : task,
        "n_shared"      : n_shared,
        "loss"          : loss,
        "epoch"         : checkpoint["epoch"],
        "val_loss"      : checkpoint["val_loss"],
        "evaluated_at"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_songs"     : int(eval_df["audio_id"].nunique()),
        "metrics"       : metrics,
        "examples"      : examples
    }

    results_path = os.path.join(
        args.results_folder,
        f"eval_results_mode-{mode}_task-{task}_shared-{n_shared}_loss-{loss}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")