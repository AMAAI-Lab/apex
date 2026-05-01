import os
import json
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as TAF
from transformers import AutoProcessor, AutoModel


# CONFIG

MERT_MODEL_NAME    = "m-a-p/MERT-v1-95M"
LAYER_INDICES      = [2, 5, 8, -1]
SEGMENT_SEC        = 30
SEED               = 42
DEFAULT_CHECKPOINT = "models/best_model.pt"




# MODEL ARCHITECTURE

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



# APEX INFERENCE CLASS

class APEX:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        # Auto detect device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load MERT
        print("Loading MERT encoder...")
        self.processor = AutoProcessor.from_pretrained(MERT_MODEL_NAME, trust_remote_code=True)
        self.mert      = AutoModel.from_pretrained(MERT_MODEL_NAME, trust_remote_code=True)
        self.mert.eval()
        self.mert.to(self.device)
        self.target_sr = self.processor.sampling_rate

        # Conv1d aggregator with fixed seed
        torch.manual_seed(SEED)
        self.aggregator = nn.Conv1d(
            in_channels  = len(LAYER_INDICES),
            out_channels = 1,
            kernel_size  = 1
        ).to(self.device)
        self.aggregator.eval()

        # Load checkpoint — auto detect task and n_shared
        print(f"Loading APEX checkpoint: {checkpoint_path}")
        checkpoint  = torch.load(checkpoint_path, map_location="cpu")
        task        = checkpoint.get("task",   "full")
        n_shared    = checkpoint.get("shared", 2)
        mode        = checkpoint.get("mode",   "song")

        self.model = apexMLP(task=task, n_shared=n_shared)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded — mode: {mode} | task: {task} | shared: {n_shared} | epoch: {checkpoint['epoch']} | val loss: {checkpoint['val_loss']:.4f}")

    def _load_audio(self, audio_path):
        waveform, sr = sf.read(audio_path, dtype="float32")
        waveform     = torch.from_numpy(waveform)

        # Stereo to mono
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1)

        waveform = waveform.to(self.device)

        # Resample if needed
        if sr != self.target_sr:
            waveform = TAF.resample(waveform, sr, self.target_sr)

        return waveform

    def _extract_embedding(self, waveform):
        segment_len        = SEGMENT_SEC * self.target_sr
        segment_embeddings = []

        for start in range(0, waveform.shape[0], segment_len):
            segment = waveform[start:start + segment_len]
            if segment.numel() == 0:
                break

            # Zero-pad last segment if needed
            if segment.shape[0] < segment_len:
                pad_len = segment_len - segment.shape[0]
                segment = torch.nn.functional.pad(segment, (0, pad_len))

            inputs = self.processor(
                segment.cpu().numpy(),
                sampling_rate  = self.target_sr,
                return_tensors = "pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.mert(**inputs, output_hidden_states=True)

            # Extract layers and aggregate
            all_hidden = torch.stack([
                outputs.hidden_states[i].mean(dim=1)
                for i in LAYER_INDICES
            ])
            all_hidden = all_hidden.squeeze(1)

            pooled = self.aggregator(all_hidden.unsqueeze(0)).squeeze()
            segment_embeddings.append(pooled)

            del segment, inputs, outputs, all_hidden, pooled

        # Average across segments → song-level embedding
        song_embedding = torch.stack(segment_embeddings).mean(dim=0)
        return song_embedding

    @torch.no_grad()
    def predict(self, audio_path, save_json=None):
        print(f"\nProcessing: {audio_path}")

        waveform = self._load_audio(audio_path)
        duration = waveform.shape[0] / self.target_sr
        n_segs   = int(np.ceil(duration / SEGMENT_SEC))
        print(f"Duration: {duration:.1f}s | Segments: {n_segs}")

        print("Extracting MERT embeddings...")
        embedding = self._extract_embedding(waveform)

        print("Running APEX model...")
        preds = self.model(embedding.unsqueeze(0))

        results = {
            task: float(preds[task].squeeze().cpu())
            for task in preds
        }

        print(f"\n{'─'*50}")
        print(f"  APEX Predictions")
        print(f"{'─'*50}")
        print(f"\n  Popularity:")
        print(f"  {'─'*40}")
        print(f"  {'Streams Score':<20} {results['score_streams']:>8.2f} / 100")
        print(f"  {'Likes Score':<20} {results['score_likes']:>8.2f} / 100")
        print(f"\n  Aesthetic Quality:")
        print(f"  {'─'*40}")
        for dim in ["coherence", "musicality", "memorability", "clarity", "naturalness"]:
            print(f"  {dim.capitalize():<20} {results[dim]:>8.2f} / 5.00")
        print(f"{'─'*50}\n")

        if save_json:
            with open(save_json, "w") as f:
                json.dump({
                    "audio_path" : audio_path,
                    "predictions": results
                }, f, indent=2)
            print(f"Results saved to {save_json}")

        return results



# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX — AI Music Popularity & Aesthetic Predictor")
    parser.add_argument(
        "--audio",
        type     = str,
        required = True,
        help     = "Path to audio file"
    )
    parser.add_argument(
        "--checkpoint",
        type    = str,
        default = DEFAULT_CHECKPOINT,
        help    = f"Path to APEX checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--save_json",
        type    = str,
        default = None,
        help    = "Optional path to save results as JSON"
    )
    parser.add_argument(
        "--device",
        type    = str,
        default = None,
        help    = "Device: cuda or cpu (default: auto-detect)"
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    apex   = APEX(checkpoint_path=args.checkpoint, device=device)
    apex.predict(args.audio, save_json=args.save_json)