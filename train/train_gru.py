import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import QuadDataset, combine_concat_dataset
from models.models import PhysQuadModel, ResidualQuadModel
from models.models_gru import RawGRUPhysResModel
from train.losses import WeightedMSELoss


def parse_json_list(raw_value):
    parsed = json.loads(raw_value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list or string")
    return parsed


def build_phys_params():
    return {
        "g": 9.81,
        "m": 0.045,
        "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
        "thrust_to_weight": 2.0,
        "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
    }


def load_split(trajs, runs, data_dir, horizon, split):
    datasets = []
    for traj in trajs:
        for run in runs:
            file_path = data_dir / f"{traj}_20251017_run{run}.csv"
            try:
                df = pd.read_csv(file_path)
                datasets.append(QuadDataset(df, horizon=horizon))
            except Exception as exc:
                print(f"Skipped {file_path}: {exc}")
    if not datasets:
        raise RuntimeError(f"No datasets loaded for {split} from {data_dir}")
    print(f"Loaded {len(datasets)} datasets for {split}")
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Train raw-GRU Physics+Residual model")
    parser.add_argument("--train_trajs", type=str, default='["random", "square", "chirp"]')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--name-suffix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    seed = 3407
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_trajs = parse_json_list(args.train_trajs)
    valid_trajs = train_trajs
    train_runs = [1, 2, 3]
    valid_runs = [4]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dt = 0.01
    lr_start = 1e-5
    lr_end = 1e-8
    residual_input_dim = 4 + args.gru_hidden_dim

    name_parts = ["raw_gru", *train_trajs]
    if args.name_suffix:
        name_parts.append(args.name_suffix)
    model_name = "_".join(name_parts)
    print(f"Model name: {model_name}")

    model_dir = PROJECT_ROOT / "out" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    print(f"Model will be saved to: {model_path}")

    scaler_dir = PROJECT_ROOT / "scalers" / model_name
    scaler_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(
        train_trajs,
        train_runs,
        PROJECT_ROOT / "data" / "train",
        args.horizon,
        "train",
    )
    valid_ds = load_split(
        valid_trajs,
        valid_runs,
        PROJECT_ROOT / "data" / "train",
        args.horizon,
        "valid",
    )

    train_dataset = combine_concat_dataset(
        ConcatDataset(train_ds),
        scale=True,
        fold="train",
        scaler_dir=scaler_dir,
    )
    valid_dataset = combine_concat_dataset(
        ConcatDataset(valid_ds),
        scale=True,
        fold="valid",
        scaler_dir=scaler_dir,
    )

    traj_info = {"train_trajs": train_trajs, "valid_trajs": valid_trajs}
    with open(scaler_dir / "trajectories.json", "w") as handle:
        json.dump(traj_info, handle, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    phys_model = PhysQuadModel(build_phys_params(), dt).to(device)
    residual_model = ResidualQuadModel(
        input_dim=residual_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=5,
        dt=dt,
    ).to(device)
    model = RawGRUPhysResModel(
        phys=phys_model,
        residual=residual_model,
        x_scaler=train_dataset.x_scaler,
        u_scaler=train_dataset.u_scaler,
        hidden_dim=args.gru_hidden_dim,
    ).to(device)

    print("Initialized RawGRUPhysResModel")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr_start, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=lr_end,
    )
    criterion = WeightedMSELoss(lambda_=0.1)
    best_val_loss = float("inf")
    wandb = None
    wandb_run = None
    if args.wandb:
        import wandb

        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        ).stdout.strip()

        wandb_run = wandb.init(
            project="nanodrone",
            group="lag-gru-ablation",
            name=model_name,
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            mode=args.wandb_mode,
            dir=str(PROJECT_ROOT / "out" / "wandb"),
            config={
                "model_name": model_name,
                "model_type": "raw_gru",
                "variant": "raw_gru",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": lr_start,
                "lr_end": lr_end,
                "weight_decay": 1e-5,
                "scheduler": "CosineAnnealingLR",
                "horizon": args.horizon,
                "hidden_dim": args.hidden_dim,
                "gru_hidden_dim": args.gru_hidden_dim,
                "trainable_params": trainable_params,
                "seed": seed,
                "scaler_dir": str(scaler_dir),
                "git_commit": git_head,
                "device": str(device),
                "command": " ".join(sys.argv),
                "log_path": os.environ.get("LOG_PATH", ""),
            },
        )
        wandb.watch(model, log="gradients", log_freq=50)

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_loss = 0.0
        for x0, u_seq, x_seq in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
        ):
            x0 = x0.to(device)
            u_seq = u_seq.to(device)
            x_seq = x_seq.to(device)

            optimizer.zero_grad()
            pred_seq = model(x0, u_seq)
            loss = criterion(pred_seq, x_seq)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x0, u_seq, x_seq in valid_loader:
                x0 = x0.to(device)
                u_seq = u_seq.to(device)
                x_seq = x_seq.to(device)
                pred_seq = model(x0, u_seq)
                loss = criterion(pred_seq, x_seq)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_sec = time.perf_counter() - epoch_start
        gpu_mem_peak_mb = 0.0
        if device.type == "cuda":
            gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(
            f"Epoch {epoch + 1}, LR={current_lr:.2e}, "
            f"Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "valid/loss": avg_valid_loss,
                    "train/lr": current_lr,
                    "train/epoch_sec": epoch_sec,
                    "train/gpu_mem_peak_mb": gpu_mem_peak_mb,
                },
                step=epoch + 1,
            )

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "config": {
                    "variant": "raw_gru",
                    "dt": dt,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": 5,
                    "residual_input_dim": residual_input_dim,
                    "gru_hidden_dim": args.gru_hidden_dim,
                    "horizon": args.horizon,
                    "batch_size": args.batch_size,
                    "train_trajs": train_trajs,
                    "valid_trajs": valid_trajs,
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": best_val_loss,
                "model_name": model_name,
                "scaler_dir": str(scaler_dir),
                "wandb": {
                    "run_id": wandb_run.id,
                    "project": "nanodrone",
                    "url": wandb_run.url,
                }
                if wandb_run is not None
                else {},
            }
            torch.save(checkpoint, model_path)
            print(f"Saved best model at epoch {epoch + 1} with valid loss {avg_valid_loss:.6f}")

        scheduler.step()

    if wandb_run is not None:
        wandb_run.summary["best/valid_loss"] = best_val_loss
        wandb_run.summary["best/checkpoint_path"] = str(model_path)
        artifact = wandb.Artifact(f"{model_name}-checkpoint", type="model")
        artifact.add_file(str(model_path))
        wandb_run.log_artifact(artifact, aliases=["best", "latest"])
        wandb_run.finish()


if __name__ == "__main__":
    main()
