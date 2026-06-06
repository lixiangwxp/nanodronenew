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
from models.actuator_torque_residual import (
    ActuatorTorqueResidualQuadModel,
    AngularWeightedMSELoss,
    torque_supervision_loss,
)
from models.models import PhysQuadModel


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


def build_model_name(train_trajs, name_suffix="", seed=None):
    parts = ["actuator_torque_residual", *train_trajs]
    if name_suffix:
        parts.append(name_suffix)
        if seed is not None:
            parts.append(f"seed{seed}")
    return "_".join(parts)


def build_variant(args):
    tags = []
    if args.disable_actuator_memory:
        tags.append("no_memory")
    if args.disable_torque_residual:
        tags.append("no_torque_residual")
    if args.fixed_torque_gate:
        tags.append("fixed_gate")
    if args.freeze_omega:
        tags.append("freeze_omega")
    return "actuator_torque_residual" if not tags else "actuator_torque_residual_" + "_".join(tags)


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
    parser = argparse.ArgumentParser(description="Train actuator-aware torque residual model")
    parser.add_argument("--train-trajs", "--train_trajs", dest="train_trajs", type=str, default='["random", "square", "chirp"]')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", "--lr_end", dest="lr_end", type=float, default=3e-6)
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=96)
    parser.add_argument("--num-layers", "--num_layers", dest="num_layers", type=int, default=3)
    parser.add_argument("--rot-weight", "--rot_weight", dest="rot_weight", type=float, default=3.0)
    parser.add_argument("--omega-weight", "--omega_weight", dest="omega_weight", type=float, default=8.0)
    parser.add_argument("--tau-loss-weight", "--tau_loss_weight", dest="tau_loss_weight", type=float, default=0.02)
    parser.add_argument("--residual-reg-weight", "--residual_reg_weight", dest="residual_reg_weight", type=float, default=0.001)
    parser.add_argument("--alpha-init", "--alpha_init", dest="alpha_init", type=float, default=0.88)
    parser.add_argument("--torque-gate-init", "--torque_gate_init", dest="torque_gate_init", type=float, default=0.05)
    parser.add_argument("--max-residual-torque", "--max_residual_torque", dest="max_residual_torque", type=float, default=0.45)
    parser.add_argument("--max-memory-torque", "--max_memory_torque", dest="max_memory_torque", type=float, default=0.25)
    parser.add_argument("--disable-actuator-memory", "--disable_actuator_memory", dest="disable_actuator_memory", action="store_true")
    parser.add_argument("--disable-torque-residual", "--disable_torque_residual", dest="disable_torque_residual", action="store_true")
    parser.add_argument("--fixed-torque-gate", "--fixed_torque_gate", dest="fixed_torque_gate", action="store_true")
    parser.add_argument("--freeze-omega", "--freeze_omega", dest="freeze_omega", action="store_true")
    parser.add_argument("--name-suffix", "--experiment-suffix", "--experiment_suffix", dest="name_suffix", type=str, default="")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_trajs = parse_json_list(args.train_trajs)
    valid_trajs = train_trajs
    train_runs = [1, 2, 3]
    valid_runs = [4]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dt = 0.01
    model_name = build_model_name(train_trajs, args.name_suffix, args.seed)
    variant = build_variant(args)
    print(f"Model name: {model_name}")
    print(f"Variant: {variant}")

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

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    phys_model = PhysQuadModel(build_phys_params(), dt).to(device)
    model = ActuatorTorqueResidualQuadModel(
        phys=phys_model,
        x_scaler=train_dataset.x_scaler,
        u_scaler=train_dataset.u_scaler,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        alpha_init=args.alpha_init,
        torque_gate_init=args.torque_gate_init,
        max_residual_torque=args.max_residual_torque,
        max_memory_torque=args.max_memory_torque,
        use_actuator_memory=not args.disable_actuator_memory,
        use_torque_residual=not args.disable_torque_residual,
        learn_torque_gate=not args.fixed_torque_gate,
        freeze_omega=args.freeze_omega,
    ).to(device)

    print("Initialized ActuatorTorqueResidualQuadModel")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = AngularWeightedMSELoss(
        lambda_=0.04,
        rot_weight=args.rot_weight,
        omega_weight=args.omega_weight,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_end,
    )

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
            group="actuator-torque-residual",
            name=model_name,
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            mode=args.wandb_mode,
            dir=str(PROJECT_ROOT / "out" / "wandb"),
            config={
                "model_name": model_name,
                "model_type": "actuator_torque_residual",
                "variant": variant,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lr_end": args.lr_end,
                "weight_decay": 1e-5,
                "scheduler": "CosineAnnealingLR",
                "horizon": args.horizon,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "rot_weight": args.rot_weight,
                "omega_weight": args.omega_weight,
                "tau_loss_weight": args.tau_loss_weight,
                "residual_reg_weight": args.residual_reg_weight,
                "alpha_init": args.alpha_init,
                "torque_gate_init": args.torque_gate_init,
                "max_residual_torque": args.max_residual_torque,
                "max_memory_torque": args.max_memory_torque,
                "use_actuator_memory": not args.disable_actuator_memory,
                "use_torque_residual": not args.disable_torque_residual,
                "learn_torque_gate": not args.fixed_torque_gate,
                "freeze_omega": args.freeze_omega,
                "trainable_params": trainable_params,
                "seed": args.seed,
                "scaler_dir": str(scaler_dir),
                "git_commit": git_head,
                "device": str(device),
                "command": " ".join(sys.argv),
                "log_path": os.environ.get("LOG_PATH", ""),
            },
        )
        wandb.watch(model, log="gradients", log_freq=50)

    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_loss = 0.0
        train_state = 0.0
        train_tau = 0.0
        train_reg = 0.0
        for x0, u_seq, x_seq in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
        ):
            x0 = x0.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_seq, aux = model.forward_with_aux(x0, u_seq)
            state_loss = criterion(pred_seq, x_seq)
            tau_loss = torque_supervision_loss(model, x0, x_seq, aux)
            reg_loss = aux["delta_tau_norm"].pow(2).mean() + 0.5 * aux["memory_delta_norm"].pow(2).mean()
            loss = state_loss + args.tau_loss_weight * tau_loss + args.residual_reg_weight * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_state += state_loss.item()
            train_tau += tau_loss.item()
            train_reg += reg_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_state = train_state / len(train_loader)
        avg_train_tau = train_tau / len(train_loader)
        avg_train_reg = train_reg / len(train_loader)

        model.eval()
        valid_loss = 0.0
        valid_state = 0.0
        valid_tau = 0.0
        valid_reg = 0.0
        with torch.no_grad():
            for x0, u_seq, x_seq in valid_loader:
                x0 = x0.to(device, non_blocking=True)
                u_seq = u_seq.to(device, non_blocking=True)
                x_seq = x_seq.to(device, non_blocking=True)
                pred_seq, aux = model.forward_with_aux(x0, u_seq)
                state_loss = criterion(pred_seq, x_seq)
                tau_loss = torque_supervision_loss(model, x0, x_seq, aux)
                reg_loss = aux["delta_tau_norm"].pow(2).mean() + 0.5 * aux["memory_delta_norm"].pow(2).mean()
                loss = state_loss + args.tau_loss_weight * tau_loss + args.residual_reg_weight * reg_loss
                valid_loss += loss.item()
                valid_state += state_loss.item()
                valid_tau += tau_loss.item()
                valid_reg += reg_loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_state = valid_state / len(valid_loader)
        avg_valid_tau = valid_tau / len(valid_loader)
        avg_valid_reg = valid_reg / len(valid_loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_sec = time.perf_counter() - epoch_start
        gpu_mem_peak_mb = 0.0
        if device.type == "cuda":
            gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

        alpha = model.actuator_alpha().detach().cpu().tolist()
        torque_gate = model.torque_gate().detach().cpu().tolist()
        print(
            f"Epoch {epoch + 1}, LR={current_lr:.2e}, "
            f"Train={avg_train_loss:.6f}, TrainState={avg_train_state:.6f}, "
            f"TrainTau={avg_train_tau:.6f}, Valid={avg_valid_loss:.6f}, "
            f"ValidState={avg_valid_state:.6f}, ValidTau={avg_valid_tau:.6f}, "
            f"Alpha={alpha}, TorqueGate={torque_gate}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/state_loss": avg_train_state,
                    "train/tau_loss": avg_train_tau,
                    "train/reg_loss": avg_train_reg,
                    "valid/loss": avg_valid_loss,
                    "valid/state_loss": avg_valid_state,
                    "valid/tau_loss": avg_valid_tau,
                    "valid/reg_loss": avg_valid_reg,
                    "train/lr": current_lr,
                    "train/epoch_sec": epoch_sec,
                    "train/gpu_mem_peak_mb": gpu_mem_peak_mb,
                    "actuator/alpha_x": alpha[0],
                    "actuator/alpha_y": alpha[1],
                    "actuator/alpha_z": alpha[2],
                    "actuator/torque_gate_x": torque_gate[0],
                    "actuator/torque_gate_y": torque_gate[1],
                    "actuator/torque_gate_z": torque_gate[2],
                },
                step=epoch + 1,
            )

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            checkpoint = {
                "model_state": model.state_dict(),
                "config": {
                    "variant": variant,
                    "dt": dt,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "alpha_init": args.alpha_init,
                    "torque_gate_init": args.torque_gate_init,
                    "max_residual_torque": args.max_residual_torque,
                    "max_memory_torque": args.max_memory_torque,
                    "use_actuator_memory": not args.disable_actuator_memory,
                    "use_torque_residual": not args.disable_torque_residual,
                    "learn_torque_gate": not args.fixed_torque_gate,
                    "freeze_omega": args.freeze_omega,
                    "rot_weight": args.rot_weight,
                    "omega_weight": args.omega_weight,
                    "tau_loss_weight": args.tau_loss_weight,
                    "residual_reg_weight": args.residual_reg_weight,
                    "horizon": args.horizon,
                    "batch_size": args.batch_size,
                    "train_trajs": train_trajs,
                    "valid_trajs": valid_trajs,
                    "seed": args.seed,
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": best_epoch,
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
            print(f"Saved best model at epoch {best_epoch} with valid loss {avg_valid_loss:.6f}")

        scheduler.step()

    print(f"TRAIN_DONE best_epoch={best_epoch} best_val={best_val_loss:.6f} model_path={model_path}")
    if wandb_run is not None:
        wandb_run.summary["best/valid_loss"] = best_val_loss
        wandb_run.summary["best/checkpoint_path"] = str(model_path)
        artifact = wandb.Artifact(f"{model_name}-checkpoint", type="model")
        artifact.add_file(str(model_path))
        wandb_run.log_artifact(artifact, aliases=["best", "latest"])
        wandb_run.finish()


if __name__ == "__main__":
    main()
