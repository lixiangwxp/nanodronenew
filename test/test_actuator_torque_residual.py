import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import QuadDataset, combine_concat_dataset
from models.actuator_torque_residual import ActuatorTorqueResidualQuadModel
from models.models import PhysQuadModel
from utils.eval_utils import compute_prediction_metrics


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


def find_latest_model(model_root):
    model_files = sorted(
        [
            path
            for path in model_root.iterdir()
            if path.is_file() and path.name.startswith("actuator_torque_residual") and path.suffix == ".pt"
        ],
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    if not model_files:
        raise RuntimeError(f"No actuator_torque_residual checkpoint found in {model_root}")
    return model_files[0]


def load_test_datasets(test_trajs, data_root, horizon):
    datasets = []
    for traj in test_trajs:
        for run in [1, 2, 3]:
            file_path = data_root / f"{traj}_20251017_run{run}.csv"
            try:
                df = pd.read_csv(file_path)
                datasets.append(QuadDataset(df, horizon=horizon))
            except Exception as exc:
                print(f"Skipped {file_path}: {exc}")
    if not datasets:
        raise RuntimeError(f"No test datasets loaded from {data_root}")
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Test actuator-aware torque residual model")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--test-trajs", "--test_trajs", dest="test_trajs", type=str, default='["melon"]')
    parser.add_argument(
        "--out-root",
        "--out_root",
        dest="out_root",
        type=str,
        default=str(PROJECT_ROOT / "out" / "predictions"),
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    test_trajs = parse_json_list(args.test_trajs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_root = PROJECT_ROOT / "out" / "models"
    model_path = Path(args.model_path) if args.model_path else find_latest_model(model_root)
    model_path = model_path.resolve()
    model_name = model_path.stem

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    dt = config.get("dt", 0.01)

    scaler_dir = Path(checkpoint.get("scaler_dir", PROJECT_ROOT / "scalers" / model_name))
    if not scaler_dir.exists():
        scaler_dir = PROJECT_ROOT / "scalers" / model_name

    test_ds = load_test_datasets(
        test_trajs,
        PROJECT_ROOT / "data" / "test",
        args.horizon,
    )
    test_dataset = combine_concat_dataset(
        ConcatDataset(test_ds),
        scale=True,
        fold="test",
        scaler_dir=scaler_dir,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    phys_model = PhysQuadModel(build_phys_params(), dt).to(device)
    model = ActuatorTorqueResidualQuadModel(
        phys=phys_model,
        x_scaler=test_dataset.x_scaler,
        u_scaler=test_dataset.u_scaler,
        hidden_dim=config.get("hidden_dim", 96),
        num_layers=config.get("num_layers", 3),
        alpha_init=config.get("alpha_init", 0.88),
        torque_gate_init=config.get("torque_gate_init", 0.05),
        max_residual_torque=config.get("max_residual_torque", 0.45),
        max_memory_torque=config.get("max_memory_torque", 0.25),
        use_actuator_memory=config.get("use_actuator_memory", True),
        use_torque_residual=config.get("use_torque_residual", True),
        learn_torque_gate=config.get("learn_torque_gate", True),
        freeze_omega=config.get("freeze_omega", False),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x0, u_seq, x_seq_true in test_loader:
            x0 = x0.to(device)
            u_seq = u_seq.to(device)
            x_pred = model(x0, u_seq).cpu()
            preds.append(x_pred)
            trues.append(x_seq_true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    x_scaler = test_dataset.x_scaler
    preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

    state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]
    num_windows = preds.shape[0]
    data = {"t": torch.arange(num_windows, dtype=torch.float32).numpy() * dt}
    for idx, name in enumerate(state_names):
        data[name] = trues[:, 0, idx]

    for h in range(1, args.horizon + 1):
        for idx, name in enumerate(state_names):
            data[f"{name}_pred_h{h}"] = preds[:, h - 1, idx]

    df_pred = pd.DataFrame(data)

    out_root = Path(args.out_root)
    out_dir = out_root / f"{model_name}_model_multistep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{'_'.join(test_trajs)}_multistep.csv"
    df_pred.to_csv(out_path, index=False)

    metrics, flat_metrics, summary_df = compute_prediction_metrics(df_pred, args.horizon)
    summary_path = out_dir / f"{'_'.join(test_trajs)}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    wandb_run = None
    if args.wandb:
        import wandb

        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        ).stdout.strip()

        wandb_init_kwargs = {
            "project": "nanodrone",
            "group": "actuator-torque-residual",
            "name": f"{model_name}-eval",
            "tags": [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            "mode": args.wandb_mode,
            "dir": str(PROJECT_ROOT / "out" / "wandb"),
            "config": {
                "model_name": model_name,
                "model_type": "actuator_torque_residual",
                "variant": config.get("variant", "actuator_torque_residual"),
                "model_path": str(model_path),
                "prediction_path": str(out_path),
                "summary_path": str(summary_path),
                "test_trajs": test_trajs,
                "test_runs": [1, 2, 3],
                "horizon": args.horizon,
                "batch_size": args.batch_size,
                "scaler_dir": str(scaler_dir),
                "git_commit": git_head,
                "device": str(device),
                "command": " ".join(sys.argv),
            },
        }
        wandb_run = wandb.init(**wandb_init_kwargs)
        wandb_run.log(flat_metrics)
        pred_artifact = wandb.Artifact(
            f"{model_name}-prediction",
            type="prediction-csv",
        )
        pred_artifact.add_file(str(out_path))
        wandb_run.log_artifact(pred_artifact, aliases=["latest"])
        summary_artifact = wandb.Artifact(
            f"{model_name}-eval-summary",
            type="eval-summary",
        )
        summary_artifact.add_file(str(summary_path))
        wandb_run.log_artifact(summary_artifact, aliases=["latest"])
        wandb_run.finish()

    print(f"Loaded model: {model_path}")
    print(f"Variant: {config.get('variant', 'actuator_torque_residual')}")
    print(f"Saved predictions to: {out_path}")
    print(f"Saved eval summary to: {summary_path}")
    print(
        "Eval h50: "
        f"pos={metrics['pos'].get(50, float('nan')):.6f}, "
        f"vel={metrics['vel'].get(50, float('nan')):.6f}, "
        f"rot={metrics['rot'].get(50, float('nan')):.6f}, "
        f"omega={metrics['omega'].get(50, float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
