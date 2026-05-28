import numpy as np
import pandas as pd

from utils.metrics_utils import compute_errors


def so3_log_to_quat_np(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_theta = 0.5 * theta
    small = theta < 1e-8
    scale = np.empty_like(theta)
    np.divide(
        np.sin(half_theta),
        theta,
        out=scale,
        where=~small,
    )
    scale = np.where(small, 0.5 - theta**2 / 48.0, scale)
    xyz = rotvec * scale
    w = np.cos(half_theta)
    return np.concatenate([xyz, w], axis=-1)


def add_rotation_columns(df):
    df = df.copy()
    new_cols = {}

    for rx_col in [col for col in df.columns if col.startswith("rx")]:
        suffix = rx_col[2:]
        ry_col = f"ry{suffix}"
        rz_col = f"rz{suffix}"
        if ry_col not in df.columns or rz_col not in df.columns:
            continue

        rotvec = df[[rx_col, ry_col, rz_col]].to_numpy(float)
        quat = so3_log_to_quat_np(rotvec)
        new_cols[f"qx{suffix}"] = quat[:, 0]
        new_cols[f"qy{suffix}"] = quat[:, 1]
        new_cols[f"qz{suffix}"] = quat[:, 2]
        new_cols[f"qw{suffix}"] = quat[:, 3]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def compute_simerr_for_horizon(metrics, max_horizon):
    return {
        key: float(sum(metrics[key][h] for h in range(1, max_horizon + 1)))
        for key in ["pos", "vel", "rot", "omega"]
    }


def flatten_eval_metrics(metrics, max_horizon, h_targets=(1, 10, 50)):
    flat = {}
    for key in ["pos", "vel", "rot", "omega"]:
        for horizon in h_targets:
            if horizon <= max_horizon:
                flat[f"eval/{key}_h{horizon}"] = float(metrics[key][horizon])
    simerr = compute_simerr_for_horizon(metrics, max_horizon)
    for key, value in simerr.items():
        flat[f"eval/simerr_{key}"] = value
    return flat


def metrics_summary_frame(metrics, max_horizon, h_targets=(1, 10, 50)):
    rows = []
    simerr = compute_simerr_for_horizon(metrics, max_horizon)
    for key in ["pos", "vel", "rot", "omega"]:
        for horizon in h_targets:
            if horizon <= max_horizon:
                rows.append(
                    {
                        "metric": key,
                        "horizon": horizon,
                        "value": float(metrics[key][horizon]),
                    }
                )
        rows.append(
            {
                "metric": key,
                "horizon": "simerr",
                "value": simerr[key],
            }
        )
    return pd.DataFrame(rows)


def compute_prediction_metrics(df_pred, max_horizon, h_targets=(1, 10, 50)):
    df_eval = add_rotation_columns(df_pred)
    metrics = compute_errors(df_eval, max_horizon)
    flat = flatten_eval_metrics(metrics, max_horizon, h_targets=h_targets)
    summary = metrics_summary_frame(metrics, max_horizon, h_targets=h_targets)
    return metrics, flat, summary
