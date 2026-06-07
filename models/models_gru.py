import math

import torch
import torch.nn as nn

from models.models import PhysQuadModel, ResidualQuadModel


class RawGRUPhysResModel(nn.Module):
    """Physics + residual model with GRU memory over raw motor inputs."""

    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        hidden_dim=64,
    ):
        super().__init__()
        if not isinstance(phys, PhysQuadModel):
            raise TypeError("phys must be an instance of PhysQuadModel")
        if not isinstance(residual, ResidualQuadModel):
            raise TypeError("residual must be an instance of ResidualQuadModel")

        self.phys = phys
        self.residual = residual
        self.dt = phys.dt
        self.gru_hidden_dim = int(hidden_dim)
        for param in self.phys.parameters():
            param.requires_grad_(False)

        state_dim = residual.out.out_features
        control_dim = 4
        feature_dim = state_dim + control_dim + self.gru_hidden_dim
        self._validate_residual_input(
            state_dim=state_dim,
            control_feat_dim=control_dim + self.gru_hidden_dim,
        )

        x_mean, x_scale = self._scaler_to_tensors(x_scaler, state_dim)
        u_mean, u_scale = self._scaler_to_tensors(u_scaler, control_dim)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)

        self.h_init = nn.Sequential(nn.Linear(state_dim, self.gru_hidden_dim), nn.Tanh())
        self.h_norm = nn.LayerNorm(self.gru_hidden_dim, elementwise_affine=False)
        self.gru_cell = nn.GRUCell(feature_dim, self.gru_hidden_dim)
        self.attn_query = nn.Linear(feature_dim, self.gru_hidden_dim)
        self.attn_key = nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim, bias=False)
        self.attn_gate = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.constant_(self.attn_gate.bias, -2.0)

    @staticmethod
    def _scaler_to_tensors(scaler, dim):
        if scaler is None:
            return (
                torch.zeros(dim, dtype=torch.float32),
                torch.ones(dim, dtype=torch.float32),
            )
        mean = torch.as_tensor(scaler.mean_, dtype=torch.float32)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32)
        return mean, scale

    def _validate_residual_input(self, state_dim, control_feat_dim):
        first_linear = None
        for layer in self.residual.mlp:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break

        expected_in_dim = state_dim + control_feat_dim
        if first_linear is None or first_linear.in_features != expected_in_dim:
            got = None if first_linear is None else first_linear.in_features
            raise ValueError(
                "ResidualQuadModel input dimension mismatch: "
                f"expected first layer input {expected_in_dim}, got {got}."
            )

    def x_denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean

    def x_normed(self, x_real):
        return (x_real - self.x_mean) / self.x_scale

    def u_denorm(self, u_norm):
        return u_norm * self.u_scale + self.u_mean

    def motor_to_phys_diff(self, u_mot):
        omega2 = u_mot ** 2
        thrust = self.phys.Kt * omega2.sum(dim=1)
        tau_x = self.phys.Kt * self.phys.arm * (
            (omega2[:, 2] + omega2[:, 3]) - (omega2[:, 0] + omega2[:, 1])
        )
        tau_y = self.phys.Kt * self.phys.arm * (
            (omega2[:, 1] + omega2[:, 2]) - (omega2[:, 0] + omega2[:, 3])
        )
        tau_z = self.phys.Kc * (
            (omega2[:, 0] + omega2[:, 2]) - (omega2[:, 1] + omega2[:, 3])
        )

        thrust_norm = thrust / self.phys.T_max
        tau_norm = torch.stack([tau_x, tau_y, tau_z], dim=1) / self.phys.max_torque
        return torch.cat([thrust_norm.unsqueeze(1), tau_norm], dim=1)

    def physics_step_from_motors(self, x_real, u_raw_real):
        pos = x_real[:, 0:3]
        vel = x_real[:, 3:6]
        so3 = x_real[:, 6:9]
        omega = x_real[:, 9:12]

        quat = self.phys.so3_log_to_quat(so3)
        x_quat = torch.cat([pos, vel, quat, omega], dim=-1)
        u_phys = self.motor_to_phys_diff(u_raw_real)
        x_next_quat = self.phys._step_from_phys(x_quat, u_phys)

        pos_next = x_next_quat[:, 0:3]
        vel_next = x_next_quat[:, 3:6]
        quat_next = x_next_quat[:, 6:10]
        omega_next = x_next_quat[:, 10:13]
        so3_next = self.phys.quat_to_so3_log(quat_next)
        return torch.cat([pos_next, vel_next, so3_next, omega_next], dim=-1)

    @staticmethod
    def _pack_features(x_norm, u_raw_norm, h):
        return torch.cat([x_norm, u_raw_norm, h], dim=-1)

    def _attend_hidden(self, x_norm, u_raw_norm, h, h_history):
        if not h_history:
            return h
        history = torch.stack(h_history, dim=1)
        query = torch.tanh(self.attn_query(self._pack_features(x_norm, u_raw_norm, h)))
        keys = torch.tanh(self.attn_key(history))
        scores = (keys * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.gru_hidden_dim)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * history).sum(dim=1)
        gate = torch.sigmoid(self.attn_gate(torch.cat([h, context], dim=-1)))
        return h + gate * (context - h)

    def forward(self, x0, u_seq):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        x_norm = x0.squeeze(1) if x0.ndim == 3 else x0

        _, horizon, _ = u_seq.shape
        preds = []
        h = self.h_norm(self.h_init(x_norm))
        h_history = []

        for t in range(horizon):
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)

            x_real = self.x_denorm(x_norm)
            x_phys_next_real = self.physics_step_from_motors(x_real, u_raw_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)

            gru_in = self._pack_features(x_norm, u_raw_norm, h)
            h = self.h_norm(self.gru_cell(gru_in, h))
            h_res = self._attend_hidden(x_norm, u_raw_norm, h, h_history)
            h_history.append(h)

            residual_in = self._pack_features(x_norm, u_raw_norm, h_res)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res

            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            preds.append(x_next_norm.unsqueeze(1))
            x_norm = x_next_norm

        return torch.cat(preds, dim=1)


__all__ = ["RawGRUPhysResModel"]
