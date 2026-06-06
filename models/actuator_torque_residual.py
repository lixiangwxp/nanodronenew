import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import BaseQuadModel, PhysQuadModel


class ActuatorTorqueResidualQuadModel(BaseQuadModel):
    """
    Physics model with an interpretable first-order actuator torque state.

    The model operates on the official normalized dataset representation during
    training and evaluation, but it performs dynamics integration in real units.
    The neural component predicts bounded residual torques rather than arbitrary
    12D state corrections.
    """

    def __init__(
        self,
        phys: PhysQuadModel,
        x_scaler,
        u_scaler,
        hidden_dim=64,
        num_layers=3,
        alpha_init=0.85,
        torque_gate_init=0.05,
        max_residual_torque=0.35,
        max_memory_torque=0.20,
        use_actuator_memory=True,
        use_torque_residual=True,
        learn_torque_gate=True,
        freeze_omega=False,
    ):
        super().__init__(phys.dt)
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be between 0 and 1")
        if not 0.0 < torque_gate_init < 1.0:
            raise ValueError("torque_gate_init must be between 0 and 1")

        self.phys = phys
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_residual_torque = float(max_residual_torque)
        self.max_memory_torque = float(max_memory_torque)
        self.use_actuator_memory = bool(use_actuator_memory)
        self.use_torque_residual = bool(use_torque_residual)
        self.learn_torque_gate = bool(learn_torque_gate)
        self.freeze_omega = bool(freeze_omega)

        x_mean, x_scale = self._scaler_tensors(x_scaler)
        u_mean, u_scale = self._scaler_tensors(u_scaler)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)
        self.register_buffer("max_torque", phys.max_torque.detach().clone().float())

        input_dim = 12 + 4 + 3 + 3
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        alpha_init = torch.full((3,), float(alpha_init), dtype=torch.float32)
        logit = torch.log(alpha_init / (1.0 - alpha_init))
        self.logit_alpha = nn.Parameter(logit)

        gate_init = torch.full((3,), float(torque_gate_init), dtype=torch.float32)
        gate_logit = torch.log(gate_init / (1.0 - gate_init))
        if self.learn_torque_gate:
            self.logit_torque_gate = nn.Parameter(gate_logit)
        else:
            self.register_buffer("logit_torque_gate", gate_logit)

    @staticmethod
    def _scaler_tensors(scaler):
        mean = torch.as_tensor(getattr(scaler, "mean_"), dtype=torch.float32)
        scale = torch.as_tensor(getattr(scaler, "scale_"), dtype=torch.float32)
        return mean, scale

    def actuator_alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def torque_gate(self):
        return torch.sigmoid(self.logit_torque_gate)

    def x_denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean

    def x_normed(self, x_real):
        return (x_real - self.x_mean) / self.x_scale

    def u_denorm(self, u_norm):
        return u_norm * self.u_scale + self.u_mean

    def nominal_thrust_torque(self, u_mot):
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
        tau = torch.stack([tau_x, tau_y, tau_z], dim=1)
        return thrust, tau

    def forward(self, x0, u_seq):
        pred, _ = self.forward_with_aux(x0, u_seq)
        return pred

    def forward_with_aux(self, x0, u_seq):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        x = x0 if x0.ndim == 2 else x0.squeeze(1)

        u0_real = self.u_denorm(u_seq[:, 0, :])
        _, tau0 = self.nominal_thrust_torque(u0_real)
        z_tau = tau0

        preds = []
        residual_terms = []
        memory_terms = []
        tau_eff_terms = []
        for t in range(u_seq.shape[1]):
            x, z_tau, aux = self._one_step_with_actuator(x, u_seq[:, t, :], z_tau)
            preds.append(x.unsqueeze(1))
            residual_terms.append(aux["delta_tau_norm"].unsqueeze(1))
            memory_terms.append(aux["memory_delta_norm"].unsqueeze(1))
            tau_eff_terms.append(aux["tau_eff_norm"].unsqueeze(1))

        aux = {
            "delta_tau_norm": torch.cat(residual_terms, dim=1),
            "memory_delta_norm": torch.cat(memory_terms, dim=1),
            "tau_eff_norm": torch.cat(tau_eff_terms, dim=1),
        }
        return torch.cat(preds, dim=1), aux

    def _one_step_with_actuator(self, x_norm, u_norm, z_tau):
        x_real = self.x_denorm(x_norm)
        u_real = self.u_denorm(u_norm)
        thrust, tau_nom = self.nominal_thrust_torque(u_real)

        z_tau_norm = z_tau / self.max_torque
        tau_nom_norm = tau_nom / self.max_torque
        features = torch.cat([x_norm, u_norm, z_tau_norm, tau_nom_norm], dim=-1)
        raw = self.out(self.mlp(features))

        memory_delta_norm = self.max_memory_torque * torch.tanh(raw[:, :3])
        delta_tau_norm = self.max_residual_torque * torch.tanh(raw[:, 3:])
        if not self.use_actuator_memory:
            memory_delta_norm = torch.zeros_like(memory_delta_norm)
        if not self.use_torque_residual:
            delta_tau_norm = torch.zeros_like(delta_tau_norm)
        alpha = self.actuator_alpha().view(1, 3)

        z_tau_next = alpha * z_tau + (1.0 - alpha) * tau_nom + memory_delta_norm * self.max_torque
        tau_unscaled = z_tau_next + delta_tau_norm * self.max_torque
        tau_eff = self.torque_gate().view(1, 3) * tau_unscaled

        x_next_real = self._integrate_real(x_real, thrust, tau_eff)
        x_next_norm = self.x_normed(x_next_real)
        aux = {
            "delta_tau_norm": delta_tau_norm,
            "memory_delta_norm": memory_delta_norm,
            "tau_eff_norm": tau_eff / self.max_torque,
        }
        return x_next_norm, z_tau_next, aux

    def _integrate_real(self, x, thrust, tau):
        pos = x[:, 0:3]
        vel = x[:, 3:6]
        so3 = x[:, 6:9]
        omega = x[:, 9:12]
        quat = self.phys.so3_log_to_quat(so3)

        dt = self.dt

        def f(pos_i, vel_i, quat_i, omega_i):
            thrust_b = torch.zeros_like(vel_i)
            thrust_b[:, 2] = torch.clamp(thrust, min=0.0)
            thrust_w = self.phys.quat_rotate(quat_i, thrust_b)
            acc = (thrust_w - self.phys.m * self.phys.gravity) / self.phys.m
            quat_dot = self.phys.quat_derivative(quat_i, omega_i)
            if self.freeze_omega:
                omega_dot = torch.zeros_like(omega_i)
            else:
                j_omega = omega_i @ self.phys.J.T
                rhs = tau - torch.cross(omega_i, j_omega, dim=-1)
                omega_dot = torch.linalg.solve(self.phys.J, rhs.unsqueeze(-1)).squeeze(-1)
                omega_dot = torch.clamp(omega_dot, min=-2500.0, max=2500.0)
            return vel_i, acc, quat_dot, omega_dot

        v1, a1, qd1, w1 = f(pos, vel, quat, omega)
        v2, a2, qd2, w2 = f(
            pos + 0.5 * dt * v1,
            vel + 0.5 * dt * a1,
            F.normalize(quat + 0.5 * dt * qd1, dim=-1),
            omega + 0.5 * dt * w1,
        )
        v3, a3, qd3, w3 = f(
            pos + 0.5 * dt * v2,
            vel + 0.5 * dt * a2,
            F.normalize(quat + 0.5 * dt * qd2, dim=-1),
            omega + 0.5 * dt * w2,
        )
        v4, a4, qd4, w4 = f(
            pos + dt * v3,
            vel + dt * a3,
            F.normalize(quat + dt * qd3, dim=-1),
            omega + dt * w3,
        )

        pos_next = pos + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        vel_next = vel + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        omega_next = omega + (dt / 6.0) * (w1 + 2.0 * w2 + 2.0 * w3 + w4)
        quat_next = quat + (dt / 6.0) * (qd1 + 2.0 * qd2 + 2.0 * qd3 + qd4)
        quat_next = F.normalize(quat_next, dim=-1)
        so3_next = self.phys.quat_to_so3_log(quat_next)

        return torch.cat([pos_next, vel_next, so3_next, omega_next], dim=-1)


class AngularWeightedMSELoss(nn.Module):
    def __init__(self, lambda_=0.05, rot_weight=3.0, omega_weight=6.0):
        super().__init__()
        self.lambda_ = float(lambda_)
        weights = torch.ones(12, dtype=torch.float32)
        weights[6:9] = float(rot_weight)
        weights[9:12] = float(omega_weight)
        self.register_buffer("state_weights", weights)

    def forward(self, pred, true):
        if pred.shape != true.shape:
            raise ValueError("pred and true must have same shape")
        horizon = pred.shape[1]
        h_weights = torch.exp(-self.lambda_ * torch.arange(horizon, device=pred.device))
        h_weights = h_weights / h_weights.mean()
        err = (pred - true) ** 2
        err = err * h_weights.view(1, horizon, 1) * self.state_weights.view(1, 1, 12)
        return err.mean()


def torque_supervision_loss(model, x0_norm, true_seq_norm, aux):
    x_scale = model.x_scale.view(1, 1, -1)
    x_mean = model.x_mean.view(1, 1, -1)
    x0_real = x0_norm.squeeze(1) * model.x_scale + model.x_mean
    true_real = true_seq_norm * x_scale + x_mean
    full = torch.cat([x0_real.unsqueeze(1), true_real], dim=1)
    omega = full[:, :-1, 9:12]
    omega_next = full[:, 1:, 9:12]
    omega_dot = (omega_next - omega) / model.dt
    j_omega = omega @ model.phys.J.T
    tau_required = (omega_dot.unsqueeze(-2) @ model.phys.J.T).squeeze(-2)
    tau_required = tau_required + torch.cross(omega, j_omega, dim=-1)
    tau_required_norm = torch.clamp(tau_required / model.max_torque.view(1, 1, 3), -3.0, 3.0)
    tau_eff_norm = torch.clamp(aux["tau_eff_norm"], -3.0, 3.0)
    return ((tau_eff_norm - tau_required_norm) ** 2).mean()
