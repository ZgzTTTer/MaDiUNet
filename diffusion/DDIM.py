import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.base import BaseDiffusionTrainer_cond, BaseDiffusionSampler_cond
from diffusion.base import extract


class DDIMTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self, ct, cbct):
        return ct


class DDIMSampler_cond(BaseDiffusionSampler_cond):
    def __init__(self, model, beta_1, beta_T, T, ddim_steps=50, ddim_eta=0.0):
        super().__init__(model, beta_1, beta_T, T)
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        # Create timestep sequence
        c = self.T // self.ddim_steps
        self.timesteps = torch.arange(0, self.ddim_steps) * c

    def forward(self, x_T):
        with torch.no_grad():
            x_t = x_T
            out_channels = self.model.out_channels
            ct = x_t[:, :out_channels, :, :]
            cbct = x_t[:, out_channels:, :, :]

            for i in reversed(range(self.ddim_steps)):
                t = torch.full((x_T.shape[0],), self.timesteps[i], device=x_T.device, dtype=torch.long)
                next_t = t - self.T // self.ddim_steps if i > 0 else torch.zeros_like(t)

                # Get alphas for current and next timestep
                at = extract(self.alphas_bar, t, ct.shape)
                at_next = extract(self.alphas_bar, next_t, ct.shape)

                x_t_cat = torch.cat((ct, cbct), 1)
                model_output = self.model(x_t_cat, t)

                # Predict x0
                pred_x0 = (ct - torch.sqrt(1 - at) * model_output) / torch.sqrt(at)

                # Direction pointing to xt
                dir_xt = torch.sqrt(
                    1 - at_next - self.ddim_eta ** 2 * (1 - at) * (1 - at_next) / (1 - at)) * model_output

                if i > 0:
                    noise = self.ddim_eta * torch.randn_like(ct)
                    ct = torch.sqrt(at_next) * pred_x0 + dir_xt + torch.sqrt(1 - at_next) * noise
                else:
                    ct = torch.sqrt(at_next) * pred_x0 + dir_xt

            x_t = torch.cat((ct, cbct), 1)
            return torch.clamp(x_t, -1, 1)


class DiffDDIMTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self, ct, cbct):
        return ct - cbct


class DiffDDIMSampler_cond(BaseDiffusionSampler_cond):
    def __init__(self, model, beta_1, beta_T, T, ddim_steps=50, ddim_eta=0.0):
        super().__init__(model, beta_1, beta_T, T)
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        # Create timestep sequence
        c = self.T // self.ddim_steps
        self.timesteps = torch.arange(0, self.ddim_steps) * c

    def forward(self, x_T):
        with torch.no_grad():
            batch_size = x_T.shape[0]
            device = x_T.device

            out_channels = self.model.out_channels
            ct = x_T[:, :out_channels, :, :]
            cbct = x_T[:, out_channels:, :, :]

            d_t = torch.randn_like(ct)

            for i in reversed(range(self.ddim_steps)):
                t = torch.full((batch_size,), self.timesteps[i], device=device, dtype=torch.long)
                next_t = t - self.T // self.ddim_steps if i > 0 else torch.zeros_like(t)

                # Get alphas for current and next timestep
                at = extract(self.alphas_bar, t, d_t.shape)
                at_next = extract(self.alphas_bar, next_t, d_t.shape)

                x_t = torch.cat((d_t, cbct), dim=1)
                model_output = self.model(x_t, t)

                # Predict x0 (difference)
                pred_x0 = (d_t - torch.sqrt(1 - at) * model_output) / torch.sqrt(at)

                # Direction pointing to xt
                dir_xt = torch.sqrt(
                    1 - at_next - self.ddim_eta ** 2 * (1 - at) * (1 - at_next) / (1 - at)) * model_output

                if i > 0:
                    noise = self.ddim_eta * torch.randn_like(d_t)
                    d_t = torch.sqrt(at_next) * pred_x0 + dir_xt + torch.sqrt(1 - at_next) * noise
                else:
                    d_t = torch.sqrt(at_next) * pred_x0 + dir_xt

            # Final reconstruction
            ct = cbct + d_t
            x_0 = torch.cat((ct, cbct), dim=1)
            return torch.clamp(x_0, -1, 1)