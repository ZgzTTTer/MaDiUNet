import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.base import BaseDiffusionTrainer_cond, BaseDiffusionSampler_cond
from diffusion.base import extract


class DDPMTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self, ct, cbct):
        return ct


class DDPMSampler_cond(BaseDiffusionSampler_cond):
    def forward(self, x_T):
        with torch.no_grad():
            x_t = x_T
            out_channels = self.model.out_channels
            ct = x_t[:, :out_channels, :, :]
            cbct = x_t[:, out_channels:, :, :]

            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, ct.shape)

                x_t_cat = torch.cat((ct, cbct), 1)
                model_output = self.model(x_t_cat, t)
                eps = model_output

                sqrt_alphas_bar_t = torch.sqrt(extract(self.alphas_bar, t, ct.shape))
                sqrt_one_minus_alphas_bar_t = torch.sqrt(1 - extract(self.alphas_bar, t, ct.shape))

                alpha_t = extract(self.alphas, t, ct.shape)
                one_minus_alpha_t = 1 - alpha_t
                mean = (1 / torch.sqrt(alpha_t)) * (ct - (one_minus_alpha_t / sqrt_one_minus_alphas_bar_t) * eps)

                if time_step > 0:
                    noise = torch.randn_like(ct)
                else:
                    noise = 0

                ct = mean + torch.sqrt(var) * noise
                x_t = torch.cat((ct, cbct), 1)

            return torch.clamp(x_t, -1, 1)


class DiffDDPMTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self, ct, cbct):
        return ct - cbct


class DiffDDPMSampler_cond(BaseDiffusionSampler_cond):
    def forward(self, x_T):
        with torch.no_grad():
            batch_size = x_T.shape[0]
            device = x_T.device

            out_channels = self.model.out_channels
            ct = x_T[:, :out_channels, :, :]
            cbct = x_T[:, out_channels:, :, :]

            d_t = torch.randn_like(ct)

            for time_step in reversed(range(self.T)):
                t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)

                x_t = torch.cat((d_t, cbct), dim=1)
                model_output = self.model(x_t, t)
                eps = model_output

                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, d_t.shape)

                sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, d_t.shape)
                sqrt_one_minus_alphas_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, d_t.shape)

                alpha_t = extract(self.alphas, t, d_t.shape)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                one_minus_alpha_t = 1 - alpha_t

                mean = (1 / sqrt_alpha_t) * (d_t - (one_minus_alpha_t / sqrt_one_minus_alphas_bar_t) * eps)

                if time_step > 0:
                    noise = torch.randn_like(d_t)
                    d_t = mean + torch.sqrt(var) * noise
                else:
                    d_t = mean

            # Final reconstruction
            ct = cbct + d_t
            x_0 = torch.cat((ct, cbct), dim=1)
            return torch.clamp(x_0, -1, 1)