from typing import Dict, Tuple

import torch
import torch.nn as nn
import math


class NGLDM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(NGLDM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ngldm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        # Sample timestep uniformly from {0,...,T}
        t = torch.randint(1, self.T + 1, (x.shape[0],)).to(x.device)
        # Sample epsilon_t from D
        # epsilon_t = torch.normal(mean=0, std=self.var(t.to(torch.int32)))
        # TODO: Schedule
        # mean = 0
        # var = math.sqrt(1 / 2)
        # dist = torch.distributions.Laplace(loc=mean, scale=var)
        low, high = -math.sqrt(3), math.sqrt(3)
        
        dist = torch.distributions.Uniform(low=low, high=high)
        epsilon_t = dist.sample(x.shape).to(x.device)

        # TODO: This should be passed into the latent space
        # x_t is the first parameter of epsilon_{theta_1} where theta_1
        # is the parameterization of our prediction of epsilon_t
        x_t = (self.sqrtab[t, None, None, None] * x + self.sqrtmab[t, None, None, None] * epsilon_t)
        
        # Return MSE loss
        return self.criterion(epsilon_t, self.eps_model(x_t, t / self.T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        low, high = -math.sqrt(3), math.sqrt(3)
        
        dist = torch.distributions.Uniform(low=low, high=high)

        x_i = dist.sample((n_sample, *size)).to(device)

        for i in range(self.T, 0, -1):
            z = dist.sample((n_sample, *size)).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def ngldm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for NGLDM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }