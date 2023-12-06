import torch
import torch.nn as nn
import noise


class NGLDM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas,
        T,
        sampler=noise.normal,
        criterion: nn.Module = nn.MSELoss(),
    ):
        super(NGLDM, self).__init__()
        self.eps_model = eps_model

        for k, v in ngldm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.sampler = sampler
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.randint(1, self.T + 1, (x.shape[0],)).to(x.device)
        epsilon_t = self.sampler(x.shape, x.device)
        x_t = (self.sqrtab[t, None, None, None] * x + self.sqrtmab[t, None, None, None] * epsilon_t)
        
        # MSE loss
        return self.criterion(epsilon_t, self.eps_model(x_t, t / self.T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = self.sampler((n_sample, *size), device)

        for i in range(self.T, 0, -1):
            z = self.sampler((n_sample, *size), device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.tensor(i / self.T).to(device).repeat(n_sample, 1))
            x_i = (self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)
        
        # Augment this to have a decoder if you want
        return x_i


def ngldm_schedules(beta1: float, beta2: float, T: int):
    assert beta1 < beta2 < 1.0

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
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }