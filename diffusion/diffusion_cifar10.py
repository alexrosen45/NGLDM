from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from minidiffusion.unet import NaiveUnet
from minidiffusion.ddpm import DDPM
import os


def train_cifar10(
    epochs = 250,
    device = "cuda",
    batch_size = 512,
    lr = 1e-5,
    load_pth = None,  # Example: ./models/normal/ngddpm_cifar10.pth
    noise_type = "normal",
    model_name = "ngddpm_cifar10",
    model_dir = "./models",
    out_dir = "./results",
):
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for i in range(epochs):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"{out_dir}/{noise_type}/ngddpm_sample{i}.png")

            torch.save(ddpm.state_dict(), f"{model_dir}/{noise_type}/{model_name}.pth")


if __name__ == "__main__":
    noise_type = "normal"
    out_dir = "./results"

    if not os.path.exists(out_dir + "/" + noise_type):
        os.makedirs(out_dir + "/" + noise_type)

    train_cifar10(
        epochs = 1,
        device = "cuda",
        batch_size = 1,
        lr = 1e-5,
        load_pth = None,
        noise_type = "normal",
        model_name = "ngddpm_cifar10",
        model_dir = "./models",
        out_dir = "./results",
    )