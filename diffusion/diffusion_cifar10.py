from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from unet import Unet
from ngldm import NGLDM
import os


def train_cifar10(
    epochs = 250,
    device = "cuda",
    batch_size = 512,
    lr = 1e-5,
    load_pth = None,  # Example: ./models/normal/ngldm_cifar10.pth
    noise_type = "normal",
    model_name = "ngldm_cifar10",
    model_dir = "./models",
    out_dir = "./results",
):
    ngldm = NGLDM(eps_model=Unet(3, 3, n_feat=128), betas=(1e-4, 0.02), T=1000)

    if load_pth is not None:
        ngldm.load_state_dict(torch.load(load_pth))

    ngldm.to(device)

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
    optim = torch.optim.Adam(ngldm.parameters(), lr=lr)

    for i in range(epochs):
        ngldm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ngldm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"Epoch: {i}, loss: {loss_ema:.4f}")
            optim.step()

        ngldm.eval()
        with torch.no_grad():
            xh = ngldm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"{out_dir}/{noise_type}/ngldm_sample{i}.png")

            torch.save(ngldm.state_dict(), f"{model_dir}/{noise_type}/{model_name}_{i}.pth")


if __name__ == "__main__":
    noise_type = "normal"
    model_dir = "./models/cifar10"
    out_dir = "./results"

    if not os.path.exists(out_dir + "/" + noise_type):
        os.makedirs(out_dir + "/" + noise_type)

    if not os.path.exists(model_dir + "/" + noise_type):
        os.makedirs(model_dir + "/" + noise_type)

    train_cifar10(
        epochs = 200,
        device = "cuda",
        batch_size = 10,
        lr = 1e-5,
        load_pth = None,
        noise_type = noise_type,
        model_name = "ngldm_cifar10",
        model_dir = model_dir,
        out_dir = out_dir,
    )