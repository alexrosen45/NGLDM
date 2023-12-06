import torch
from torchvision.utils import save_image
from unet import Unet
from ngldm import NGLDM
import os


def load_model(model_path, device = "cuda") -> NGLDM:
    ngldm = NGLDM(eps_model=Unet(3, 3, n_feat=128), betas=(1e-4, 0.02), T=1000)
    ngldm.load_state_dict(torch.load(model_path, map_location=device))
    ngldm.to(device)
    return ngldm


def generate_images(model: NGLDM, num_images, device = "cuda") -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        generated_images = model.sample(num_images, (3, 32, 32), device)
    return generated_images


def save_generated_images(images: torch.Tensor, directory, prefix = "gen_image") -> None:
    for i, image in enumerate(images):
        save_image(image, f"{directory}/{prefix}_{i}.png", normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    device = "cuda"
    model_path = "./models/cifar10/normal/ngldm_cifar10_100.pth"
    num_images = 10
    output_dir = "./generated_images/normal"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ngldm_model = load_model(model_path, device)

    images = generate_images(ngldm_model, num_images, device)
    save_generated_images(images, output_dir)
