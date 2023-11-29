import torch
from torchvision.utils import save_image
from minidiffusion.unet import NaiveUnet
from minidiffusion.ddpm import DDPM
import os


def load_model(model_path, device = "cuda") -> DDPM:
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.to(device)
    return ddpm


def generate_images(model: DDPM, num_images, device = "cuda") -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        generated_images = model.sample(num_images, (3, 32, 32), device)
    return generated_images


def save_generated_images(images: torch.Tensor, directory, prefix = "gen_image") -> None:
    for i, image in enumerate(images):
        save_image(image, f"{directory}/{prefix}_{i}.png", normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    device = "cuda"
    model_path = "./models/test/ddpm_cifar.pth"
    num_images = 10  # Number of images to generate
    output_dir = "./generated_images/test1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the trained model
    ddpm_model = load_model(model_path, device)

    # Generate images
    images = generate_images(ddpm_model, num_images, device)

    # Save the generated images
    save_generated_images(images, output_dir)
