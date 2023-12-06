"""
For making images in NGLDM paper Appendix.
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from unet import Unet
from ngldm import NGLDM
import os


def load_model(model_path, device):
    ngldm = NGLDM(eps_model=Unet(3, 3, n_feat=128), betas=(1e-4, 0.02), T=1000)
    ngldm.load_state_dict(torch.load(model_path, map_location=device))
    ngldm.to(device)
    return ngldm


def generate_images(model, num_images, device):
    model.eval()
    with torch.no_grad():
        return model.sample(num_images, (3, 32, 32), device)


def imsave(img, file):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(file)


if __name__ == "__main__":
    device = "cuda"
    noise_type = "normal"
    model_path = f"./models/cifar10/{noise_type}/"
    sample_size = 10  # Number of samples to save per model
    output_dir = f"./results/{noise_type}/images/"

    print(f"Saving samples: {noise_type} trained on CIFAR-10")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoints = [199]
    # 0, 1, 5, 10, 20, 50, 100, 150, 

    image_per_epoch = torch.zeros(len(checkpoints), 3, 32, 32)
    # Modify FID score calculation to process images in batches
    for i in checkpoints:
        samples_file = f"{output_dir}/samples_{i}.png"
        ngldm_model = load_model(f"{model_path}/ngldm_cifar10_{i}.pth", device)

        images = generate_images(ngldm_model, sample_size, device)
        image_per_epoch[checkpoints.index(i)] = images[0]

        if i == 199:
            grid = torchvision.utils.make_grid(images)
            imsave(grid, samples_file)

    row_grid = torchvision.utils.make_grid(image_per_epoch, nrow=len(checkpoints))
    imsave(row_grid, f"{output_dir}/progress.png")
