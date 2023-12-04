import os
import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import transforms, datasets
from torchvision.models import inception_v3
from torchvision.utils import save_image
from unet import Unet
from ngldm import NGLDM


def load_model(model_path, device="cuda") -> NGLDM:
    ngldm = NGLDM(eps_model=Unet(3, 3, n_feat=128), betas=(1e-4, 0.02), T=1000)
    ngldm.load_state_dict(torch.load(model_path, map_location=device))
    ngldm.to(device)
    return ngldm


def generate_images(model: NGLDM, num_images, device="cuda") -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        generated_images = model.sample(num_images, (3, 32, 32), device)
    return generated_images


def save_generated_images(images: torch.Tensor, directory, prefix="gen_image") -> None:
    for i, image in enumerate(images):
        save_image(image, f"{directory}/{prefix}_{i}.png", normalize=True, value_range=(-1, 1))


def calculate_fid(model, images1, images2):
    def calculate_activation_statistics(images, model):
        model.eval()
        with torch.no_grad():
            # Here, only the first output of the model is used
            activations = model(images)[0]

            # Resize activations to 1x1 if necessary
            if activations.shape[2] != 1 or activations.shape[3] != 1:
                activations = torch.nn.functional.adaptive_avg_pool2d(activations, output_size=(1, 1))
            
            activations = activations.view(activations.size(0), -1)
        mu = torch.mean(activations, dim=0).cpu().numpy()
        sigma = torch.cov(activations.t()).cpu().numpy()
        return mu, sigma

    mu1, sigma1 = calculate_activation_statistics(images1, model)
    mu2, sigma2 = calculate_activation_statistics(images2, model)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    device = "cuda"
    model_path = "./models/cifar10/laplace/ngldm_cifar10.pth"
    num_images = 10  # Number of images to generate
    output_dir = "./generated_images/test1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the trained model
    ngldm_model = load_model(model_path, device)

    # Generate images
    images = generate_images(ngldm_model, num_images, device)

    # Save the generated images
    save_generated_images(images, output_dir)

    # Load real images for FID calculation
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    real_images = torch.stack([cifar10_dataset[i][0] for i in range(num_images)]).to(device)

    # Adjust transform for Inception model
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images to 299x299
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load real images for FID calculation
    cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    real_images = torch.stack([cifar10_dataset[i][0] for i in range(num_images)]).to(device)

    # Resize generated images for FID calculation
    resized_images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

    # Calculate FID score
    fid_score = calculate_fid(inception_model, real_images, resized_images)
    print(f"FID Score: {fid_score}")