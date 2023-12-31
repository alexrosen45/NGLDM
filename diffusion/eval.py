import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
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


# Resize images to Inception v3 input size
def resize_images(images):
    transform = transforms.Compose([
        transforms.Resize((299, 299), antialias=True)
    ])
    return torch.stack([transform(image) for image in images])


def get_inception_features(model, images, device, batch_size):
    model.eval()
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        with torch.no_grad():
            batch_features = model(batch).detach().cpu()
        features.append(batch_features)
    return torch.cat(features, dim=0)


# Calculate FID score with numerical stability
def calculate_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    device = "cuda"
    num_models = 2
    noise_type = "uniform"
    model_path = f"./models/cifar10/{noise_type}/"
    batches = 30
    sample_size = 100  # Number of samples per batch (to solve VRAM out of memory issues)
    num_images = batches * sample_size  # Total number of samples
    output_dir = "./eval"

    print(f"Evaluating {num_models} models: {noise_type} trained on CIFAR-10")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load pre-trained Inception v3 model
    inception_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inception_model.fc = torch.nn.Flatten()
    inception_model.to(device)

    # Load CIFAR10 dataset and resize real images
    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
    real_images = next(iter(dataloader))[0]
    resized_real_images = resize_images(real_images)
    real_features = get_inception_features(inception_model, resized_real_images, device, sample_size)

    # Modify FID score calculation to process images in batches
    fid_scores_file = f"{output_dir}/{noise_type}_fid_scores.txt"
    with open(fid_scores_file, "w") as file:
        for i in range(num_models):
            ngldm_model = load_model(f"{model_path}/ngldm_cifar10_{i}.pth", device)

            all_generated_features = []
            for _ in range(batches):
                images = generate_images(ngldm_model, sample_size, device)
                resized_generated_images = resize_images(images)

                # Get feature vectors for generated images in batches
                generated_features = get_inception_features(inception_model, resized_generated_images, device, sample_size)
                all_generated_features.append(generated_features.numpy())

            all_generated_features = np.concatenate(all_generated_features, axis=0)
            fid_score = calculate_fid(real_features.numpy(), all_generated_features)
            print(f"FID score for model {i}: {fid_score}")
            file.write(f"Model {i}: FID score = {fid_score}\n")