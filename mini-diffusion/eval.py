import argparse
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
from tqdm import tqdm

def calculate_fid_score(real_features, gen_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_features_from_model(images, model, device):
    model.eval()
    batch_size = 32  # You can adjust the batch size depending on your GPU
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Processing images"):
            batch = images[i:i + batch_size]
            if len(batch.shape) == 3:  # Add batch dimension if it's missing
                batch = batch.unsqueeze(0)
            batch = batch.to(device)
            features.append(model(batch)[0].view(batch.size(0), -1).cpu().numpy())
    return np.concatenate(features, axis=0)

def load_images_from_npz(npz_file):
    with np.load(npz_file) as data:
        images = data['arr_0']
    return torch.tensor(images).float()

def create_argparser():
    parser = argparse.ArgumentParser(description="Evaluate FID Score")
    parser.add_argument("--real_images_npz", type=str, required=True, help="Path to the NPZ file of real images")
    parser.add_argument("--generated_images_npz", type=str, required=True, help="Path to the NPZ file of generated images")
    return parser

if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)

    # Load images from NPZ files
    real_images = load_images_from_npz(args.real_images_npz).to(device)
    generated_images = load_images_from_npz(args.generated_images_npz).to(device)

    # Normalize the images if required (depending on how they were saved)
    # For example, if images are saved in the range [0, 1]:
    # transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # real_images = transform(real_images)
    # generated_images = transform(generated_images)

    # Calculate features
    real_features = get_features_from_model(real_images, inception_model, device)
    generated_features = get_features_from_model(generated_images, inception_model, device)

    # Calculate FID Score
    fid_score = calculate_fid_score(real_features, generated_features)
    print(f"FID Score: {fid_score}")
