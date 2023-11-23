import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import data
from tqdm import tqdm
import os
import argparse


# Variance scheduling
class Schedule:
    def __init__(self, timesteps, type, start, increment):
        self.timesteps = timesteps
        # linear or quadratic schedule
        self.type = type
        self.start = start
        self.increment = increment

    def sample_variances(self):
        t = np.random.randint(1, self.timesteps)
        # list of variances by t=index
        variance = []
        for i in range(t):
            i_v = i if self.type == "linear" else i ** 2
            variance.append(self.start + self.increment * i_v)
        return t, variance


# Define a PyTorch Dataset
class NoisyDataset(Dataset):
    def __init__(self, data, labels, noise_fn, schedule):
        self.data = data
        self.labels = labels
        self.noise_fn = noise_fn
        self.schedule = schedule

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, ):
        t, variance = self.schedule.sample_variances()
        noisy_data = apply_noise(self.noise_fn, self.data[idx], variance)
        return noisy_data, self.data[idx], t


def apply_noise(noise_fn, data, schedule):
    noisy_data = data.copy()
    # loop over variance steps
    for i in range(len(schedule)):
        variance = schedule[i]
        # add noise to data
        noisy_data += noise_fn(noisy_data.shape, variance)
    return noisy_data


def normal(shape, var):
    return np.random.normal(scale=math.sqrt(var), size=shape)


def logistic(shape, var):
    s = math.sqrt((3 * var) / (math.pi ** 2))
    return np.random.logistic(loc=0, scale=s, size=shape)


def gumbel(shape, var):
    s = math.sqrt((6 * var) / (math.pi ** 2))
    return np.random.gumbel(loc=0, scale=s, size=shape) - s * 0.57721


def exponential(shape, var):
    s = math.sqrt(var)
    return np.random.exponential(scale=s, size=shape) - s


ALL_NOISE = {"normal": normal, "gumbel": gumbel, "logistic": logistic, "exponential": exponential}
# Deprecated
# Apply Markov chain noise to the data
def apply_markov_noise(data, steps, step_size, noise_type):
    noisy_data = data.copy()
    for i in range(steps):
        if noise_type == "normal":
            noisy_data += np.random.normal(scale=step_size, size=noisy_data.shape)
        elif noise_type == "laplace":
            noisy_data += np.random.laplace(loc=0, scale=step_size, size=noisy_data.shape)
        elif noise_type == "uniform":
            noisy_data += np.random.uniform(low=-step_size, high=step_size, size=noisy_data.shape)
        elif noise_type == "poisson":
            noisy_data += np.random.poisson(lam=step_size, size=noisy_data.shape)
        elif noise_type == "binomial":
            noisy_data += np.random.binomial(n=1, p=0.5, size=noisy_data.shape) * step_size
        elif noise_type == "exponential":
            noisy_data += np.random.exponential(scale=step_size, size=noisy_data.shape)
        elif noise_type == "gamma":
            noisy_data += np.random.gamma(shape=2.0, scale=step_size, size=noisy_data.shape)
        elif noise_type == "beta":
            noisy_data += np.random.beta(a=0.5, b=0.5, size=noisy_data.shape) * step_size
        elif noise_type == "chisquare":
            noisy_data += np.random.chisquare(df=2, size=noisy_data.shape)
        elif noise_type == "rayleigh":
            noisy_data += np.random.rayleigh(scale=step_size, size=noisy_data.shape)
        elif noise_type == "logistic":
            noisy_data += np.random.logistic(loc=0, scale=step_size, size=noisy_data.shape)
        elif noise_type == "gumbel":
            noisy_data += np.random.gumbel(loc=0, scale=step_size, size=noisy_data.shape)
        else:
            raise ValueError("Invalid noise argument; unsupported noise type.")
    return noisy_data


# Define the neural network
class DenoisingNet(nn.Module):
    def __init__(self):
        super(DenoisingNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.fc3(x)


# Denoise the test data
def denoise_data(loader, model, device):
    model.eval()
    denoised_data = []
    with torch.no_grad():
        for noisy_data, _ in tqdm(loader, desc="Denoising Test Data"):
            noisy_data = noisy_data.to(device)
            outputs = model(noisy_data.float())
            denoised_data.append(outputs.cpu().numpy())
    return np.concatenate(denoised_data, axis=0)


def select_one_image_per_label(data, labels):
    unique_images = []
    unique_labels = []
    for label in range(10):  # Loop through labels 0 to 9
        index = np.where(labels == label)[0][0]  # Find the first occurrence of each label
        unique_images.append(data[index])
        unique_labels.append(labels[index])
    return np.array(unique_images), np.array(unique_labels)


def plot_images(images, labels, noise_type, num_images=10, title="", save_directory="results/mini_mnist/"):
    # Create the directory if it does not exist
    save_directory += noise_type
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))  # Adjusted for 10 images
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'Label: {int(labels[i])}')
        ax.axis('off')
    plt.suptitle(title)

    # Save the figure
    plt.savefig(os.path.join(save_directory, title + ".png"), dpi=300, format='png')
    plt.close(fig)


def train_model(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Instantiate model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train model
    for epoch in range(epochs):
        #TODO: Add timesteps to model inputs
        for noisy_data, clean_data, timesteps in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_data.float())
            loss = criterion(outputs, clean_data.float())
            loss.backward()
            optimizer.step()

    return model


if __name__ == '__main__':
    # Select type of noise and number of epochs
    parser = argparse.ArgumentParser(description="Denoising with PyTorch")
    parser.add_argument("--noise_type", type=str, default="normal", help="Type of noise to apply (default: normal)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments and use them
    args = parser.parse_args()
    noise_type = args.noise_type
    epochs = args.epochs

    # Load the dataset
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    if noise_type != "all" and noise_type not in ALL_NOISE.keys():
        raise ValueError("Invalid noise type.")

    noise_types = ALL_NOISE if args.noise_type == "all" else {args.noise_type: ALL_NOISE[args.noise_type]}
    for noise_type in noise_types.keys():
        # Create datasets and dataloaders
        schedule = Schedule(timesteps=25, type="linear", start=0.1, increment=0.01)
        train_dataset = NoisyDataset(train_data, train_labels, ALL_NOISE[noise_type], schedule)
        test_dataset = NoisyDataset(test_data, test_labels, ALL_NOISE[noise_type], schedule)
        model = train_model(train_dataset)

        # save generated images to npz file
        denoised_test = denoise_data(DataLoader(test_dataset, batch_size=32, shuffle=True), model, device)
        out_dir = f"results/mini_mnist/{noise_type}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.savez(out_dir + "/test.npz", denoised_test)

        # Select one image per label from test data
        unique_test_images, unique_test_labels = select_one_image_per_label(test_data, test_labels)

        # Generate noisy versions
        test_schedule = schedule.sample_variances()
        unique_noisy_images = apply_noise(ALL_NOISE[noise_type], unique_test_images, test_schedule)

        # Generate denoised versions
        unique_noisy_images_tensor = torch.tensor(unique_noisy_images, dtype=torch.float32).to(device)
        with torch.no_grad():
            unique_denoised_images_tensor = model(unique_noisy_images_tensor)
        # Move tensor back to CPU if needed for further processing
        unique_denoised_images = unique_denoised_images_tensor.cpu().numpy()

        # Visualize original, noisy, and denoised images
        plot_images(unique_test_images, unique_test_labels, noise_type, num_images=10, title="Original Images")
        plot_images(unique_noisy_images, unique_test_labels, noise_type, num_images=10, title="Noisy Images")
        plot_images(unique_denoised_images, unique_test_labels, noise_type, num_images=10, title="Denoised Images")