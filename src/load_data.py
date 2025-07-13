import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_root = "data/"
processed_dir = os.path.join(data_root, "processed")
batch_size = 64

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def download_and_cache():
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "train_dataset.pt")
    test_path = os.path.join(processed_dir, "test_dataset.pt")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading cached datasets...")
        train_dataset = torch.load(train_path, weights_only=False)
        test_dataset = torch.load(test_path, weights_only=False)
    else:
        print("Downloading and processing CIFAR-10...")
        train_dataset = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=transform, download=True)

        torch.save(train_dataset, train_path)
        torch.save(test_dataset, test_path)

    return train_dataset, test_dataset

def get_dataloaders(batch_size=batch_size):
    train_dataset, test_dataset = download_and_cache()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(test_loader.dataset)} test samples.")
