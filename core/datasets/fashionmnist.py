import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from core.utils.registry import registry

@registry.register_dataset("fashion")
class FashionMNISTDataset(Dataset):
    def __init__(self, train=True, root='./data', ratio=1.):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=True, transform=transform)
        self.ratio = ratio

    def __len__(self):
        return int(len(self.dataset)*self.ratio)

    def __getitem__(self, idx):
        return self.dataset[idx]
