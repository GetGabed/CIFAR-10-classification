import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def subset(dataset, sample_per_class):
    # Create a subset of the dataset
    classes = {i: 0 for i in range(10)}
    indices = []
    print('Classes: ', classes)

    for i, (_, label) in enumerate(dataset):
        if classes[label] < sample_per_class:
            indices.append(i) 
            classes[label] += 1 
        if sum(classes.values()) == 10 * sample_per_class: 
            break

    return Subset(dataset, indices)

def transformsResize():
    # Define transformations: resize to 224x224 and normalize
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resizes images to 224x224 for ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalization values for CIFAR-10
    ])


def load(sample_per_class=500, batch_size=32):
    # Load the dataset
    transforms = transformsResize()

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)

    # Create a subset of the dataset
    train_dataset = subset(train_dataset, sample_per_class)
    test_dataset = subset(test_dataset, 100)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader