import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA


# Load Pre-trained ResNet-18
def get_resnet_feature_extractor():
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()
    return resnet.eval()

# Preprocessing Function
def preprocess_cifar10_images():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

# Extract Features Using ResNet-18
def extract_features(model, dataloader, device):
    features, labels = [], []
    model.to(device)
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)

# Apply PCA
def reduce_features_with_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca

# Main Process
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load preprocessed CIFAR-10 dataset
    from data.download import load
    train_loader, test_loader = load(sample_per_class=500, batch_size=32)

    # Load ResNet-18 model
    resnet = get_resnet_feature_extractor()

    # Extract features for train and test sets
    print("Extracting training features...")
    train_features, train_labels = extract_features(resnet, train_loader, device)
    print("Extracting test features...")
    test_features, test_labels = extract_features(resnet, test_loader, device)

    # Apply PCA
    print("Applying PCA on training features...")
    train_features_reduced, pca_model = reduce_features_with_pca(train_features, n_components=50)
    print("Transforming test features using PCA...")
    test_features_reduced = pca_model.transform(test_features)

    # Save or return reduced features for Naive Bayes
    print("Feature extraction and dimensionality reduction complete.")
