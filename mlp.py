import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from feature_extraction import get_resnet_feature_extractor, extract_features, reduce_features_with_pca
from data.download import load

class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size1=512, hidden_size2=512, output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(model, train_loader, criterion, optimizer, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_mlp(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def run_mlp():
    # Load Data
    train_loader, test_loader = load(sample_per_class=500, batch_size=32)

    # Feature Extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = get_resnet_feature_extractor()

    train_features, train_labels = extract_features(resnet, train_loader, device)
    test_features, test_labels = extract_features(resnet, test_loader, device)

    # PCA
    train_features_reduced, pca_model = reduce_features_with_pca(train_features, n_components=50)
    test_features_reduced = pca_model.transform(test_features)

    # Convert to PyTorch DataLoader
    train_tensor = torch.tensor(train_features_reduced, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_tensor = torch.tensor(test_features_reduced, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define Model, Loss, and Optimizer
    model = MLP(input_size=50, hidden_size1=512, hidden_size2=512, output_size=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train Model
    print("Training MLP...")
    for epoch in range(10):  # Adjust the number of epochs as needed
        train_mlp(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/10 completed.")

    # Evaluate Model
    print("Evaluating MLP...")
    preds, labels = evaluate_mlp(model, test_loader, device)

    # Compute Metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    conf_matrix = confusion_matrix(labels, preds)

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Return results for comparison
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
