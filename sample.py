
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
from torch import optim, nn
from torchmetrics.classification import Accuracy
from tqdm import tqdm

# Set paths
train_dir = 'C:/Users/harsh/OneDrive/Desktop/B14 projects/sample/Skin Disease Trained Data Set/skin-disease-datasaet/train_set'
test_dir = 'C:/Users/harsh/OneDrive/Desktop/B14 projects/sample/Skin Disease Trained Data Set\skin-disease-datasaet/test_set'
model_save_path = 'trained_vit_model.pth'  # Path to save the trained model

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Check class distribution
class_counts = [len([label for _, label in train_dataset.samples if label == i]) for i in range(len(train_dataset.classes))]
print(f"Class counts: {class_counts}")

# Weighted loss to handle class imbalance
class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Custom ViT configuration
model_name = "google/vit-base-patch16-224-in21k"
config = ViTConfig.from_pretrained(model_name)
config.image_size = 224
config.num_labels = len(train_dataset.classes)

# Initialize the model
model = ViTForImageClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

# Define optimizer and metrics
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
accuracy_metric = Accuracy(task="multiclass", num_classes=len(train_dataset.classes))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
def train_epoch(model, train_loader, optimizer, criterion, accuracy_metric):
    model.train()
    running_loss = 0.0
    accuracy_metric.reset()

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        accuracy_metric.update(outputs, labels)

        # Calculate and print batch accuracy
        if (batch_idx + 1) % 1 == 0:  # This condition always holds true for every batch
            batch_accuracy = accuracy_metric.compute().item()
            print(f"\nBatch {batch_idx + 1}/{len(train_loader)} - Accuracy: {batch_accuracy:.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = accuracy_metric.compute().item()
    accuracy_metric.reset()

    return epoch_loss, epoch_accuracy


# Evaluation loop
def evaluate(model, test_loader, accuracy_metric):
    model.eval()
    accuracy_metric.reset()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            accuracy_metric.update(outputs, labels)

    accuracy = accuracy_metric.compute().item()
    accuracy_metric.reset()
    return accuracy

# Training the model
epochs = 30
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, accuracy_metric)
    print(f"Training loss: {train_loss:.6f}, Training accuracy: {train_accuracy:.6f}")

    test_accuracy = evaluate(model, test_loader, accuracy_metric)
    print(f"Test accuracy: {test_accuracy:.6f}")

# Save the model using joblib
# dump(model.state_dict(), 'trained_vit_model.joblib')
torch.save(model, 'trained_vit_model.pth')

print("Model saved as trained_vit_model.pth")
