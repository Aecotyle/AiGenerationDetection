import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from tqdm import tqdm

def load_model(model_name="microsoft/beit-base-patch16-224-pt22k-ft22k", num_labels=2):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return feature_extractor, model

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def main():
    # Dataset directory structure:
    # dataset/
    #   ai_generated/
    #   not_ai_generated/
    dataset_dir = "dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor, model = load_model()
    model.to(device)

    # Define transforms compatible with feature extractor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")
    feature_extractor.save_pretrained("fine_tuned_model")
    print("Model fine-tuning complete and saved to 'fine_tuned_model' directory.")

if __name__ == "__main__":
    main()
