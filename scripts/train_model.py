import torch
import torch.nn as nn
from tqdm import tqdm
from models.CNN import CNN
from scripts.data_preprocessing import get_data_for_training

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(loss=loss.item())

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_loader_tqdm.set_postfix(loss=loss.item())

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f"   Train Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"   Val Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    from models.CNN import CNN
    from scripts.data_preprocessing import get_data_for_training

    train_loaderz, val_loaderz, test_loaderz = get_data_for_training(num_workers=6)
    train_model(CNN(), train_loaderz, val_loaderz)