import torch
import torch.nn as nn
from tqdm import tqdm

from utils.TrainedModelStatistics import TrainedModelStatistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, model_statistics: TrainedModelStatistics, train_loader, val_loader, epochs=10, show_info=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    for epoch in range(epochs):
        if show_info:
            print(f"\nEpoch {epoch + 1}/{epochs}")

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

        model_statistics.add_train_loss(train_loss / len(train_loader))
        model_statistics.add_train_accuracies(100 * correct / total)

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

        model_statistics.add_val_losses(val_loss / len(val_loader))
        model_statistics.add_val_accuracies(100 * correct / total)

        if show_info:
            print(f"\nTrain Loss: {model_statistics.get_train_loss()[-1]:.4f}, Accuracy: {model_statistics.get_train_accuracies()[-1]:.2f}%")
            print(f"Val Loss: {model_statistics.get_val_losses()[-1]:.4f}, Accuracy: {model_statistics.get_val_accuracies()[-1]:.2f}%")

    return model


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Testing accuracy: {100 * correct / total:.2f}%")