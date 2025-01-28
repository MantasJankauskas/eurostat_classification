import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainedModelStatistics:
    def __init__(self, model=None):
        self.model = model
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def add_train_loss(self, val):
        self.train_losses.append(val)

    def add_val_losses(self, val):
        self.val_losses.append(val)

    def add_train_accuracies(self, val):
        self.train_accuracies.append(val)

    def add_val_accuracies(self, val):
        self.val_accuracies.append(val)

    def get_train_loss(self):
        return self.train_losses

    def get_val_losses(self):
        return self.val_losses

    def get_train_accuracies(self):
        return self.train_accuracies

    def get_val_accuracies(self):
        return self.val_accuracies

    def get_statistics(self):
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

    def plot_training_graphs(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Change Each Epoch")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label="Training loss")
        plt.plot(self.val_accuracies, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Change Each Epoch")
        plt.legend()

        image_title = self.model.return_model_name()
        plt.savefig(f"images/{image_title}_training_graphs.png")

        plt.tight_layout()
        plt.show()

    def show_confusion_matrix(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        # Ensure no gradients are calculated
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")

        image_title = self.model.return_model_name()
        plt.savefig(f"images/{image_title}_confusion_matrix.png")

        plt.show()
