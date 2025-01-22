import matplotlib.pyplot as plt

class TrainedModelStatistics:
    def __init__(self):
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

        plt.tight_layout()
        plt.show()