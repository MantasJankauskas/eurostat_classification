import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 kanalai (RGB), 16 filtrų
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 filtrai
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 64 filtrai
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 baseinavimas
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected sluoksnis
        self.fc2 = nn.Linear(128, 10)  # Klasės skaičius

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pool
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected sluoksnis
        x = self.fc2(x)  # Klasifikavimo sluoksnis
        return x