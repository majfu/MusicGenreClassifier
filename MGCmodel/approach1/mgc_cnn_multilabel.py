import torch.nn as nn


class MultilabelMusicGenreCNN(nn.Module):
    def __init__(self, num_classes, hidden_size=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.average_pooling(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
