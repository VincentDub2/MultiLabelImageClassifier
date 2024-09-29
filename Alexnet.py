
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetMultiLabelClassifier(nn.Module):
    def __init__(self):
        super(AlexNetMultiLabelClassifier, self).__init__()

        # Première couche convolutionnelle
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)  # Taille: 224x224
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # Taille: 55x55 après pooling

        # Deuxième couche convolutionnelle
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)  # Taille: 27x27
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # Taille: 27x27 après pooling

        # Troisième couche convolutionnelle
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)  # Taille: 13x13

        # Quatrième couche convolutionnelle
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)  # Taille: 13x13

        # Cinquième couche convolutionnelle
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Taille: 13x13
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # Taille: 6x6 après pooling

        # Couches fully connected
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 17)  # 17 labels pour la classification multi-label

    def forward(self, x):
        # Appliquer les couches convolutionnelles + pooling
        x = self.pool(F.relu(self.conv1(x)))  # Taille: 55x55
        x = self.pool2(F.relu(self.conv2(x)))  # Taille: 27x27
        x = F.relu(self.conv3(x))  # Taille: 13x13
        x = F.relu(self.conv4(x))  # Taille: 13x13
        x = self.pool3(F.relu(self.conv5(x)))  # Taille: 6x6

        # Aplatir les caractéristiques
        x = x.view(-1, 256 * 6 * 6)  # Flatten

        # Appliquer les couches fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Utiliser la sigmoïde pour la classification multi-label
        x = torch.sigmoid(self.fc3(x))  # Pour chaque label, on sort une probabilité (entre 0 et 1)
        return x
