import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np


from EarlyStopping import EarlyStopping
from MultiLabelLandUseDataset import MultiLabelLandUseDataset
from ResNet18 import ResNet18

# Initialisation et configuration de base
image_dir = 'ucmdata'  # Ajuste ce chemin à celui de tes images
label_file = 'ucmdata/LandUse_Multilabeled.txt'  # Fichier de labels

# Transformations des images
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Redimensionner les images à 256x256
    transforms.ToTensor(),  # Convertir les images en tenseurs
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
])

# Création du dataset
dataset = MultiLabelLandUseDataset(image_dir=image_dir, label_file=label_file, transform=transform)


# Extraire les labels du DataFrame
labels = dataset.labels_df.iloc[:, 1:].values  # Labels sans la première colonne (nom de l'image)

# Calculer les occurrences de chaque classe (somme des 1 pour chaque colonne)
class_counts = np.sum(labels, axis=0)

# Créer un dictionnaire avec le nom des classes et leurs occurrences
class_names = dataset.labels_df.columns[1:]  # Obtenir les noms des classes
class_counts_dict = dict(zip(class_names, class_counts))

# Afficher les résultats
for cls, count in class_counts_dict.items():
    print(f"Classe {cls} : {int(count)} occurrences")

dataset_size = len(dataset)
train_size = int(0.75 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# Séparation en ensembles d'entraînement, de validation et de test
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modèle
#model = AlexNetMultiLabelClassifier()
model = ResNet18(17)
#model = SimpleCNN(17)
#model = LeNet(17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fonction de perte et optimiseur

# Nombre total d'images dans le dataset
total_samples = len(dataset)

# Calculer les poids pour chaque classe (inverse de la fréquence des occurrences)
class_weights = {cls: total_samples / count for cls, count in class_counts_dict.items()}

# Convertir les poids en une liste ordonnée pour les passer à BCEWithLogitsLoss
weights_list = [class_weights[cls] for cls in class_names]

# Convertir en tensor
weights_tensor = torch.tensor(weights_list).to(device)

# Utiliser ces poids dans la fonction de perte
#criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
criterion = nn.BCELoss()
#criterion = nn.BCELoss(weight=weights_tensor)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Initialisation de l'early stopping
early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.005,path='best_model.pt')

# Entraînement
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #print(f"Batch {i+1}, Training Loss: {loss.item()}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Training Loss: {running_loss / len(train_dataloader):.4f}')

    # Scheduler
    scheduler.step(val_loss)
    current_lr = scheduler.get_last_lr()[0]
    print(f'Current learning rate: {current_lr}')

    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Charger le meilleur modèle
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
