from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import numpy as np
import torch

from main import model, val_dataloader, device, test_dataloader, criterion

# Liste des classes de votre dataset (dans l'ordre de vos colonnes dans LandUse_Multilabeled.txt)
classes = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
           "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks",
           "trees", "water"]

# Charger le meilleur modèle

model.load_state_dict(torch.load('/best_model.pt', weights_only=True))
# Mode évaluation pour le modèle
model.eval()

# Variables pour suivre la perte et les prédictions correctes
all_outputs = []
all_labels = []

# Boucle d'évaluation sur l'ensemble de validation (pour ajuster les seuils)
with torch.no_grad():  # Désactiver les gradients pour l'évaluation
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs)

        # Stocker les prédictions et les labels réels pour ajuster les seuils
        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Convertir les listes en NumPy arrays pour ajuster les seuils
all_outputs = np.concatenate(all_outputs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Ajuster les seuils pour chaque classe
best_thresholds = []
for i in range(all_labels.shape[1]):  # Pour chaque classe
    best_f1 = 0
    best_threshold = 0.5
    for t in np.arange(0.0, 1.0, 0.05):  # Tester les seuils de 0 à 1 par pas de 0.05
        preds_binary_class = (all_outputs[:, i] > t).astype(float)
        f1 = f1_score(all_labels[:, i], preds_binary_class)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    best_thresholds.append(best_threshold)

# Utiliser les meilleurs seuils trouvés pour chaque classe
best_thresholds = torch.tensor(best_thresholds).to(device)
print(f'Best thresholds for each class: {best_thresholds}')

# Phase d'inférence (par exemple sur le jeu de test)
model.eval()
test_loss = 0.0
all_preds = []
all_labels = []

# Boucle d'inférence sur le jeu de test
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs)

        # Calcul de la perte
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Appliquer les meilleurs seuils trouvés pour chaque classe
        preds_binary = (outputs > best_thresholds).float()
        #preds_binary =  (outputs > 0.5).float()

        # Stocker les prédictions et les labels
        all_preds.append(preds_binary.cpu())
        all_labels.append(labels.cpu())

# Convertir les prédictions et labels en un seul tensor pour le calcul des métriques
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Calcul de la précision pour chaque image (Exact Match Accuracy)
accuracy = (all_preds == all_labels).float().mean().item() * 100  # En pourcentage
print(f"Précision sur le test set: {accuracy:.2f}%")
# Calcul de la Hamming Loss
hamming_loss = (all_preds != all_labels).float().mean().item() * 100
print(f"Hamming Loss sur le test set: {hamming_loss:.2f}%")

# Calcul du F1 Score (Micro et Macro)
all_labels_np = all_labels.numpy()
preds_binary_np = all_preds.numpy()


# Calcul de la précision pour chaque classe
class_precisions = precision_score(all_labels_np, preds_binary_np, average=None) * 100

# Calcul du rappel pour chaque classe (optionnel)
class_recalls = recall_score(all_labels_np, preds_binary_np, average=None) * 100

# Afficher la précision pour chaque classe
for i, precision in enumerate(class_precisions):
    print(f"Précision pour la classe {classes[i]}: {precision:.2f}%")

# Afficher le rappel pour chaque classe (optionnel)
for i, recall in enumerate(class_recalls):
    print(f"Rappel pour la classe {classes[i]}: {recall:.2f}%")


f1_micro = f1_score(all_labels_np, preds_binary_np, average='micro') * 100
f1_macro = f1_score(all_labels_np, preds_binary_np, average='macro') * 100

print(f"F1 Score (Micro) sur le test set: {f1_micro:.2f}%")
print(f"F1 Score (Macro) sur le test set: {f1_macro:.2f}%")
