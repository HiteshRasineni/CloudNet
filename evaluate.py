import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import config
from model import get_model
from tqdm import tqdm

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- Load Dataset ----------
dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=transform)
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# ---------- Load Model ----------
device = config.DEVICE
model = get_model().to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()

# ---------- Evaluation ----------
all_preds = []
all_labels = []

print("Evaluating model...")
with torch.no_grad():
    for images, labels in tqdm(data_loader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------- Metrics ----------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES))

acc = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy: {acc*100:.2f}%")

# ---------- Confusion Matrix ----------
print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)

# ---------- Plot ----------
print("Plotting confusion matrix...")
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(config.CLASS_NAMES))
plt.xticks(tick_marks, config.CLASS_NAMES, rotation=45)
plt.yticks(tick_marks, config.CLASS_NAMES)

# Add annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
