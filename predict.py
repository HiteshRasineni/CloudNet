import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import config
from model import get_model

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Predict Single Image ===
def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1).cpu().numpy().flatten()
        pred_idx = int(probs.argmax())
    return pred_idx, probs

# === Predict Folder ===
def predict_folder(folder_path, model, device, threshold=0.3):
    y_pred, y_true, filenames = [], [], []

    # === Load ground-truth labels if available ===
    label_map = {}
    label_file = "labels.txt"
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                file, label = line.strip().split()
                label_map[file] = config.CLASS_NAMES.index(label)

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(folder_path, file_name)
        pred_class, probs = predict_image(image_path, model, device)

        # Reject low-confidence predictions
        if probs[pred_class] < threshold:
            print(f"\nImage: {file_name} → Rejected (Low confidence: {probs[pred_class]:.4f})")
            continue

        y_pred.append(pred_class)
        filenames.append(file_name)

        print(f"\nImage: {file_name} → Predicted: {config.CLASS_NAMES[pred_class]}")
        for i, cls in enumerate(config.CLASS_NAMES):
            print(f"  {cls}: {probs[i]:.4f}")

        # Append true label if available
        if file_name in label_map:
            y_true.append(label_map[file_name])

    # === Evaluation ===
    if y_true:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))
    else:
        print("\n[!] No ground-truth labels found in 'labels.txt'. Skipping evaluation.")

# === Run Prediction ===
predict_folder("testing", model, device, threshold=0.3)
