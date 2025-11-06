import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from model import get_model
import config

# Path to dataset
DATA_DIR = "data2"

# 1. Count images in each folder
class_counts = {}
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len([
            f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

# Plot class distribution
plt.figure(figsize=(10,6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel("Cloud Types")
plt.ylabel("Number of Images")
plt.title("Number of Images in Each Cloud Type")
plt.xticks(rotation=45)
plt.show()

# 2. Prepare data for t-SNE
def load_images(data_dir, img_size=(64,64), max_images=100):
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        for img_file in tqdm(images[:max_images], desc=f"Loading {class_name}"):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).convert("RGB").resize(img_size)
                    X.append(np.array(img).flatten())
                    y.append(idx)
                except:
                    continue
    return np.array(X), np.array(y), class_names

# Load subset of data (to keep t-SNE manageable)
X, y, class_names = load_images(DATA_DIR, img_size=(64,64), max_images=200)

# Normalize
X = StandardScaler().fit_transform(X)

# Dimensionality reduction (PCA before t-SNE for speed)
X_pca = PCA(n_components=50).fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_pca)

# Plot t-SNE
plt.figure(figsize=(10,7))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="tab10", alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Cloud Types", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE Visualization of Cloud Types (Raw Features)")
plt.show()

# 3. Extract features from trained model
def extract_features_from_model(model, data_dir, img_size=(224, 224), max_images=200):
    """Extract features from the trained SupConResNet model (encoder output)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Define transforms for the model
    transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    
    print(f"Extracting features from {len(class_names)} classes...")
    
    with torch.no_grad():
        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"Skipping {class_name}: not a directory")
                continue
            
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {class_name}: {len(images)} images available, processing {min(max_images, len(images))}")
            
            success_count = 0
            for img_file in tqdm(images[:max_images], desc=f"Extracting features from {class_name}"):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    # Use the model's encoder to extract features
                    # The encoder gives us the features before the classification head
                    feat = model.encoder(img_tensor).squeeze()
                    
                    # Ensure we have a 1D feature vector
                    if feat.dim() > 1:
                        feat = feat.flatten()
                    
                    features.append(feat.cpu().numpy())
                    labels.append(idx)
                    success_count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            print(f"Successfully processed {success_count} images from {class_name}")
    
    print(f"Total features extracted: {len(features)}")
    return np.array(features), np.array(labels), class_names

# Load the trained model
print("Loading trained model...")
model = get_model()
if os.path.exists(config.MODEL_PATH):
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location='cpu', weights_only=True))
    print("Model loaded successfully!")
else:
    print("Warning: Trained model not found. Using untrained model for feature extraction.")

# Extract features from the trained model
print("Extracting features from trained model...")
X_features, y_features, class_names_features = extract_features_from_model(model, DATA_DIR, img_size=(224, 224), max_images=200)

# Check if features were successfully extracted
if len(X_features) == 0:
    print("Warning: No features were extracted. This could be because:")
    print("1. The model file doesn't exist at the specified path")
    print("2. No images were successfully processed")
    print("3. There were errors during feature extraction")
    print("Skipping feature-based visualization.")
else:
    # Normalize the extracted features
    X_features = StandardScaler().fit_transform(X_features)

# Only proceed with feature-based visualization if features were extracted
if len(X_features) > 0:
    # Dimensionality reduction (PCA before t-SNE for speed)
    X_features_pca = PCA(n_components=50).fit_transform(X_features)

    # Apply t-SNE to the extracted features
    tsne_features = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne_features = tsne_features.fit_transform(X_features_pca)

    # Plot t-SNE for extracted features
    plt.figure(figsize=(10,7))
    scatter_features = plt.scatter(X_tsne_features[:,0], X_tsne_features[:,1], c=y_features, cmap="tab10", alpha=0.7)
    plt.legend(handles=scatter_features.legend_elements()[0], labels=class_names_features, title="Cloud Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE Visualization of Cloud Types (After Training Features)")
    plt.show()
else:
    print("Feature-based visualization skipped due to empty feature array.")
