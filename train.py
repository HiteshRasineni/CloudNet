# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import config
from model import get_model
from supcon_loss import SupConLoss

# ---------- Transforms ----------
# Strong augmentation for contrastive learning (first view)
train_transform_1 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Different augmentation for second view
train_transform_2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Dataset & Split ----------
# Create dataset with two different transforms for SupCon
class TwoAugmentDataset(torch.utils.data.Dataset):
    """Dataset that applies two different augmentations to each image"""
    def __init__(self, base_dataset, transform1, transform2):
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image1 = self.transform1(image)
        image2 = self.transform2(image)
        return image1, image2, label

# Load base dataset
base_dataset = datasets.ImageFolder(root=config.DATA_DIR)

val_split = 0.2
val_size = int(len(base_dataset) * val_split)
train_size = len(base_dataset) - val_size

train_data_base, val_data_base = random_split(base_dataset, [train_size, val_size])

# Create training dataset with dual augmentations
train_dataset = TwoAugmentDataset(train_data_base, train_transform_1, train_transform_2)

# Create validation dataset with proper indices from the split
val_dataset_full = datasets.ImageFolder(root=config.DATA_DIR, transform=val_transform)
val_subset = torch.utils.data.Subset(val_dataset_full, val_data_base.indices)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- Model & Optimizer ----------
device = config.DEVICE
model = get_model(embedding_dim=config.EMBEDDING_DIM, use_supcon=config.USE_SUPCON).to(device)
criterion_cls = nn.CrossEntropyLoss()

# Initialize SupCon loss if enabled
if config.USE_SUPCON:
    criterion_supcon = SupConLoss(temperature=config.SUPCON_TEMPERATURE).to(device)
else:
    criterion_supcon = None

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_acc = 0.0

# ---------- Training ----------
for epoch in range(config.EPOCHS):
    print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")

    # --- Training ---
    model.train()
    running_loss, running_cls_loss, running_supcon_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0
    loop = tqdm(train_loader, desc="Training")

    for batch_data in loop:
        if config.USE_SUPCON:
            images1, images2, labels = batch_data
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            
            # Concatenate two views
            images = torch.cat([images1, images2], dim=0)
            labels_concat = torch.cat([labels, labels], dim=0)
            
            optimizer.zero_grad()
            
            # Forward pass with embeddings
            embeddings, logits = model(images, return_features=True)
            
            # Classification loss (on original batch size)
            logits_view1 = logits[:len(labels)]
            cls_loss = criterion_cls(logits_view1, labels)
            
            # Contrastive loss
            supcon_loss = criterion_supcon(embeddings, labels_concat)
            
            # Combined loss
            loss = cls_loss + config.SUPCON_WEIGHT * supcon_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_supcon_loss += supcon_loss.item()
            
            _, predicted = torch.max(logits_view1, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(
                loss=running_loss / (len(loop) if len(loop) > 0 else 1),
                cls_loss=running_cls_loss / (len(loop) if len(loop) > 0 else 1),
                supcon_loss=running_supcon_loss / (len(loop) if len(loop) > 0 else 1),
                acc=100. * correct / total if total > 0 else 0
            )
        else:
            # Standard training without SupCon
            images, labels = batch_data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(
                loss=running_loss / (len(loop) if len(loop) > 0 else 1),
                acc=100. * correct / total if total > 0 else 0
            )

    scheduler.step()

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Only classification head for validation
            loss = criterion_cls(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
    if config.USE_SUPCON:
        print(f"  Avg Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"  Avg Classification Loss: {running_cls_loss/len(train_loader):.4f}")
        print(f"  Avg SupCon Loss: {running_supcon_loss/len(train_loader):.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), config.MODEL_PATH)
        print(f"  ✓ Saved new best model!")

print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
