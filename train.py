# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from model import get_model
import copy

# ---------- Transforms ----------
train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Data ----------
train_data = datasets.ImageFolder(root=f"{config.DATA_DIR}/train", transform=train_transform)
val_data = datasets.ImageFolder(root=f"{config.DATA_DIR}/test", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

# ---------- Model & Optimizer ----------
device = config.DEVICE
model = get_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ---------- Early Stopping Setup ----------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
patience = 5
counter = 0

for epoch in range(config.EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{config.EPOCHS}]")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc="Training")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=running_loss / len(train_loader),
                         acc=100. * correct / total)

    scheduler.step()

    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")

    # ---------- Early Stopping Check ----------
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# ---------- Save Best Model ----------
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), config.MODEL_PATH)
print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
