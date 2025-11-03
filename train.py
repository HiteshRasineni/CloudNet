# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import config
from model import get_model
from loss_supcon import SupConLoss


# ---------- Dataset wrapper for 2 augmented views ----------
class ContrastiveDataset(torch.utils.data.Dataset):
    """Returns two differently augmented views of the same image"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2, label

    def __len__(self):
        return len(self.dataset)


# ---------- Augmentations ----------
supcon_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ce_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------- Dataset & Dataloaders ----------
base_dataset = datasets.ImageFolder(root=config.DATA_DIR)
contrastive_dataset = ContrastiveDataset(base_dataset, supcon_transform)

val_split = 0.2
val_size = int(len(contrastive_dataset) * val_split)
train_size = len(contrastive_dataset) - val_size
train_data, val_data = random_split(contrastive_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)


# ---------- Model ----------
device = config.DEVICE
model = get_model().to(device)


# ---------- STAGE 1: SupCon Training ----------
contrast_loss = SupConLoss(temperature=0.07).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

print("\n===== Stage 1: Training with SupCon Loss =====")

for epoch in range(config.SUPCON_EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"SupCon Epoch [{epoch+1}/{config.SUPCON_EPOCHS}]")
    avg_loss = 0

    for img1, img2, labels in loop:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        feats1 = model(img1, proj=True)
        feats2 = model(img2, proj=True)

        feats = torch.cat([feats1.unsqueeze(1), feats2.unsqueeze(1)], dim=1)  # [bs, 2, feat_dim]
        loss = contrast_loss(feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        loop.set_postfix(loss=avg_loss / len(train_loader))

torch.save(model.state_dict(), config.SUPCON_MODEL_PATH)
print("\n✅ Stage 1 complete. Encoder weights saved.")


# ---------- STAGE 2: Fine-tune Classifier ----------
print("\n===== Stage 2: Fine-tuning with CrossEntropy =====")

# Reload SupCon encoder
model.load_state_dict(torch.load(config.SUPCON_MODEL_PATH))

# Freeze encoder, train only classifier
for param in model.encoder.parameters():
    param.requires_grad = False

# CE dataset (normal transform)
ce_dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=ce_transform)
train_data, val_data = random_split(ce_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.FINETUNE_LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_acc = 0

for epoch in range(config.FINETUNE_EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"CE Epoch [{epoch+1}/{config.FINETUNE_EPOCHS}]")
    correct, total, loss_sum = 0, 0, 0

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, proj=False)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        loop.set_postfix(loss=loss_sum/len(train_loader), acc=100.*correct/total)

    scheduler.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, proj=False)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), config.MODEL_PATH)

print(f"\n✅ Training complete. Best validation accuracy: {best_acc:.2f}%")
