import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms, models

# ======================
# 配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

DATA_ROOT = "/kaggle/input/d/cliveirving/neu-image-emotion-classification/fer_data/fer_data"
train_dir = os.path.join(DATA_ROOT, "train")
test_dir  = os.path.join(DATA_ROOT, "test")

class_names = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
num_classes = len(class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}

img_size = 224
batch_size = 64          # ResNet50 比较大，64 更稳；显存不够可改32
epochs = 20              # 每折20轮，3折总算力可控
n_splits = 3
base_lr = 3e-4

# ======================
# Dataset
# ======================
class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            new_idx = np.random.randint(0, len(self.file_paths))
            img_path = self.file_paths[new_idx]
            img = Image.open(img_path).convert("L")

        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img, os.path.basename(img_path)

# ======================
# 数据增强
# ======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize(int(img_size * 1.1)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================
# 读取所有训练文件 & 标签
# ======================
train_files = []
train_labels = []

for cls in class_names:
    cls_dir = os.path.join(train_dir, cls)
    label = class_to_idx[cls]
    for f in os.listdir(cls_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            train_files.append(os.path.join(cls_dir, f))
            train_labels.append(label)

train_files = np.array(train_files)
train_labels = np.array(train_labels)
print("Total train images:", len(train_files))

# 类频率 & 权重
class_counts = np.bincount(train_labels, minlength=num_classes)
print("Class counts:", dict(zip(class_names, class_counts)))

class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# ======================
# ResNet50 模型
# ======================
class ResNet50FER(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes):
    return ResNet50FER(num_classes)

# ======================
# K-fold 训练 + TTA 推理
# ======================
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_files = sorted([
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])
print("Total test images:", len(test_files))

test_preds_folds = []
fold_best_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_files, train_labels)):
    print(f"\n========== Fold {fold+1}/{n_splits} ==========")

    train_ds = EmotionDataset(train_files[train_idx],
                              train_labels[train_idx],
                              transform=train_transform)
    val_ds   = EmotionDataset(train_files[val_idx],
                              train_labels[val_idx],
                              transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=1)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=1)

    model = get_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor,
                                    label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs} - Train",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            total += bs
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Val ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs} - Val",
                                     leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Fold {fold+1} Epoch {epoch+1}/{epochs} "
              f"- Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_resnet50_fold{fold}.pth")

    print(f"Fold {fold+1} best Val Acc = {best_val_acc:.4f}")
    fold_best_accs.append(best_val_acc)

    # ======================
    # 用当前fold最佳模型对测试集预测（带TTA）
    # ======================
    model.load_state_dict(torch.load(f"best_resnet50_fold{fold}.pth", map_location=device))
    model.eval()

    test_dataset = EmotionDataset(test_files, labels=None, transform=val_test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=1)

    fold_probs = []

    with torch.no_grad():
        for imgs, ids in tqdm(test_loader,
                              desc=f"Predicting test with fold {fold+1} (TTA)",
                              leave=False):
            imgs = imgs.to(device)

            # 原图预测
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)

            # 水平翻转 TTA
            imgs_flip = torch.flip(imgs, dims=[3])
            logits_flip = model(imgs_flip)
            probs_flip = torch.softmax(logits_flip, dim=1)

            probs_mean = (probs + probs_flip) / 2.0  # (B, num_classes)
            fold_probs.append(probs_mean.cpu().numpy())

    fold_probs = np.concatenate(fold_probs, axis=0)  # (N_test, num_classes)
    test_preds_folds.append(fold_probs)

print("Fold best val accs:", fold_best_accs)

# ======================
# 跨折 Ensemble
# ======================
test_preds_folds = np.array(test_preds_folds)      # (n_splits, N_test, num_classes)
ensemble_probs = np.mean(test_preds_folds, axis=0) # (N_test, num_classes)
final_preds = np.argmax(ensemble_probs, axis=1)

# ======================
# 生成提交文件
# ======================
submission_df = pd.DataFrame({
    "ID": [os.path.basename(f) for f in test_files],
    "Emotion": final_preds
})
submission_df.to_csv("submission_resnet50_k3_tta.csv", index=False)
print("Saved submission_resnet50_k3_tta.csv")

