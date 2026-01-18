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
import torchvision.transforms.functional as F

# ======================
# 配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

DATA_ROOT = "/kaggle/input/neu-image-emotion-classification/fer_data/fer_data"
train_dir = os.path.join(DATA_ROOT, "train")
test_dir  = os.path.join(DATA_ROOT, "test")

class_names = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
num_classes = len(class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}

img_size = 224
batch_size = 64
epochs = 15          # EfficientNet 稍微减一点 epoch，控制时间
n_splits = 3
base_lr = 3e-4

# 现成的 ResNet50 三折权重
resnet_weight_paths = [
    "/kaggle/input/resnet50-fold01-weights/best_resnet50_fold0.pth",
    "/kaggle/input/resnet50-fold01-weights/best_resnet50_fold1.pth",
    "/kaggle/working/best_resnet50_fold2.pth",
]

# EfficientNet 三折权重保存位置
eff_weight_paths = [
    "/kaggle/working/best_effnet_fold0.pth",
    "/kaggle/working/best_effnet_fold1.pth",
    "/kaggle/working/best_effnet_fold2.pth",
]

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
# Transforms
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
# 读取训练数据 & 类权重
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

class_counts = np.bincount(train_labels, minlength=num_classes)
print("Class counts:", dict(zip(class_names, class_counts)))

class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# ======================
# 模型：ResNet50（和你原来保持一致，用于加载老权重）
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

def get_resnet50(num_classes):
    return ResNet50FER(num_classes)

# ======================
# 模型：EfficientNet-B0（新 backbone）
# ======================
class EfficientNetB0FER(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

def get_effnet_b0(num_classes):
    return EfficientNetB0FER(num_classes)

# ======================
# 强 TTA 推理（4 次）
# ======================
def predict_with_strong_tta(model, loader):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc="Predicting (strong TTA)", leave=False):
            imgs = imgs.to(device)

            img1 = imgs
            img2 = torch.flip(imgs, dims=[3])
            img3 = F.resize(imgs, int(img_size * 1.2))
            img3 = F.center_crop(img3, img_size)
            img4 = F.resize(imgs, int(img_size * 1.4))
            img4 = F.center_crop(img4, img_size)

            aug_imgs = [img1, img2, img3, img4]

            probs_sum = 0
            for aug in aug_imgs:
                logits = model(aug)
                probs = torch.softmax(logits, dim=1)
                probs_sum += probs

            probs_mean = probs_sum / len(aug_imgs)
            all_probs.append(probs_mean.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs

# ======================
# Test Loader
# ======================
test_files = sorted([
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])
print("Total test images:", len(test_files))

test_dataset = EmotionDataset(test_files, labels=None, transform=val_test_transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=1)

# ======================
# 1) 加载 ResNet50 三折权重 + 强 TTA 推理
# ======================
resnet_fold_probs = []

for fold, w_path in enumerate(resnet_weight_paths):
    print(f"\n=== ResNet50 Fold {fold} | load {w_path} ===")
    assert os.path.exists(w_path), f"ResNet weight not found: {w_path}"
    model_r = get_resnet50(num_classes).to(device)
    state = torch.load(w_path, map_location=device)
    model_r.load_state_dict(state)
    fold_probs = predict_with_strong_tta(model_r, test_loader)
    resnet_fold_probs.append(fold_probs)

resnet_fold_probs = np.array(resnet_fold_probs)
resnet_probs = np.mean(resnet_fold_probs, axis=0)   # (N_test, num_classes)
print("ResNet probs shape:", resnet_probs.shape)

# ======================
# 2) 训练 EfficientNet-B0 三折
# ======================
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
eff_fold_best = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_files, train_labels)):
    print(f"\n========== EfficientNet-B0 Fold {fold+1}/{n_splits} ==========")

    train_ds = EmotionDataset(train_files[tr_idx],
                              train_labels[tr_idx],
                              transform=train_transform)
    val_ds   = EmotionDataset(train_files[va_idx],
                              train_labels[va_idx],
                              transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=1)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=1)

    model_e = get_effnet_b0(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor,
                                    label_smoothing=0.05)
    optimizer = optim.AdamW(model_e.parameters(), lr=base_lr, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_path = eff_weight_paths[fold]

    for epoch in range(epochs):
        model_e.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"EffNet Fold {fold+1} Epoch {epoch+1}/{epochs} - Train",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_e(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            total += bs
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        model_e.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"EffNet Fold {fold+1} Epoch {epoch+1}/{epochs} - Val",
                                     leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model_e(imgs)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"EffNet Fold {fold+1} Epoch {epoch+1}/{epochs} "
              f"- Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_e.state_dict(), best_path)

    print(f"EffNet Fold {fold+1} best Val Acc = {best_val_acc:.4f}")
    eff_fold_best.append(best_val_acc)

print("EfficientNet fold best val accs:", eff_fold_best)

# ======================
# 3) 加载 EfficientNet 三折权重 + 强 TTA 推理
# ======================
eff_fold_probs = []

for fold, w_path in enumerate(eff_weight_paths):
    print(f"\n=== EffNet Fold {fold} | load {w_path} ===")
    assert os.path.exists(w_path), f"EffNet weight not found: {w_path}"
    model_e = get_effnet_b0(num_classes).to(device)
    state = torch.load(w_path, map_location=device)
    model_e.load_state_dict(state)
    fold_probs = predict_with_strong_tta(model_e, test_loader)
    eff_fold_probs.append(fold_probs)

eff_fold_probs = np.array(eff_fold_probs)
eff_probs = np.mean(eff_fold_probs, axis=0)
print("EffNet probs shape:", eff_probs.shape)

# ======================
# 4) 最终 ResNet50 + EfficientNet-B0 ensemble
# ======================
alpha = 0.6  # ResNet 权重
probs_final = alpha * resnet_probs + (1 - alpha) * eff_probs
final_preds = np.argmax(probs_final, axis=1)

submission_df = pd.DataFrame({
    "ID": [os.path.basename(f) for f in test_files],
    "Emotion": final_preds
})
save_name = "submission_resnet50_effnet_strongtta_ensemble.csv"
submission_df.to_csv(save_name, index=False)
print("Saved", save_name)
