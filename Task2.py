import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# ----------------------
# 通用配置
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 5

teacher_epochs = 25
student_epochs = 25
n_splits = 5
patience = 5
pseudo_threshold = 0.95

train_dir = "/kaggle/input/neu-plant-seedling-classification-num2-2025fall/dataset-for-task2/dataset-for-task2/train"
test_dir  = "/kaggle/input/neu-plant-seedling-classification-num2-2025fall/dataset-for-task2/dataset-for-task2/test"

classes = sorted(os.listdir(train_dir))

os.makedirs("weights_b2_teacher", exist_ok=True)
os.makedirs("weights_b2_student", exist_ok=True)
os.makedirs("weights_b3_teacher", exist_ok=True)
os.makedirs("weights_b3_student", exist_ok=True)

# ----------------------
# Dataset
# ----------------------
class PlantDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img

# ----------------------
# 读取数据
# ----------------------
train_files, train_labels = [], []
for idx, cls in enumerate(classes):
    cls_dir = os.path.join(train_dir, cls)
    for f in os.listdir(cls_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            train_files.append(os.path.join(cls_dir, f))
            train_labels.append(idx)

train_files = np.array(train_files)
train_labels = np.array(train_labels)

train_df = pd.DataFrame({
    "filepath": train_files,
    "label": train_labels,
})

test_files = sorted([
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])

# ----------------------
# 根据模型类型返回 模型 + 尺寸 + batch_size + transform
# ----------------------
def get_model_and_transforms(model_name):
    if model_name == "b2":
        img_size = 288
        batch_size = 16
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
    elif model_name == "b3":
        img_size = 300
        batch_size = 12
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
    else:
        raise ValueError("model_name must be 'b2' or 'b3'")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    tta_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    return model, img_size, batch_size, train_transform, val_transform, tta_transform

# ----------------------
# 通用：K 折训练（teacher / student）
# ----------------------
def train_one_fold(df, train_idx, val_idx, fold, save_prefix, model_name,
                   train_transform, val_transform, batch_size, epochs, tag):
    print(f"\n==== [{tag} {model_name}] Fold {fold+1}/{n_splits} ====")

    train_ds = PlantDataset(df["filepath"].values[train_idx],
                            df["label"].values[train_idx],
                            transform=train_transform)
    val_ds   = PlantDataset(df["filepath"].values[val_idx],
                            df["label"].values[val_idx],
                            transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model, _, _, _, _, _ = get_model_and_transforms(model_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    wait = 0

    for epoch in range(1, epochs+1):
        # train
        model.train()
        running_loss, total = 0.0, 0
        for imgs, lbls in tqdm(train_loader,
                               desc=f"[{tag} {model_name} Fold {fold+1}] Epoch {epoch}/{epochs}",
                               leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            total += bs
        train_loss = running_loss / total

        # val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)
        val_acc = val_correct / val_total
        scheduler.step()

        print(f"[{tag} {model_name} Fold {fold+1}] Epoch {epoch}/{epochs} - Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), f"{save_prefix}_fold{fold}.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"[{tag} {model_name} Fold {fold+1}] Early stopping at epoch {epoch}, best val acc={best_val_acc:.4f}")
                break

    return best_val_acc

def train_kfold(df, save_prefix, model_name, train_transform, val_transform, batch_size, epochs, tag, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_best_accs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["filepath"].values, df["label"].values)):
        best_acc = train_one_fold(df, train_idx, val_idx, fold, save_prefix,
                                  model_name, train_transform, val_transform,
                                  batch_size, epochs, tag)
        fold_best_accs.append(best_acc)
        print(f"[{tag} {model_name}] Fold {fold+1} best Val Acc = {best_acc:.4f}")
    print(f"{tag} {model_name} All folds best val acc:", fold_best_accs)
    return np.array(fold_best_accs)

# ----------------------
# 通用：K 折预测 test（集成 + TTA）
# ----------------------
def predict_test_with_kfold_with_accs(save_prefix, model_name,
                                      fold_best_accs, val_transform, tta_transform,
                                      top_k=3, n_tta=4):
    fold_best_accs = np.array(fold_best_accs)
    topk_indices = np.argsort(fold_best_accs)[-top_k:]
    print(f"[{model_name}] Use folds (0-based indices):", topk_indices)

    all_test_probs = []

    for fold in topk_indices:
        print(f"[{model_name}] Predicting with fold {fold+1} model ...")
        model, _, _, _, _, _ = get_model_and_transforms(model_name)
        model = model.to(device)
        model.load_state_dict(torch.load(f"{save_prefix}_fold{fold}.pth", map_location=device))
        model.eval()

        fold_test_probs = []

        with torch.no_grad():
            for fpath in tqdm(test_files, desc=f"[{model_name}] TTA Predict Fold {fold+1}", leave=False):
                img = Image.open(fpath).convert("RGB")

                probs_sum = torch.zeros(1, num_classes, device=device)

                # 原图
                x = val_transform(img).unsqueeze(0).to(device)
                probs_sum += F.softmax(model(x), dim=1)

                # TTA
                for _ in range(n_tta):
                    x_tta = tta_transform(img).unsqueeze(0).to(device)
                    probs_sum += F.softmax(model(x_tta), dim=1)

                probs_mean = (probs_sum / (1 + n_tta)).cpu().numpy()
                fold_test_probs.append(probs_mean)

        fold_test_probs = np.concatenate(fold_test_probs, axis=0)
        all_test_probs.append(fold_test_probs)

    ensemble_probs = np.mean(all_test_probs, axis=0)
    final_preds = np.argmax(ensemble_probs, axis=1)

    return ensemble_probs, final_preds

# ----------------------
# 主流程：先训 B2，再训 B3，最后集成
# ----------------------
all_probs = {}

for model_name in ["b2", "b3"]:
    print(f"\n################## Training & pseudo-labeling with {model_name.upper()} ##################")

    model, img_size, batch_size_model, train_tf, val_tf, tta_tf = get_model_and_transforms(model_name)

    # Round 1: teacher
    teacher_prefix = f"weights_{model_name}_teacher/teacher_{model_name}"
    print(f"########## Round 1 ({model_name}): Train teacher on original train ##########")
    teacher_fold_accs = train_kfold(train_df, teacher_prefix, model_name,
                                    train_tf, val_tf, batch_size_model,
                                    teacher_epochs, tag="Teacher")

    print(f"########## Round 1 ({model_name}): Teacher predicting pseudo labels ##########")
    teacher_probs, _ = predict_test_with_kfold_with_accs(
        save_prefix=teacher_prefix,
        model_name=model_name,
        fold_best_accs=teacher_fold_accs,
        val_transform=val_tf,
        tta_transform=tta_tf,
        top_k=3,
        n_tta=2
    )

    pseudo_labels = teacher_probs.argmax(axis=1)
    pseudo_conf   = teacher_probs.max(axis=1)
    pseudo_mask = pseudo_conf >= pseudo_threshold
    print(f"[{model_name}] Pseudo-label threshold = {pseudo_threshold}, keep {pseudo_mask.sum()} / {len(test_files)} samples.")

    pseudo_df = pd.DataFrame({
        "filepath": np.array(test_files)[pseudo_mask],
        "label": pseudo_labels[pseudo_mask],
    })

    # Round 2: student
    print(f"########## Round 2 ({model_name}): Train student on train + pseudo ##########")
    train_pl_df = pd.concat([train_df, pseudo_df], ignore_index=True)
    print(f"[{model_name}] Train + pseudo total size: {len(train_pl_df)}")

    student_prefix = f"weights_{model_name}_student/student_{model_name}"
    student_fold_accs = train_kfold(train_pl_df, student_prefix, model_name,
                                    train_tf, val_tf, batch_size_model,
                                    student_epochs, tag="Student")

    print(f"########## Round 2 ({model_name}): Student predicting test ##########")
    student_probs, student_preds = predict_test_with_kfold_with_accs(
        save_prefix=student_prefix,
        model_name=model_name,
        fold_best_accs=student_fold_accs,
        val_transform=val_tf,
        tta_transform=tta_tf,
        top_k=3,
        n_tta=4
    )

    all_probs[model_name] = student_probs

# ----------------------
# 最终 B2 + B3 集成与提交
# ----------------------
print("########## Ensemble B2 + B3 ##########")
b2_probs = all_probs["b2"]
b3_probs = all_probs["b3"]

ensemble_probs = (b2_probs + b3_probs) / 2.0
final_preds = ensemble_probs.argmax(axis=1)

submission_df = pd.DataFrame({
    "ID": [os.path.basename(f) for f in test_files],
    "Category": [classes[p] for p in final_preds]
})
submission_df.to_csv("submission_ensemble_b2_b3.csv", index=False)
print("Saved submission_ensemble_b2_b3.csv")
