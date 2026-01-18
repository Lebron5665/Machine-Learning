import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image

# ================= SOTA 设置 =================
CONFIG = {
    'train_dir': './train',
    'test_dir': './test',
    'pred_output_dir': './image',
    'img_size': 768,       
    'batch_size': 2,       
    'epochs': 70,
    'lr': 2e-4, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 自动创建目录
os.makedirs(CONFIG['pred_output_dir'], exist_ok=True)

# ================= 1. 数据增强 =================
def get_train_transforms():
    return A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # 弹性形变对血管分割至关重要
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ================= 2. Dataset =================
class VesselDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'image') if os.path.exists(os.path.join(root_dir, 'image')) else root_dir
        if mode == 'train':
            self.mask_dir = os.path.join(root_dir, 'label') if os.path.exists(os.path.join(root_dir, 'label')) else root_dir
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'train':
            base_name = os.path.splitext(img_name)[0]
            mask_name = None
            for f in os.listdir(self.mask_dir):
                if os.path.splitext(f)[0] == base_name:
                    mask_name = f; break
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = np.array(Image.open(mask_path).convert('L'))
            
            # 标签归一化处理
            mask = mask.astype(np.float32) / 255.0
            if mask.mean() > 0.5: mask = 1.0 - mask
            mask = (mask > 0.5).astype(np.float32)
            
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            return img, mask.unsqueeze(0)
        else:
            h, w = img.shape[:2]
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            return img, h, w, img_name

# ================= 3. Combo Loss =================
class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.focal = smp.losses.FocalLoss(mode='binary') # 加入Focal Loss关注难分样本
        
    def forward(self, logits, mask):
        # 0.4 BCE + 0.4 Dice + 0.2 Focal
        return 0.4 * self.bce(logits, mask) + 0.4 * self.dice(logits.sigmoid(), mask) + 0.2 * self.focal(logits, mask)

# ================= 4. 主程序 =================
def run():
    print(f"[-] 设备: {CONFIG['device']} | 尺寸: {CONFIG['img_size']} | 模型: EfficientNet-B4")
    
    train_ds = VesselDataset(CONFIG['train_dir'], mode='train', transform=get_train_transforms())
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    
    #骨干网络 EfficientNet-B4
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = ComboLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # --- 训练循环 ---
    print("[-] 开始训练...")
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        run_loss = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}/{CONFIG['epochs']}", leave=False)
        
        for img, mask in pbar:
            img, mask = img.to(CONFIG['device']), mask.to(CONFIG['device'])
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = run_loss / len(train_dl)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model_sota.pth")
    
    print("[-] 训练完成，开始 TTA 增强预测...")
    
    # --- 预测 (包含 TTA Test Time Augmentation) ---
    model.load_state_dict(torch.load("best_model_sota.pth"))
    model.eval()
    
    test_ds = VesselDataset(CONFIG['test_dir'], mode='test', transform=get_test_transforms())
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for img, h_orig, w_orig, name in tqdm(test_dl):
            img = img.to(CONFIG['device'])
            
            # --- 手动 TTA (Test Time Augmentation) ---
            # 1. 原图预测
            pred_1 = model(img).sigmoid()
            
            # 2. 水平翻转预测
            img_flip = torch.flip(img, [3])
            pred_flip = model(img_flip).sigmoid()
            pred_2 = torch.flip(pred_flip, [3])
            
            # 3. 垂直翻转预测
            img_vflip = torch.flip(img, [2])
            pred_vflip = model(img_vflip).sigmoid()
            pred_3 = torch.flip(pred_vflip, [2])
            
            # 平均结果 (Ensemble)
            pred = (pred_1 + pred_2 + pred_3) / 3.0
            pred = pred.cpu().numpy()[0, 0]
            
            # 还原尺寸
            pred_resized = cv2.resize(pred, (w_orig.item(), h_orig.item()))
            
            # 后处理：形态学操作 (填补血管中的小黑洞)
            mask_binary = (pred_resized > 0.5).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # 闭运算：连接断裂的血管
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
            
            # 保存为白色血管图
            final_img = mask_binary * 255
            cv2.imwrite(os.path.join(CONFIG['pred_output_dir'], name[0]), final_img)
            
    print(f"[-] 预测完成！")

if __name__ == '__main__':
    run()
