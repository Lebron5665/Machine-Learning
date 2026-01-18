import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp  # 核心库

# ================= 配置参数 =================
CONFIG = {
    'base_path': '.',               # 根目录
    'train_img_dir': 'train',       # 训练集图片文件夹
    'test_img_dir': 'test',         # 测试集图片文件夹
    'train_csv': 'fovea_localization_train_GT.csv', # 训练标签
    'submit_file': 'submission.csv',# 输出文件名
    'vis_dir': 'vis_results',       # 预测结果可视化保存路径（用于检查）
    
    'img_size': 512,                # 输入尺寸 (512x512是眼底图的标准黄金尺寸)
    'batch_size': 4,                # 显存如果不够（报错OOM），改成 2
    'epochs': 150,                   # 训练轮数
    'lr': 3e-4,                     # 学习率
    'sigma': 20,                    # 热图高斯核半径 (控制光斑大小)
    'threshold': 0.1,               # 判定“不可见”的阈值
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 创建可视化目录
os.makedirs(CONFIG['vis_dir'], exist_ok=True)
print(f"[-] Using device: {CONFIG['device']}")

# ================= 1. 图像增强工具 (提分关键) =================
def apply_clahe(img_path):
    """
    使用 CLAHE (对比度受限自适应直方图均衡化) 增强眼底图像
    能显著提高黄斑的可视度
    """
    img = cv2.imread(img_path)
    if img is None: 
        return Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换到 LAB 空间对亮度通道做增强
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final)

def generate_heatmap(size, center, sigma=20):
    """生成高斯热图作为训练目标"""
    w, h = size
    x0, y0 = center
    
    # 如果坐标是 (0,0) 或负数，认为不可见，返回全黑热图
    if x0 <= 1 and y0 <= 1:
        return torch.zeros((h, w), dtype=torch.float32)
    
    # 生成网格
    x = torch.arange(0, w, 1, dtype=torch.float32)
    y = torch.arange(0, h, 1, dtype=torch.float32)
    y, x = torch.meshgrid(y, x, indexing='ij')
    
    # 高斯公式
    heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return heatmap

# ================= 2. 数据集定义 =================
class FoveaDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, mode='train', transform=None):
        self.img_dir = os.path.join(CONFIG['base_path'], img_dir)
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            # 读取标签 CSV
            csv_path = os.path.join(CONFIG['base_path'], csv_file)
            try:
                self.df = pd.read_csv(csv_path, encoding='utf-8')
            except:
                self.df = pd.read_csv(csv_path, encoding='gbk')
            
            self.img_names = self.df.iloc[:, 0].values
            self.coords = self.df.iloc[:, 1:3].values # [X, Y]
        else:
            # 测试集读取所有图片
            self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        raw_name = str(self.img_names[idx]).strip()
        
        # --- 智能查找图片文件 (处理 .jpg/.png 后缀问题) ---
        img_path = None
        if self.mode == 'test':
             img_path = os.path.join(self.img_dir, raw_name)
        else:
            # 训练集可能有名字匹配问题 (如 CSV是 '36', 文件是 '0036.jpg')
            candidates = [raw_name, raw_name + '.jpg', raw_name + '.png']
            if raw_name.isdigit():
                candidates.append(raw_name.zfill(4) + '.jpg')
                candidates.append(raw_name.zfill(4) + '.png')
            
            for c in candidates:
                p = os.path.join(self.img_dir, c)
                if os.path.exists(p):
                    img_path = p; break
            
            if img_path is None:
                # 容错：如果找不到图，返回黑图
                print(f"[Warn] Image not found: {raw_name}")
                return torch.zeros((3, CONFIG['img_size'], CONFIG['img_size'])), torch.zeros((1, CONFIG['img_size'], CONFIG['img_size']))

        # 读取并应用 CLAHE 增强
        img = apply_clahe(img_path)
        w_orig, h_orig = img.size
        
        if self.transform:
            img_tensor = self.transform(img)
        
        if self.mode == 'train':
            x, y = self.coords[idx]
            
            # 坐标映射到 512x512
            scale_x = CONFIG['img_size'] / w_orig
            scale_y = CONFIG['img_size'] / h_orig
            target_x, target_y = x * scale_x, y * scale_y
            
            # 生成热图标签 [1, 512, 512]
            heatmap = generate_heatmap(
                (CONFIG['img_size'], CONFIG['img_size']), 
                (target_x, target_y), 
                sigma=CONFIG['sigma']
            )
            return img_tensor, heatmap.unsqueeze(0)
        else:
            return img_tensor, w_orig, h_orig, raw_name

# ================= 3. 训练函数 =================
def train_model():
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds = FoveaDataset(CONFIG['train_img_dir'], CONFIG['train_csv'], mode='train', transform=train_transform)
    # Windows下建议 num_workers=0，Linux可设为4
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    # === 构建模型 (SMP U-Net) ===
    print("[-] Building SMP U-Net (EfficientNet-B4)...")
    model = smp.Unet(
        encoder_name="efficientnet-b4",  # 强力骨干网
        encoder_weights="imagenet",      # 加载预训练权重
        in_channels=3,
        classes=1,                       # 输出单通道热图
    ).to(CONFIG['device'])
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    print(f"[-] Start training on {len(ds)} images...")
    best_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        run_loss = 0.0
        pbar = tqdm(dl, desc=f"Ep {epoch+1}/{CONFIG['epochs']}")
        
        for imgs, maps in pbar:
            imgs, maps = imgs.to(CONFIG['device']), maps.to(CONFIG['device'])
            
            optimizer.zero_grad()
            preds = model(imgs)        # 输出 Logits
            preds = torch.sigmoid(preds) # 转为 0-1 概率图
            
            loss = criterion(preds, maps)
            
            loss.backward()
            optimizer.step()
            
            # Loss 乘以 1000 只是为了打印好看，不影响梯度
            run_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss.item()*1000:.4f}"})
        
        scheduler.step()
        epoch_loss = run_loss / len(dl)
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"    [Saved] Best Model Loss: {best_loss*1000:.4f}")

    return model

# ================= 4. 预测与生成提交文件 =================
def predict_and_submit(model_path='best_unet_model.pth'):
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = FoveaDataset(CONFIG['test_img_dir'], mode='test', transform=test_transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 加载模型
    model = smp.Unet(encoder_name="efficientnet-b4", classes=1).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    results = []
    print(f"[-] Starting inference with TTA... Visualizations in '{CONFIG['vis_dir']}'")
    
    with torch.no_grad():
        for i, (imgs, w_orig, h_orig, name_tuple) in enumerate(tqdm(dl)):
            imgs = imgs.to(CONFIG['device'])
            
            # === TTA: Test Time Augmentation ===
            # 1. 正向预测
            pred1 = torch.sigmoid(model(imgs))
            
            # 2. 水平翻转预测
            pred2 = torch.sigmoid(model(torch.flip(imgs, [3])))
            pred2 = torch.flip(pred2, [3]) # 翻转回来
            
            # 3. 平均
            heatmap = (pred1 + pred2) / 2.0 # [1, 1, 512, 512]
            
            # === 坐标解析 ===
            hm_np = heatmap.squeeze().cpu().numpy()
            # 找到最大值位置
            y_idx, x_idx = np.unravel_index(np.argmax(hm_np), hm_np.shape)
            max_val = hm_np[y_idx, x_idx]
            
            file_name = str(name_tuple[0])
            scale_x = w_orig.item() / CONFIG['img_size']
            scale_y = h_orig.item() / CONFIG['img_size']
            
            # 判断是否可见
            if max_val < CONFIG['threshold']:
                pred_x, pred_y = 0.0, 0.0
            else:
                pred_x = x_idx * scale_x
                pred_y = y_idx * scale_y
            
            # === 格式化 ID ===
            # 去除后缀: "0081.jpg" -> "0081"
            base_name = os.path.splitext(file_name)[0]
            # 去除前导零: "0081" -> "81"
            file_id = str(int(base_name)) if base_name.isdigit() else base_name
            
            # 保存到列表
            results.append([f"{file_id}_Fovea_X", pred_x])
            results.append([f"{file_id}_Fovea_Y", pred_y])
            
            # === 可视化前10张 ===
            if i < 10:
                vis_path = os.path.join(CONFIG['test_img_dir'], file_name)
                vis_img = cv2.imread(vis_path)
                if vis_img is not None:
                    # 画十字
                    cv2.drawMarker(vis_img, (int(pred_x), int(pred_y)), (0, 0, 255), 
                                   markerType=cv2.MARKER_CROSS, thickness=3, markerSize=30)
                    cv2.imwrite(os.path.join(CONFIG['vis_dir'], f"vis_{file_name}"), vis_img)

    # === 生成 CSV ===
    submission_df = pd.DataFrame(results, columns=['ImageID', 'value'])
    submission_df.to_csv(CONFIG['submit_file'], index=False)
    print(f"[-] Submission saved to {CONFIG['submit_file']}")

if __name__ == '__main__':
    # 1. 训练
    if os.path.exists(os.path.join(CONFIG['base_path'], CONFIG['train_csv'])):
        train_model()
    else:
        print("[!] Train CSV not found. Skipping training.")

    # 2. 预测
    if os.path.exists('best_unet_model.pth'):
        predict_and_submit()
    else:
        print("[!] No model found. Please train first.")
