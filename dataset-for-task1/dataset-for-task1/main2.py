import os
import cv2
import numpy as np
import pandas as pd
import warnings

# --- 机器学习模型 ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# --- 特征工程库 ---
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

warnings.filterwarnings('ignore')

# ==========================================
# 1. 自动路径配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'TRAIN_DIR': os.path.join(CURRENT_DIR, 'train'),
    'TEST_DIR':  os.path.join(CURRENT_DIR, 'test'),
    'TEMPLATE_CSV': os.path.join(CURRENT_DIR, 'submission-for-task1.csv'),
    'OUTPUT_FILE': os.path.join(CURRENT_DIR, 'submission_final_boost.csv'),
    'IMG_SIZE': 128,  
    'SEED': 42
}

# ==========================================
# 2. 图像预处理
# ==========================================
def remove_background(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def preprocess_image(img):
    if img is None: return None
    img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = remove_background(img)
    return img

# ==========================================
# 3. 特征工程 (新增植物专属特征)
# ==========================================

def extract_exg_hue_stats(img):
    """
    [新增] 提取 ExG (超绿指数) 和 Hue (色调) 的统计特征
    这对于区分不同种类的绿色植物非常敏感且有效。
    """
    # 1. 计算 ExG (Excess Green Index)
    # 这是一个专门用于植物分割的指数: 2G - R - B
    img_float = img.astype(np.float32)
    B, G, R = cv2.split(img_float)
    exg = 2 * G - R - B
    exg_mean = np.mean(exg)
    exg_std = np.std(exg)
    
    # 2. 计算 Hue (色调) 统计
    # 不同的草，其绿色的"波长"是有细微区别的
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 只统计非背景区域 (H > 0)
    valid_mask = h > 0
    if np.sum(valid_mask) > 0:
        h_mean = np.mean(h[valid_mask])
        h_std = np.std(h[valid_mask])
    else:
        h_mean, h_std = 0, 0
        
    return np.array([exg_mean, exg_std, h_mean, h_std])

def extract_hog_features(gray_img):
    return hog(gray_img, orientations=9, pixels_per_cell=(16, 16), 
               cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", feature_vector=True)

def extract_color_histogram(hsv_img):
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(gray_img):
    radius = 3; n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_glcm_features(gray_img):
    glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def get_combined_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 原有特征
    f_hog = extract_hog_features(gray)
    f_color = extract_color_histogram(hsv)
    f_lbp = extract_lbp_features(gray)
    f_glcm = extract_glcm_features(gray)
    
    # [新增] 植物专属特征
    f_exg = extract_exg_hue_stats(img)
    
    return np.hstack([f_hog, f_color, f_lbp, f_glcm, f_exg])

# ==========================================
# 4. 数据加载 (保持 4x 增强)
# ==========================================
def load_train_data(train_dir):
    print(f"[-] 正在加载训练数据 (4x 增强)...")
    if not os.path.exists(train_dir): raise FileNotFoundError(f"找不到: {train_dir}")

    X, y = [], []
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.png', '.jpg')): continue
            img = cv2.imread(os.path.join(cls_path, fname))
            if img is None: continue
            
            img_p = preprocess_image(img)
            
            # 1. 原图
            X.append(get_combined_features(img_p)); y.append(cls)
            # 2. 水平翻转
            X.append(get_combined_features(cv2.flip(img_p, 1))); y.append(cls)
            # 3. 垂直翻转
            X.append(get_combined_features(cv2.flip(img_p, 0))); y.append(cls)
            # 4. 旋转 90度
            X.append(get_combined_features(cv2.rotate(img_p, cv2.ROTATE_90_CLOCKWISE))); y.append(cls)
            
    return np.array(X), np.array(y)

# ==========================================
# 5. 主程序
# ==========================================
def main():
    print("=== 植物分类终极优化版 (Weighted Voting + ExG Features) ===")
    
    # 1. 加载
    try:
        X, y = load_train_data(CONFIG['TRAIN_DIR'])
        print(f"✓ 数据加载完毕，样本数: {len(X)}")
    except Exception as e:
        print(f"[错误] {e}"); return

    # 2. 预处理
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 构建模型 (微调权重)
    print("[-] 构建加权集成模型...")
    
    clf1 = SVC(C=100, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=CONFIG['SEED'])
    clf2 = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=CONFIG['SEED'])
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=CONFIG['SEED'])
    
    # [关键] 权重调整: [2, 1, 1]
    # SVM 在高维特征下最稳，RF和GBDT辅助
    eclf = VotingClassifier(
        estimators=[('svm', clf1), ('rf', clf2), ('gb', clf3)], 
        voting='soft',
        weights=[2, 1, 1] 
    )
    
    print("[-] 开始训练...")
    eclf.fit(X_scaled, y_encoded)
    print("✓ 训练完成！")
    
    # ==========================================
    # [新增] 输出训练集准确率
    # ==========================================
    print("\n" + "="*40)
    print("[-] 正在计算训练集准确率 (Self-Check)...")
    train_pred = eclf.predict(X_scaled)
    acc = accuracy_score(y_encoded, train_pred)
    print(f"★ 训练集准确率: {acc * 100:.2f}%")
    print("="*40 + "\n")

    # 4. 预测
    print("[-] 准备预测...")
    if not os.path.exists(CONFIG['TEMPLATE_CSV']):
        print(f"[错误] 找不到模板文件"); return

    template_df = pd.read_csv(CONFIG['TEMPLATE_CSV'])
    id_col, label_col = template_df.columns[0], template_df.columns[1]
    test_files = template_df[id_col].astype(str).tolist()
    
    print(f"[-] 处理测试集 ({len(test_files)} 张)...")
    X_test = []
    
    for fname in test_files:
        img_path = os.path.join(CONFIG['TEST_DIR'], fname)
        if not os.path.exists(img_path) and not fname.endswith('.png'): img_path += '.png'
            
        img = cv2.imread(img_path)
        if img is not None:
            img = preprocess_image(img)
            X_test.append(get_combined_features(img))
        else:
            X_test.append(np.zeros(X.shape[1]))
            
    X_test_scaled = scaler.transform(np.array(X_test))
    preds = eclf.predict(X_test_scaled)
    pred_labels = le.inverse_transform(preds)
    
    submission = pd.DataFrame({id_col: test_files, label_col: pred_labels})
    submission.to_csv(CONFIG['OUTPUT_FILE'], index=False)
    print(f"[成功] 结果已保存至: {CONFIG['OUTPUT_FILE']}")

if __name__ == '__main__':
    main()