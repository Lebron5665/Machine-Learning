import os
import cv2
import numpy as np
import pandas as pd
import warnings

# --- 机器学习模型 ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    'OUTPUT_FILE': os.path.join(CURRENT_DIR, 'submission_final_enhanced.csv'),
    
    'IMG_SIZE': 128,  # 保持图片尺寸
    'SEED': 42
}

# ==========================================
# 2. 图像预处理：背景去除 (关键步骤!)
# ==========================================
def remove_background(img):
    """
    使用 HSV 颜色空间提取绿色植物，去除土壤和石头背景。
    这一步能大幅提高特征的纯度。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 绿色的 HSV 范围 (根据经验调整)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 创建掩膜
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 对掩膜进行形态学操作，填补孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 应用掩膜，背景变黑
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def preprocess_image(img):
    if img is None: return None
    img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
    # 先去噪
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # 再去背景
    img = remove_background(img)
    return img

# ==========================================
# 3. 高级特征工程 (HOG + Color + LBP + GLCM)
# ==========================================

def extract_hog_features(gray_img):
    """特征 1: HOG (形状/边缘)"""
    return hog(gray_img, orientations=9, pixels_per_cell=(16, 16), 
               cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", feature_vector=True)

def extract_color_histogram(hsv_img):
    """特征 2: HSV 颜色直方图 (色彩分布)"""
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(gray_img):
    """特征 3: LBP (局部二值模式 - 纹理特征)"""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # 归一化
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_glcm_features(gray_img):
    """特征 4: GLCM (灰度共生矩阵 - 统计纹理)"""
    # 计算共生矩阵，距离=1，角度=0, 45, 90, 135
    glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    # 提取统计量: 对比度, 差异性, 同质性, 能量, 相关性
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def get_combined_features(img):
    """融合所有特征"""
    # 转换为灰度图和HSV图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    f_hog = extract_hog_features(gray)
    f_color = extract_color_histogram(hsv)
    f_lbp = extract_lbp_features(gray)
    f_glcm = extract_glcm_features(gray)
    
    # 拼接所有向量
    return np.hstack([f_hog, f_color, f_lbp, f_glcm])

# ==========================================
# 4. 数据加载
# ==========================================
def load_train_data(train_dir):
    print(f"[-] 正在加载训练数据 (启用去背景+增强): {train_dir}")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"找不到: {train_dir}")

    X, y = [], []
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.png', '.jpg')): continue
            
            img = cv2.imread(os.path.join(cls_path, fname))
            if img is None: continue
            
            img_processed = preprocess_image(img)
            
            # --- 多重数据增强 (扩充到 4 倍) ---
            # 1. 原图
            X.append(get_combined_features(img_processed))
            y.append(cls)
            
            # 2. 水平翻转
            X.append(get_combined_features(cv2.flip(img_processed, 1)))
            y.append(cls)
            
            # 3. 垂直翻转
            X.append(get_combined_features(cv2.flip(img_processed, 0)))
            y.append(cls)
            
            # 4. 旋转 90度 (可选，这里加上增加鲁棒性)
            X.append(get_combined_features(cv2.rotate(img_processed, cv2.ROTATE_90_CLOCKWISE)))
            y.append(cls)
            
    return np.array(X), np.array(y)

# ==========================================
# 5. 主程序：集成模型训练
# ==========================================
def main():
    print("=== 高级植物分类系统 (Voting Ensemble + Advanced Features) ===")
    
    # 1. 加载数据
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
    
    # 3. 构建集成模型 (Voting Classifier)
    print("[-] 正在构建集成模型 (SVM + Random Forest + Gradient Boosting)...")
    
    # 模型 1: SVM (擅长高维特征)
    clf1 = SVC(C=100, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=CONFIG['SEED'])
    
    # 模型 2: 随机森林 (擅长抗噪和非线性)
    clf2 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=CONFIG['SEED'])
    
    # 模型 3: 梯度提升 (擅长修正错误)
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=CONFIG['SEED'])
    
    # 软投票 (Soft Voting): 基于概率的加权平均，通常比硬投票效果好
    eclf = VotingClassifier(estimators=[
        ('svm', clf1), ('rf', clf2), ('gb', clf3)
    ], voting='soft')
    
    print("[-] 开始训练集成模型 (这可能需要几分钟)...")
    # 使用交叉验证看一眼准确率
    # scores = cross_val_score(eclf, X_scaled, y_encoded, cv=3, scoring='f1_macro')
    # print(f"    预估交叉验证 F1 Score: {scores.mean():.4f}")
    
    eclf.fit(X_scaled, y_encoded)
    print("✓ 训练完成！")

    # 4. 预测
    print("\n[-] 准备预测...")
    
    # 读取模板
    if not os.path.exists(CONFIG['TEMPLATE_CSV']):
        print(f"[错误] 找不到模板文件: {CONFIG['TEMPLATE_CSV']}")
        return

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
            print(f"  [Warn] 缺失图片: {fname}")
            X_test.append(np.zeros(X.shape[1])) # 占位
            
    X_test_scaled = scaler.transform(np.array(X_test))
    
    # 预测并解码
    preds = eclf.predict(X_test_scaled)
    pred_labels = le.inverse_transform(preds)
    
    # 保存
    submission = pd.DataFrame({id_col: test_files, label_col: pred_labels})
    submission.to_csv(CONFIG['OUTPUT_FILE'], index=False)
    print(f"\n[成功] 增强版结果已保存至: {CONFIG['OUTPUT_FILE']}")

if __name__ == '__main__':
    main()