import os
import sys
import random
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from dataset.dataset_RMI import get_dataloader

# 设置随机种子
def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONASHSEED'] = str(seed_value)

set_random_seed(42)

# 数据配置
config = {
    'batch_size_train': 32,
    'batch_size_val': 32,
    'num_workers': 2,
    'train_datalist': ['RMT_MRI-train'],
    'test_datalist': ['RMT_MRI-test'],
    'val_split_ratio': 0.2,
}

# 加载数据
trainloader, valloader = get_dataloader(
    datalist=config['train_datalist'],
    batch_size=config['batch_size_train'],
    shuffle=True,
    num_workers=config['num_workers'],
    drop_last=True,
    val_split_ratio=config['val_split_ratio'],
    random_state=42
)

testloader = get_dataloader(
    datalist=config['test_datalist'],
    batch_size=config['batch_size_val'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=False
)

# 提取特征和标签用于 sklearn 模型
def extract_features_labels(dataloader):
    features = []
    labels = []
    for batch in dataloader:
        _, feat, _, label, _ = batch
        features.extend(feat.numpy())
        labels.extend(label.numpy())
    return np.array(features), np.array(labels)

# 加载数据用于 sklearn 模型
train_features, train_labels = extract_features_labels(trainloader)
val_features, val_labels = extract_features_labels(valloader)
test_features, test_labels = extract_features_labels(testloader)

print(f"Training data shape: {train_features.shape}")
print(f"Validation data shape: {val_features.shape}")
print(f"Test data shape: {test_features.shape}")

# 评估函数
def evaluate_model(model, features, labels, model_name):
    predictions = model.predict(features)
    
    # 获取概率预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[:, 1]  # 二分类问题，取正类概率
    else:
        probs = model.predict(features)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)
    
    # 计算AUC
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # 如果只有一个类别，无法计算AUC
        auc = 0.0
    
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, precision, recall, f1, auc

# 1. 随机森林模型
def train_random_forest():
    print("\n" + "="*50)
    print("        Training Random Forest Model        ")
    print("="*50)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(train_features, train_labels)
    
    evaluate_model(model, val_features, val_labels, "Random Forest (Validation)")
    evaluate_model(model, test_features, test_labels, "Random Forest (Test)")
    
    return model

# 2. XGBoost模型
def train_xgboost():
    print("\n" + "="*50)
    print("        Training XGBoost Model        ")
    print("="*50)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(train_features, train_labels)
    
    evaluate_model(model, val_features, val_labels, "XGBoost (Validation)")
    evaluate_model(model, test_features, test_labels, "XGBoost (Test)")
    
    return model

# 主函数
def main():
    print("Starting baseline model training...")
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练和评估各个模型
    rf_model = train_random_forest()
    xgb_model = train_xgboost()
    
    print("\n" + "="*50)
    print("        Baseline Models Trained        ")
    print("="*50)

if __name__ == "__main__":
    main()
