import os
import numpy as np
import torch
from dataset.dataset_RMI import get_dataloader

# 数据配置
config = {
    'batch_size_train': 32,
    'batch_size_val': 32,
    'num_workers': 1,
    'train_datalist': ['RMT_MRI-train'],
    'test_datalist': ['RMT_MRI-test'],
    'val_split_ratio': 0.2,
}

# 加载数据
print("Loading training data...")
trainloader, valloader = get_dataloader(
    datalist=config['train_datalist'],
    batch_size=config['batch_size_train'],
    shuffle=True,
    num_workers=config['num_workers'],
    drop_last=True,
    val_split_ratio=config['val_split_ratio'],
    random_state=42
)

print("Loading test data...")
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
print("Extracting features from training data...")
train_features, train_labels = extract_features_labels(trainloader)
print("Extracting features from validation data...")
val_features, val_labels = extract_features_labels(valloader)
print("Extracting features from test data...")
test_features, test_labels = extract_features_labels(testloader)

print(f"Training data shape: {train_features.shape}")
print(f"Validation data shape: {val_features.shape}")
print(f"Test data shape: {test_features.shape}")
print(f"Training labels: {np.unique(train_labels, return_counts=True)}")
print(f"Validation labels: {np.unique(val_labels, return_counts=True)}")
print(f"Test labels: {np.unique(test_labels, return_counts=True)}")

print("Data loading test completed successfully!")
