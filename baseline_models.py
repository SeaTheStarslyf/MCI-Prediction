import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

# 设备设置
def setup_gpu(gpu_id='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = setup_gpu()
print(f"Using device: {device}")

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
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        predictions = model.predict(features)
        # 获取概率预测
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[:, 1]  # 二分类问题，取正类概率
        else:
            probs = model.predict(features)
    else:
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(device)
            outputs = model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
            # 获取概率预测
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 二分类问题，取正类概率
    
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

# 3. CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (input_dim // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_cnn():
    print("\n" + "="*50)
    print("        Training CNN Model        ")
    print("="*50)
    
    input_dim = train_features.shape[1]
    model = CNNModel(input_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in trainloader:
            _, features, _, labels, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in valloader:
                _, features, _, labels, _ = batch
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/cnn_best.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoints/cnn_best.pth', map_location=device))
   # 评估模型 - 使用整个验证集
    val_features_list = []
    val_labels_list = []
    for batch in valloader:
        _, features, _, labels, _ = batch
        val_features_list.append(features.numpy())
        val_labels_list.append(labels.numpy())
    val_features = np.concatenate(val_features_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    
    evaluate_model(model, val_features, val_labels, "CNN (Validation)")
    evaluate_model(model, test_features, test_labels, "CNN (Test)")
    
    return model

# 4. Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        # 确保d_model能被nhead整除
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.embedding = nn.Linear(input_dim, d_model)
        # 位置编码的形状应该是(1, 1, d_model)，因为输入是(batch_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True  # 设置batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        # 输入形状: (batch_size, input_dim)
        x = self.embedding(x)  # 输出形状: (batch_size, d_model)
        
        # 添加位置编码
        x = x.unsqueeze(1)  # 输出形状: (batch_size, 1, d_model)
        x = x + self.positional_encoding  # 广播位置编码到所有批次
        
        x = self.transformer_encoder(x)  # 输出形状: (batch_size, 1, d_model)
        x = x.squeeze(1)  # 输出形状: (batch_size, d_model)
        x = self.fc(x)  # 输出形状: (batch_size, num_classes)
        return x

def train_transformer():
    print("\n" + "="*50)
    print("        Training Transformer Model        ")
    print("="*50)
    
    input_dim = train_features.shape[1]
    model = TransformerModel(input_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in trainloader:
            _, features, _, labels, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in valloader:
                _, features, _, labels, _ = batch
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/transformer_best.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoints/transformer_best.pth', map_location=device))
   # 评估模型 - 使用整个验证集
    val_features_list = []
    val_labels_list = []
    for batch in valloader:
        _, features, _, labels, _ = batch
        val_features_list.append(features.numpy())
        val_labels_list.append(labels.numpy())
    val_features = np.concatenate(val_features_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    
    evaluate_model(model, val_features, val_labels, "Transformer (Validation)")
    evaluate_model(model, test_features, test_labels, "Transformer (Test)")
    
    return model

# 主函数
def main():
    print("Starting baseline model training...")
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练和评估各个模型
    rf_model = train_random_forest()
    xgb_model = train_xgboost()
    cnn_model = train_cnn()
    transformer_model = train_transformer()
    
    print("\n" + "="*50)
    print("        All Baseline Models Trained        ")
    print("="*50)

if __name__ == "__main__":
    main()
