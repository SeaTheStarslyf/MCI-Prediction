import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from dataset.dataset_RMI import get_dataloader
from medblip.modeling_medblip import MedBLIPModel
try:
    import joblib
except Exception as e:
    joblib = None
    _joblib_import_error = e

# GPU设备设置
def setup_gpu(gpu_id='1'):
    """设置并验证GPU设备是否可用"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print(f"警告: GPU ID {gpu_id} 不可用，将使用CPU")
            return torch.device('cpu')
        else:
            print(f"使用GPU ID {gpu_id}, 可用GPU数量: {device_count}")
            return torch.device('cuda')
    else:
        print("警告: CUDA不可用，将使用CPU")
        return torch.device('cpu')

# 设备设置
device = setup_gpu(gpu_id='1')
print(f"Using device: {device}")

# set random seed
seed = 42
def set_random_seed(seed_value):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 多GPU情况
    os.environ['PYTHONASHSEED'] = str(seed_value)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    # 确保确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed)

# 数据集配置
config = {
    'txt_len': 209,  # 语音pca维度
    'cog_len': 80,  # 认知量表特征数量
    'batch_size_val': 24,
    'num_workers': 2,
    'train_datalist': ['RMT_MRI-train'],
    'test_datalist': ['RMT_MRI-test'],  # 测试集
    'checkpoint_dir': 'checkpoints',
    'scaler_path': os.path.join('checkpoints', 'scalers.pkl'),  # 由 train.py 生成
    'teacher_model_name': 'teacher_model',
    'student_model_name': 'student_distilled',
    'single_modality_model_name': 'single_modality_model',
    'use_position_embedding': True,
    'use_cross_attention': False,
}

def load_scalers():
    """加载由 train.py 生成并保存的标准化器"""
    if joblib is None:
        raise ImportError(f"未找到 joblib，无法加载 scaler。请先安装：pip install joblib。原始错误: {_joblib_import_error}")
    if not os.path.exists(config['scaler_path']):
        raise FileNotFoundError(
            f"未找到标准化器文件: {config['scaler_path']}。"
            f"请先运行 train.py 生成该文件。"
        )
    payload = joblib.load(config['scaler_path'])
    if not isinstance(payload, dict) or 'scaler' not in payload or 'cog_scaler' not in payload:
        raise ValueError(f"标准化器文件格式不正确: {config['scaler_path']}")
    return payload['scaler'], payload['cog_scaler']

# 加载模型
def load_model(model_path, modalities, training_mode):
    """加载模型"""
    model = MedBLIPModel(
        max_txt_len=config['txt_len'], 
        num_numerical_features=config['txt_len'],
        num_cog_features=config['cog_len'],
        freeze_teacher=False,
        kd_temp=1.0,
        kd_weight=0.0,
        student_cls_weight=0.0,
        use_positional_embedding=config['use_position_embedding'],
        use_cross_attention=config['use_cross_attention'],
        modalities=modalities,
        training_mode=training_mode
    )
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None
    
    model.to(device)
    model.eval()
    return model

# 测试模型
def test_model(model, testloader, model_name, use_student_prob=False):
    """测试模型并返回评估指标"""
    print(f"\n测试 {model_name}...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in testloader:
            if len(batch) == 5:
                # 新格式：image, features, cog_features, label, id
                image, features, cog_features, label, batch_id = batch
            else:
                # 旧格式：image, text, label, id
                image, features, label, batch_id = batch
                cog_features = None
            
            image = image.to(device)
            features = features.to(device)
            if cog_features is not None:
                cog_features = cog_features.to(device)
            label = label.to(device)
            
            # 模型预测
            if cog_features is not None:
                # 使用predict方法，传入完整的样本元组
                y_hat_prob, y_hat_speech_prob, _, _ = model.predict((image, features, cog_features, label, batch_id))
                # 根据模型类型选择正确的预测结果
                outputs = y_hat_speech_prob if use_student_prob else y_hat_prob
            else:
                # 使用predict方法，传入完整的样本元组
                y_hat_prob, y_hat_speech_prob, _, _ = model.predict((image, features, label, batch_id))
                # 根据模型类型选择正确的预测结果
                outputs = y_hat_speech_prob if use_student_prob else y_hat_prob
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 计算AUC（与训练评估保持一致）
    try:
        probs = np.asarray(all_probs)
        unique_classes = np.unique(all_labels)
        if probs.ndim == 1:
            auc = roc_auc_score(all_labels, probs)
        elif probs.ndim == 2 and probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, probs[:, 1])
        elif probs.ndim == 2 and probs.shape[1] > 2:
            auc = roc_auc_score(all_labels, probs, multi_class='ovr', average='macro')
        else:
            raise ValueError(f"不支持的概率形状: {probs.shape}, 类别: {unique_classes}")
    except Exception as e:
        print(f"计算AUC时出错: {e}")
        auc = 0.0
    
    print(f"{model_name} 测试结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)
    print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }

# 主测试函数
def main():
    """主测试流程"""
    try:
        # 加载训练阶段保存的标准化器
        print("加载训练阶段保存的标准化器...")
        scaler, cog_scaler = load_scalers()

        # 创建测试集数据加载器（复用训练同款的数据构造逻辑 + 训练scaler）
        print("创建测试集数据加载器...")
        testloader = get_dataloader(
            datalist=config['test_datalist'],
            batch_size=config['batch_size_val'],
            txt_len=config['txt_len'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False,
            val_split_ratio=None,
            scaler=scaler,
            cog_scaler=cog_scaler,
        )
        print(f"测试集批次数量: {len(testloader)}")
        
        # 模型路径
        models_to_test = [
            {
                'name': '教师模型',
                'path': os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_best.pth'),
                'modalities': 'speech_cog',
                'training_mode': 'teacher'
            },
            {
                'name': '学生模型',
                'path': os.path.join(config['checkpoint_dir'], f'{config["student_model_name"]}_best.pth'),
                'modalities': 'speech_cog',
                'training_mode': 'student'
            },
            {
                'name': '单语音模型',
                'path': os.path.join(config['checkpoint_dir'], f'{config["single_modality_model_name"]}_text_best.pth'),
                'modalities': 'text',
                'training_mode': 'single'
            },
            {
                'name': '单认知量表模型',
                'path': os.path.join(config['checkpoint_dir'], f'{config["single_modality_model_name"]}_cog_best.pth'),
                'modalities': 'cog',
                'training_mode': 'single'
            }
        ]
        
        # 测试所有模型
        results = {}
        for model_info in models_to_test:
            model = load_model(model_info['path'], model_info['modalities'], model_info['training_mode'])
            if model:
                # 根据模型类型选择正确的预测结果
                use_student_prob = model_info['training_mode'] == 'student'
                results[model_info['name']] = test_model(model, testloader, model_info['name'], use_student_prob)
            else:
                print(f"⚠️  跳过测试 {model_info['name']}，模型加载失败")
        
        # 汇总结果
        print("\n" + "="*60)
        print("            所有模型测试结果汇总            ")
        print("="*60)
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print()
        
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理资源
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()