import os
import sys
import torch
import numpy as np
from dataset.dataset_RMI import get_dataloader
from medblip.modeling_medblip import MedBLIPModel
from medblip.trainer import Trainer

# 设备设置
def setup_gpu(gpu_id='0'):
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

# 配置设置
config = {
    'txt_len': 209,  # 语音pca维度
    'cog_len': 80,  # 认知量表特征数量
    'batch_size_train': 4,
    'batch_size_val': 24,
    'num_workers': 2,
    'train_datalist': ['RMT_MRI-train'],  # 训练数据集
    'test_datalist': ['RMT_MRI-test'],  # 测试集
    'val_split_ratio': 0.2,  # 从训练数据中划分验证集的比例
    'checkpoint_dir': 'checkpoints',
    'teacher_model_name': 'teacher_model',
    'modalities': 'cog_speech',  # 训练使用的模态: 'both'(多模态), 'vision'(仅视觉), 'text'(仅文本/语音), 'cog'(仅认知量表), 'speech_cog'(语音+认知量表)
    'use_position_embedding': True,  # 是否使用位置嵌入
    'use_cross_attention': False,      # 是否使用跨模态注意力
    'balance_data': False,  # 是否在数据加载时进行类别平衡
}

# 设备初始化
device = setup_gpu(gpu_id='0')
print(f"Using device: {device}")

# 随机种子设置
seed = 42

def set_random_seed(seed_value):
    """设置随机种子以确保结果可复现"""
    import random
    import numpy as np
    import torch
    import os
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

# 教师模型路径
TEACHER_MODEL_PATH = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_best.pth')

# 检查模型文件是否存在
if not os.path.exists(TEACHER_MODEL_PATH):
    print(f"错误: 未找到教师模型文件: {TEACHER_MODEL_PATH}")
    print("请先训练教师模型，确保模型文件已保存")
    print("检查点目录内容:")
    for file in os.listdir(config['checkpoint_dir']):
        if file.endswith('.pth'):
            print(f"  - {file}")
    sys.exit(1)

print(f"找到教师模型文件: {TEACHER_MODEL_PATH}")

# 创建验证数据加载器
try:
    # 使用与train.py相同的方式加载训练数据并划分验证集
    print("加载训练数据集并划分验证集...")
    _, valloader = get_dataloader(
        datalist=config['train_datalist'],
        batch_size=config['batch_size_val'],
        txt_len=config['txt_len'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        val_split_ratio=config['val_split_ratio'],
        random_state=seed,
        balance_data=config['balance_data']  # 启用数据平衡处理
    )
    print(f"成功创建验证数据加载器，验证集批次: {len(valloader)}")
except Exception as e:
    print(f"创建数据加载器失败: {e}")
    sys.exit(1)

# 创建教师模型
print("创建教师模型...")
try:
    model = MedBLIPModel(
        max_txt_len=config['txt_len'], 
        num_numerical_features=config['txt_len'],
        num_cog_features=config['cog_len'],  # 认知量表特征数量
        freeze_teacher=False,  # 评估时不需要冻结教师模型
        kd_temp=1.0,          # 蒸馏温度
        kd_weight=0.0,         # 蒸馏权重为0
        student_cls_weight=0.0,
        use_positional_embedding=config['use_position_embedding'],
        use_cross_attention=config['use_cross_attention'],
        modalities=config['modalities'],  # 添加模态参数
        training_mode='teacher'  # 教师模型模式
    )
    model.to(device)
    print("教师模型创建成功")
except Exception as e:
    print(f"创建模型失败: {e}")
    sys.exit(1)

# 加载最佳教师模型权重
print(f"加载最佳教师模型权重: {TEACHER_MODEL_PATH}")
try:
    model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device), strict=False)
    print("✅ 教师模型权重加载成功")
except Exception as e:
    print(f"❌ 加载教师模型失败: {e}")
    sys.exit(1)

# 创建评估器
trainer = Trainer(phase_name="teacher", training_mode=1)  # 1表示教师模型模式

# 评估模型性能
print("\n" + "="*60)
print("        开始评估最佳教师模型        ")
print("="*60)

# 评估模型在验证集上的性能
metric_path = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_evaluation_metrics.txt')
print(f"评估结果将保存到: {metric_path}")

# 执行评估
print("\n正在评估模型...")
teacher_auc, teacher_acc, _, _, *rest_metrics = trainer.test(model, valloader, metric_path, -1)  # -1表示评估模式
teacher_f1, teacher_precision, teacher_recall = rest_metrics[-6], rest_metrics[-5], rest_metrics[-4]

# 打印评估结果
print("\n" + "="*60)
print("        教师模型评估结果        ")
print("="*60)
print(f"教师模型 AUC: {teacher_auc:.4f}")
print(f"教师模型 准确率: {teacher_acc:.4f}")
print(f"教师模型 F1: {teacher_f1:.4f}")
print(f"教师模型 Precision: {teacher_precision:.4f}")
print(f"教师模型 Recall: {teacher_recall:.4f}")
print("="*60)

print("\n评估完成！")
print(f"详细评估结果已保存到: {metric_path}")