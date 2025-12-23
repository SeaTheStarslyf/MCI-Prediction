import os
# 指定GPU - 在代码开头设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第0个GPU

import random
import numpy as np
import torch
from medblip.modeling_medblip import MedBLIPModel, load_pretrained_vision_encoder
# from dataset.dataset import get_dataloader
from dataset.dataset_RMI import get_dataloader
from medblip.trainer import Trainer
from medblip.utils import TrainingVisualizer
from sklearn.utils import class_weight

# 添加类别平衡权重计算函数
def calculate_class_weights(trainloader):
    """计算训练集中各类别的权重，用于平衡类别不平衡问题"""
    labels = []
    # 遍历训练集收集所有标签
    for batch in trainloader:
        _, _, label, _ = batch
        labels.extend(label.cpu().numpy())
    
    # 计算类别权重
    class_weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    
    # 转换为tensor并移至设备
    class_weights_tensor = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights_tensor}")
    return class_weights_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# only pretrain on ADNI train data and NACC train data
train_datalist = [
    # 'ADNI-train-2',
    # 'NACC-train',
    'RMT_MRI-train'
]

val_datalist = [
    # 'ADNI-test-2',
    # 'demo',
    # 'ADNI-test',
    # 'ADNI-train',
    # 'NACC-test',
    # 'AIBL',
    # 'OASIS2',
    # 'MIRIAD',
    'RMT_MRI-test'
]

txt_len = 209 #语音pca dim

trainloader = get_dataloader(train_datalist, batch_size=4, txt_len=txt_len, shuffle=True,num_workers=2, drop_last=True)
valloader = get_dataloader(val_datalist, batch_size=24, txt_len=txt_len, shuffle=False,num_workers=2, drop_last=False)

# ===== 分步训练策略配置 =====
# 训练阶段选择: 1=训练教师模型, 2=蒸馏训练学生模型, 3=两步训练都执行
TRAINING_PHASE = 1

# 教师模型训练配置
TEACHER_MODEL_NAME = 'teacher_model'
TEACHER_EPOCHS = 5  # 教师模型训练轮数
TEACHER_LEARNING_RATE = 2e-5

# 学生模型蒸馏配置
STUDENT_MODEL_NAME = 'student_distilled'
STUDENT_EPOCHS = 5  # 蒸馏训练轮数
STUDENT_LEARNING_RATE = 2e-5  # 学生模型可以使用相同或略小的学习率
KD_WEIGHT = 0.7      # 蒸馏损失权重 (增大以更关注教师模型的软目标)
STUDENT_CLS_WEIGHT = 0.3  # 学生模型分类权重 (减小以平衡软目标学习)
TEMPERATURE = 3.0    # 蒸馏温度 (提高温度以产生更平滑的软目标)
FEAT_DISTILL_WEIGHT = 0.5  # 特征级蒸馏权重 (控制特征对齐的重要性)

# 检查点保存路径
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 教师模型保存路径
TEACHER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{TEACHER_MODEL_NAME}_best.pth')

def train_teacher_model():
    """训练多模态教师模型"""
    print("\n" + "="*50)
    print("          开始训练多模态教师模型          ")
    print("="*50)
    
    # 计算类别权重
    print("计算训练集类别权重...")
    class_weights = calculate_class_weights(trainloader)
    
    # 创建模型 - 完全专注于教师模型训练，不使用蒸馏或学生模型
    model = MedBLIPModel(max_txt_len=txt_len, 
                        num_numerical_features=txt_len,
                        freeze_teacher=False,  # 不冻结教师模型
                        kd_temp=1.0,          # 蒸馏温度
                        kd_weight=0.0,         # 蒸馏权重为0
                        student_cls_weight=0.0,
                        disable_student=True)  # 禁用学生模型
    
    # 设置平衡交叉熵损失
    model.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    model.to(device)
    # 为教师模型训练创建专用的可视化器，指定不同的保存目录
    trainer = Trainer(phase_name="teacher")
    # 修改默认的可视化目录，避免被第二阶段覆盖
    trainer.visualizer = TrainingVisualizer(save_dir="training_plots_teacher", phase_name="teacher")
    
    # 打印训练配置
    print("\n===== 教师模型训练配置 =====")
    print(f"冻结教师模型: {model.freeze_teacher}")
    print(f"使用蒸馏: False (教师模型模式)")
    print(f"训练轮数: {TEACHER_EPOCHS}")
    print(f"学习率: {TEACHER_LEARNING_RATE}")
    print(f"权重衰减: 1e-4")
    print(f"批次大小: {trainloader.batch_size}")
    print(f"类别权重: {class_weights}")
    print("训练目标: 多模态特征融合 + 分类任务")
    print("=====================\n")
    
    # 训练教师模型 - 只优化教师模型相关参数
    trainer.train(
        model,
        trainloader,
        valloader,
        warmup_ratio=0.1,
        epochs=TEACHER_EPOCHS,
        optimizer_params={'lr': TEACHER_LEARNING_RATE},
        output_path=os.path.join(CHECKPOINT_DIR, TEACHER_MODEL_NAME),
        metric_path=os.path.join(CHECKPOINT_DIR, f'{TEACHER_MODEL_NAME}_metrics.txt'),
        weight_decay=1e-4,
    )
    
    # 保存完整教师模型
    torch.save(model.state_dict(), TEACHER_MODEL_PATH)
    print(f"\n✅ 教师模型已保存到: {TEACHER_MODEL_PATH}")
    print("教师模型训练阶段完成！")
    
    return model

def train_student_model(teacher_model=None):
    """固定教师模型，蒸馏训练单模态学生模型"""
    print("\n" + "="*50)
    print("         开始蒸馏训练单模态学生模型         ")
    print("="*50)
    
    # 创建模型 - 启用蒸馏，冻结教师模型
    model = MedBLIPModel(max_txt_len=txt_len, 
                        num_numerical_features=txt_len,
                        freeze_teacher=True,  # 强制冻结教师模型
                        kd_temp=TEMPERATURE,  # 蒸馏温度
                        kd_weight=KD_WEIGHT,   # 蒸馏损失权重
                        student_cls_weight=STUDENT_CLS_WEIGHT,
                        feat_distill_weight=FEAT_DISTILL_WEIGHT) # 特征级蒸馏权重
    
    # 设置平衡交叉熵损失
    class_weights = calculate_class_weights(trainloader)
    model.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 加载训练好的教师模型权重
    teacher_loaded = False
    if teacher_model is not None:
        # 从内存加载教师模型权重
        model.load_state_dict(teacher_model.state_dict())
        print("✅ 从内存加载教师模型权重")
        teacher_loaded = True
    elif os.path.exists(TEACHER_MODEL_PATH):
        # 从文件加载教师模型权重
        try:
            model.load_state_dict(torch.load(TEACHER_MODEL_PATH))
            print(f"✅ 从文件加载教师模型权重: {TEACHER_MODEL_PATH}")
            teacher_loaded = True
        except Exception as e:
            print(f"❌ 加载教师模型失败: {e}")
    else:
        print(f"❌ 未找到训练好的教师模型: {TEACHER_MODEL_PATH}")
    
    # 强制冻结教师模型参数 - 确保所有教师相关参数不可训练
    model._freeze_teacher_parameters()
    model.to(device)
    # 为学生模型训练创建专用的可视化器，指定不同的保存目录
    trainer = Trainer(phase_name="student")
    # 修改默认的可视化目录，避免与第一阶段冲突
    trainer.visualizer = TrainingVisualizer(save_dir="training_plots_student", phase_name="student")
    
    # 打印训练配置
    print("\n===== 学生模型蒸馏配置 =====")
    print(f"冻结教师模型: {model.freeze_teacher}")
    print(f"使用蒸馏: True (学生模型模式)")
    print(f"蒸馏权重: {model.kd_weight}")
    print(f"学生分类权重: {model.student_cls_weight}")
    print(f"蒸馏温度: {model.distillation_temp.item()}")
    print(f"训练轮数: {STUDENT_EPOCHS}")
    print(f"学习率: {STUDENT_LEARNING_RATE}")
    print(f"类别权重: {class_weights}")
    print("训练目标: 语音单模态分类 + 教师模型软目标")
    print(f"可训练参数: student_ft_transformer (特征提取), mlp_speech (分类头)")
    print(f"训练方式: 响应式蒸馏模式 - 同时进行logits级和特征级对齐")
    print(f"特征级蒸馏权重: {FEAT_DISTILL_WEIGHT}")
    print("=====================\n")
    
    # 蒸馏训练学生模型
    trainer.train(
        model,
        trainloader,
        valloader,
        warmup_ratio=0.1,
        epochs=STUDENT_EPOCHS,
        optimizer_params={'lr': STUDENT_LEARNING_RATE},
        output_path=os.path.join(CHECKPOINT_DIR, STUDENT_MODEL_NAME),
        metric_path=os.path.join(CHECKPOINT_DIR, f'{STUDENT_MODEL_NAME}_metrics.txt'),
        weight_decay=1e-4,
    )
    
    # 保存训练好的学生模型
    student_model_path = os.path.join(CHECKPOINT_DIR, f'{STUDENT_MODEL_NAME}_best.pth')
    torch.save(model.state_dict(), student_model_path)
    print(f"\n✅ 学生模型已保存到: {student_model_path}")
    print("学生模型蒸馏训练完成！")
    
    return model

# 执行训练流程
if TRAINING_PHASE == 1 or TRAINING_PHASE == 3:
    # 训练教师模型
    teacher_model = train_teacher_model()
else:
    teacher_model = None

if TRAINING_PHASE == 2 or TRAINING_PHASE == 3:
    # 蒸馏训练学生模型
    student_model = train_student_model(teacher_model)

print("\n训练流程全部完成!")