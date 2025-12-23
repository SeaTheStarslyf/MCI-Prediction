# Most commonly used
import sys
import os
import json
import pickle
import math
from collections import Counter, defaultdict
from functools import partial
from tqdm import tqdm, trange
# from colors import blue, red, green, cyan

# Numerical computation
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score

# Visualization
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.cluster import KMeans

# Density estimation
# sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/vonmiseskde")
# from vonmiseskde import VonMisesKDE
from sklearn.neighbors import KernelDensity

# Image processing
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Multimodal model
# sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/CLIP")
import clip
# from clip.model import CLIP

def eval_metrics(y_true,y_pred,epoch,metric_path,if_student=False):
    # 确保y_true和y_pred是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # 检查样本数量是否足够进行5折评估
    n_samples = len(y_true)
    if n_samples < 5:
        # 样本数量不足时，直接计算整体指标
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        y_true_one_hot = torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=3).numpy()
        y_pred_one_hot = torch.nn.functional.one_hot(torch.tensor(y_pred), num_classes=3).numpy()
        auc = roc_auc_score(y_true_one_hot.ravel(), y_pred_one_hot.ravel())
        
        print(f'样本数量不足，计算整体指标 - AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}')
        
        with open(metric_path,'a+') as f:
            if if_student:
                line = f'epoch {epoch}, AUC_Student: {auc:.4f}, ACC_Student: {acc:.4f}, F1_Student: {f1:.4f}\n'
            else:
                line = f'epoch {epoch}, AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}\n'
            f.write(line)
            
        return auc, acc
    
    # 5折交叉验证评估
    precisions, recalls, f1s, accs, aucs = [], [], [], [], []
    fold_size = n_samples // 5
    
    for i in range(5):
        bi = i * fold_size
        ei = n_samples if i == 4 else (i+1) * fold_size
        y_true_i, y_pred_i = y_true[bi:ei], y_pred[bi:ei]
        
        # 计算指标
        precision = precision_score(y_true_i, y_pred_i, average='macro', zero_division=0)
        recall = recall_score(y_true_i, y_pred_i, average='macro', zero_division=0)
        f1 = f1_score(y_true_i, y_pred_i, average='macro', zero_division=0)
        acc = accuracy_score(y_true_i, y_pred_i)
        
        # 计算AUC (使用one-hot编码)
        y_true_i_one_hot = torch.nn.functional.one_hot(torch.tensor(y_true_i), num_classes=3).numpy()
        y_pred_i_one_hot = torch.nn.functional.one_hot(torch.tensor(y_pred_i), num_classes=3).numpy()
        auc = roc_auc_score(y_true_i_one_hot.ravel(), y_pred_i_one_hot.ravel())
        
        # 记录各折指标
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accs.append(acc)
        aucs.append(auc)
    
    # 计算平均指标和标准差
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    mean_precision, std_precision = np.mean(precisions), np.std(precisions)
    mean_recall, std_recall = np.mean(recalls), np.std(recalls)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
    
    # 打印评估结果
    print(f'accs mean±std: {mean_acc:.4f}±{std_acc:.4f}')
    print(f'aucs mean±std: {mean_auc:.4f}±{std_auc:.4f}')
    print(f'precisions mean±std: {mean_precision:.4f}±{std_precision:.4f}')
    print(f'recalls mean±std: {mean_recall:.4f}±{std_recall:.4f}')
    print(f'f1s mean±std: {mean_f1:.4f}±{std_f1:.4f}')
    
    # 保存结果到文件
    try:
        with open(metric_path, 'a+') as f:
            if if_student:
                line = f'epoch {epoch}, AUC_Student: {mean_auc:.4f}, ACC_Student: {mean_acc:.4f}, F1_Student: {mean_f1:.4f}\n'
            else:
                line = f'epoch {epoch}, AUC: {mean_auc:.4f}, ACC: {mean_acc:.4f}, F1: {mean_f1:.4f}\n'
            f.write(line)
    except Exception as e:
        print(f'保存评估指标到文件失败: {e}')
    
    return mean_auc, mean_acc

class TrainingVisualizer:
    def __init__(self, save_dir="training_plots", phase_name=None):
        """初始化训练可视化器
        
        Args:
            save_dir: 保存图表的目录路径
            phase_name: 训练阶段名称（如"teacher"或"student"），用于区分不同训练阶段
        """
        self.save_dir = save_dir
        self.phase_name = phase_name or ""
        # 初始化所有训练损失列表
        self.train_losses = []
        self.loss_itc = []
        self.loss_text_res = []
        self.loss_image_res = []
        self.loss_cls = []
        self.loss_cls_tea = []
        self.loss_cls_stu = []
        self.loss_kl = []
        self.loss_local = []
        # 添加学习率曲线
        self.lr_values = []
        # 初始化训练集指标列表
        self.train_accs = []
        self.train_aucs = []
        self.train_accs_stu = []
        self.train_aucs_stu = []
        # 初始化验证指标列表
        self.val_accs = []
        self.val_aucs = []
        self.val_accs_stu = []
        self.val_aucs_stu = []
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 创建图表和子图 - 增加学习率曲线图
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 15))
        # 使用非交互式后端，避免显示窗口
        plt.switch_backend('Agg')
    
    def update_metrics(self, epoch, train_loss=None, loss_itc=None, loss_text_res=None, 
                    loss_image_res=None, loss_cls=None, loss_cls_tea=None, loss_cls_stu=None, 
                    loss_kl=None, loss_local=None, val_acc=None, val_auc=None, 
                    val_acc_stu=None, val_auc_stu=None, lr=None, train_acc=None, train_auc=None,
                    train_acc_stu=None, train_auc_stu=None):
        """更新指标并在同一张图上绘制
        
        Args:
            epoch: 当前训练轮数
            train_loss: 总体训练损失
            loss_itc: 图像-文本对比损失
            loss_text_res: 文本重构损失
            loss_image_res: 图像重构损失
            loss_cls: 分类损失
            loss_cls_tea: 教师模型分类损失
            loss_cls_stu: 学生模型分类损失
            loss_kl: KL散度损失（蒸馏损失）
            loss_local: 局部损失
            val_acc: 教师模型验证准确率
            val_auc: 教师模型验证AUC
            val_acc_stu: 学生模型验证准确率
            val_auc_stu: 学生模型验证AUC
            lr: 当前学习率
            train_acc: 教师模型训练准确率
            train_auc: 教师模型训练AUC
            train_acc_stu: 学生模型训练准确率
            train_auc_stu: 学生模型训练AUC
        """
        # 更新损失指标
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
        if loss_itc is not None:
            self.loss_itc.append(float(loss_itc))
        if loss_text_res is not None:
            self.loss_text_res.append(float(loss_text_res))
        if loss_image_res is not None:
            self.loss_image_res.append(float(loss_image_res))
        if loss_cls is not None:
            self.loss_cls.append(float(loss_cls))
        if loss_cls_tea is not None:
            self.loss_cls_tea.append(float(loss_cls_tea))
        if loss_cls_stu is not None:
            self.loss_cls_stu.append(float(loss_cls_stu))
        if loss_kl is not None:
            self.loss_kl.append(float(loss_kl))
        if loss_local is not None:
            self.loss_local.append(float(loss_local))
        # 更新学习率
        if lr is not None:
            self.lr_values.append(float(lr))
        # 更新训练集指标
        if train_acc is not None:
            self.train_accs.append(float(train_acc))
        if train_auc is not None:
            self.train_aucs.append(float(train_auc))
        if train_acc_stu is not None:
            self.train_accs_stu.append(float(train_acc_stu))
        if train_auc_stu is not None:
            self.train_aucs_stu.append(float(train_auc_stu))
        # 更新验证指标
        if val_acc is not None:
            self.val_accs.append(float(val_acc))
        if val_auc is not None:
            self.val_aucs.append(float(val_auc))
        if val_acc_stu is not None:
            self.val_accs_stu.append(float(val_acc_stu))
        if val_auc_stu is not None:
            self.val_aucs_stu.append(float(val_auc_stu))
        
        # 更新图表
        try:
            self._update_plot()
        except Exception as e:
            print(f"更新训练图表时出错: {e}")

    def _update_plot(self):
        """更新图表，显示所有loss曲线和验证指标"""
        # 清除之前的绘图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # 确定训练损失的x轴范围
        if self.train_losses:
            epochs = range(1, len(self.train_losses) + 1)
            
            # 绘制总损失曲线
            self.ax1.plot(epochs, self.train_losses, 'b-', label='Total Loss', linewidth=2, marker='o')
            
            # 根据训练阶段动态显示相关损失曲线
            # 只显示有数据的损失曲线，避免图表过于混乱
            if self.loss_itc and any(loss is not None and not math.isnan(loss) for loss in self.loss_itc):
                self.ax1.plot(epochs, self.loss_itc, 'r-', label='ITC Loss', linewidth=1, marker='^')
            
            if self.loss_text_res and any(loss is not None and not math.isnan(loss) for loss in self.loss_text_res):
                self.ax1.plot(epochs, self.loss_text_res, 'g-', label='Text Res Loss', linewidth=1, marker='s')
            
            if self.loss_image_res and any(loss is not None and not math.isnan(loss) for loss in self.loss_image_res):
                self.ax1.plot(epochs, self.loss_image_res, 'c-', label='Image Res Loss', linewidth=1, marker='d')
            
            if self.loss_cls and any(loss is not None and not math.isnan(loss) for loss in self.loss_cls):
                self.ax1.plot(epochs, self.loss_cls, 'm-', label='CLS Loss', linewidth=1, marker='v')

            if self.loss_cls_tea and any(loss is not None and not math.isnan(loss) for loss in self.loss_cls_tea):
                self.ax1.plot(epochs, self.loss_cls_tea, 'y-', label='Teacher CLS Loss', linewidth=1, marker='>')

            if self.loss_cls_stu and any(loss is not None and not math.isnan(loss) for loss in self.loss_cls_stu):
                self.ax1.plot(epochs, self.loss_cls_stu, 'k-', label='Student CLS Loss', linewidth=1, marker='<')

            if self.loss_kl and any(loss is not None and not math.isnan(loss) for loss in self.loss_kl):
                self.ax1.plot(epochs, self.loss_kl, 'gray', label='KL Loss', linewidth=1, marker='p')

            if self.loss_local and any(loss is not None and not math.isnan(loss) for loss in self.loss_local):
                self.ax1.plot(epochs, self.loss_local, 'orange', label='Local Loss', linewidth=1, marker='o')
            
            # 设置第一个子图的属性
            title = f'Training Losses'
            if self.phase_name:
                title += f' ({self.phase_name})'
            self.ax1.set_title(title, fontsize=14)
            self.ax1.set_xlabel('Epoch', fontsize=12)
            self.ax1.set_ylabel('Loss', fontsize=12)
            self.ax1.legend(loc='best', fontsize=10)
            self.ax1.grid(True, alpha=0.3)
        
        # 绘制训练和验证指标
        metrics_present = bool(self.train_accs or self.train_aucs or self.train_accs_stu or self.train_aucs_stu or 
                             self.val_accs or self.val_aucs or self.val_accs_stu or self.val_aucs_stu)
        
        if metrics_present:
            # 确定最大的epoch数
            max_epoch = 0
            for metric_list in [self.train_accs, self.train_aucs, self.train_accs_stu, self.train_aucs_stu,
                               self.val_accs, self.val_aucs, self.val_accs_stu, self.val_aucs_stu]:
                if metric_list:
                    max_epoch = max(max_epoch, len(metric_list))
                    
            epochs_metrics = range(1, max_epoch + 1)
            
            # 只显示有数据的指标曲线
            # 绘制训练集指标（带星号的线表示训练集）
            if self.train_accs and any(acc is not None and not math.isnan(acc) for acc in self.train_accs):
                self.ax2.plot(epochs_metrics[:len(self.train_accs)], self.train_accs, 'r:', label='Teacher Train Acc', linewidth=1, marker='*')
            if self.train_aucs and any(auc is not None and not math.isnan(auc) for auc in self.train_aucs):
                self.ax2.plot(epochs_metrics[:len(self.train_aucs)], self.train_aucs, 'g:', label='Teacher Train AUC', linewidth=1, marker='*')
            if self.train_accs_stu and any(acc is not None and not math.isnan(acc) for acc in self.train_accs_stu):
                self.ax2.plot(epochs_metrics[:len(self.train_accs_stu)], self.train_accs_stu, 'b:', label='Student Train Acc', linewidth=1, marker='*')
            if self.train_aucs_stu and any(auc is not None and not math.isnan(auc) for auc in self.train_aucs_stu):
                self.ax2.plot(epochs_metrics[:len(self.train_aucs_stu)], self.train_aucs_stu, 'c:', label='Student Train AUC', linewidth=1, marker='*')
            
            # 绘制验证集指标（实线表示验证集）
            if self.val_accs and any(acc is not None and not math.isnan(acc) for acc in self.val_accs):
                self.ax2.plot(epochs_metrics[:len(self.val_accs)], self.val_accs, 'r-', label='Teacher Val Acc', linewidth=2, marker='o')
            if self.val_aucs and any(auc is not None and not math.isnan(auc) for auc in self.val_aucs):
                self.ax2.plot(epochs_metrics[:len(self.val_aucs)], self.val_aucs, 'g-', label='Teacher Val AUC', linewidth=2, marker='s')
            if self.val_accs_stu and any(acc is not None and not math.isnan(acc) for acc in self.val_accs_stu):
                self.ax2.plot(epochs_metrics[:len(self.val_accs_stu)], self.val_accs_stu, 'b-', label='Student Val Acc', linewidth=2, marker='*')
            if self.val_aucs_stu and any(auc is not None and not math.isnan(auc) for auc in self.val_aucs_stu):
                self.ax2.plot(epochs_metrics[:len(self.val_aucs_stu)], self.val_aucs_stu, 'c-', label='Student Val AUC', linewidth=2, marker='+')
            
            # 设置第二个子图的属性
            title = f'Training and Validation Metrics'
            if self.phase_name:
                title += f' ({self.phase_name})'
            self.ax2.set_title(title, fontsize=14)
            self.ax2.set_xlabel('Epoch', fontsize=12)
            self.ax2.set_ylabel('Score', fontsize=12)
            self.ax2.legend(loc='best', fontsize=10)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(0, 1.0)  # 确保y轴范围在0-1之间
        
        # 绘制学习率曲线（如果有数据）
        if self.lr_values and any(lr is not None and not math.isnan(lr) for lr in self.lr_values):
            lr_epochs = range(1, len(self.lr_values) + 1)
            self.ax3.plot(lr_epochs, self.lr_values, 'purple', label='Learning Rate', linewidth=2, marker='x')
            
            # 设置第三个子图的属性
            title = f'Learning Rate Schedule'
            if self.phase_name:
                title += f' ({self.phase_name})'
            self.ax3.set_title(title, fontsize=14)
            self.ax3.set_xlabel('Epoch', fontsize=12)
            self.ax3.set_ylabel('Learning Rate', fontsize=12)
            self.ax3.legend(loc='best', fontsize=10)
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_yscale('log')  # 使用对数刻度更直观地显示学习率变化
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        self._save_plot()
    
    def _save_plot(self):
        """保存训练进度图表到文件，并同时保存每个epoch的单独图表"""
        try:
            # 生成文件名前缀
            prefix = ""
            if self.phase_name:
                prefix = f"{self.phase_name}_"
            
            # 保存总体训练进度图表
            plot_path = os.path.join(self.save_dir, f'{prefix}training_progress.png')
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            # 同时保存最新epoch的图表副本，方便查看历史记录
            if self.train_losses:
                epoch = len(self.train_losses)
                epoch_plot_path = os.path.join(self.save_dir, f'{prefix}training_progress_epoch_{epoch}.png')
                self.fig.savefig(epoch_plot_path, dpi=150, bbox_inches='tight')
            
            print(f"训练曲线已更新: {plot_path} (共{len(self.train_losses)}个epoch)")
        except Exception as e:
            print(f"保存训练图表失败: {e}")
    
    def close(self):
        """清理资源，关闭图表"""
        try:
            plt.close(self.fig)
        except:
            pass

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(model_path=None):
    device = get_device()
    if model_path is None:
        print("Loading original model...")
        model, _ = clip.load("ViT-B/16", device=device)
        model.float()
    else:
        print(f"Loading model from {model_path}...")
        model = CLIP(
            embed_dim=512,
            image_resolution=224,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        ).to(device)
        if model_path != "random":
            model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Temperature: {model.logit_scale.exp()}")
    return model


def encode_clip(model, dataset, batch_size=32):
    device = get_device()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
    )

    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_inputs, text_inputs = batch
            image_inputs, text_inputs = image_inputs.to(device), text_inputs.to(device)

            image_features = model.encode_image(image_inputs).cpu()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)

            text_features = model.encode_text(text_inputs).cpu()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features)

        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

    return all_image_features, all_text_features

def itc_loss(image_feat, text_feat, ids, labels, temp=0.07):
    """
    向量化多正样本实现
    """
    # 计算相似度
    sim_i2t = image_feat @ text_feat.T / temp
    sim_t2i = text_feat @ image_feat.T / temp
    
    batch_size = len(ids)
    
    # 创建正样本掩码
    if isinstance(ids, torch.Tensor):
        id_matrix = (ids.detach().clone().unsqueeze(1) == ids.detach().clone().unsqueeze(0))
    else:
        id_matrix = (torch.tensor(ids).unsqueeze(1) == torch.tensor(ids).unsqueeze(0))
        
    if isinstance(labels, torch.Tensor):
        label_matrix = (labels.detach().clone().unsqueeze(1) == labels.detach().clone().unsqueeze(0))
    else:
        label_matrix = (torch.tensor(labels).unsqueeze(1) == torch.tensor(labels).unsqueeze(0))
    pos_mask = (id_matrix & label_matrix).to(image_feat.device)
    
    # 向量化计算损失
    loss_i2t = compute_multi_positive_loss(sim_i2t, pos_mask, temp)
    loss_t2i = compute_multi_positive_loss(sim_t2i, pos_mask.T, temp)
    
    return (loss_i2t + loss_t2i) / 2


def compute_multi_positive_loss(similarity, pos_mask, temp):
    """向量化多正样本损失计算"""
    similarity = similarity / temp
    exp_sim = torch.exp(similarity)
    
    # 计算每个样本的正样本和
    pos_sum = torch.sum(exp_sim * pos_mask.float(), dim=1)
    # 计算每个样本的总和
    all_sum = torch.sum(exp_sim, dim=1)
    
    # 计算损失
    losses = -torch.log(pos_sum / all_sum)
    # 过滤掉没有正样本的情况
    valid_mask = (pos_sum > 0)
    
    return torch.mean(losses[valid_mask]) if torch.any(valid_mask) else torch.tensor(0.0)


def distillation_loss(y_hat_fusion, y_hat_speech, T=2.0):
    """
    y_hat_fusion: 教师模型 logits (多模态)
    y_hat_speech: 学生模型 logits (语音)
    T: 温度参数，默认2.0，可通过模型配置调整
    """
    # 教师概率分布
    p_teacher = F.softmax(y_hat_fusion / T, dim=1)
    # 学生对数概率分布
    log_p_student = F.log_softmax(y_hat_speech / T, dim=1)

    # KL散度 (student模仿teacher)
    loss_kd = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T * T)
    return loss_kd


def align_loss_(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss_(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def ce_loss(model, image_features, text_features):
    loss_func = torch.nn.CrossEntropyLoss()

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = image_features.size(0)
    device = get_device()
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

    loss = (
        loss_func(logits_per_image, ground_truth)
        + loss_func(logits_per_text, ground_truth)
    ) / 2
    return loss


def uniform_loss(model, image_features, text_features):
    loss = (uniform_loss_(image_features) + uniform_loss_(text_features)) / 2
    return loss


def dual_ce_loss(model, image_features, text_features):
    loss_func = torch.nn.CrossEntropyLoss()

    features = torch.cat([image_features, text_features], 0)
    sims = features @ features.t()

    logit_scale = model.logit_scale.exp()
    logits = sims * logit_scale

    batch_size = image_features.size(0)
    logits_per_image = logits[:batch_size, :].contiguous()
    logits_per_image[torch.arange(batch_size), torch.arange(batch_size)] -= 10000
    logits_per_text = logits[batch_size:, :].contiguous()
    logits_per_text[
        torch.arange(batch_size), torch.arange(batch_size) + batch_size
    ] -= 10000

    device = get_device()
    image_ground_truth = (
        torch.arange(batch_size, dtype=torch.long, device=device) + batch_size
    )
    text_ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

    loss = (
        loss_func(logits_per_image, image_ground_truth)
        + loss_func(logits_per_text, text_ground_truth)
    ) / 2
    return loss


def simple_ce_loss(model, image_features, text_features):
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    preds_per_image = torch.argmax(logits_per_image, dim=1)
    preds_per_text = torch.argmax(logits_per_text, dim=1)

    batch_size = image_features.size(0)
    device = get_device()
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

    correct_per_image = (preds_per_image == ground_truth).float()
    correct_per_text = (preds_per_text == ground_truth).float()

    loss_img = (loss_func(logits_per_image, ground_truth) * correct_per_image).sum() / (
        correct_per_image.sum() + 1e-6
    )
    loss_text = (loss_func(logits_per_text, ground_truth) * correct_per_text).sum() / (
        correct_per_text.sum() + 1e-6
    )

    loss = (loss_img + loss_text) / 2
    return loss


def train_clip_toy_fix_init(
    model,
    dataset,
    model_path,
    batch_size=32,
    start_epoch=0,
    end_epoch=10,
    loss_funcs=[ce_loss],
):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    device = get_device()

    if start_epoch == 0:
        print("Training original model...")
        torch.save(model.state_dict(), f"{model_path}/model_epoch_{start_epoch}.pt")
    else:
        print(f"Loading model from {model_path} and continue training...")
        assert os.path.exists(f"{model_path}/model_epoch_{start_epoch}.pt")
        model.load_state_dict(torch.load(f"{model_path}/model_epoch_{start_epoch}.pt"))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    all_image_features, all_text_features = encode_clip(model, dataset)
    yx = all_image_features.t() @ all_text_features
    u, s, v = torch.svd(yx)
    w = u @ v.T
    torch.save([w, all_image_features, all_text_features], f"{model_path}/w.pt")
    all_text_features_transform = all_text_features @ w.T
    w = w.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    logs = {}
    for epoch in range(start_epoch + 1, end_epoch + 1):
        logs[epoch] = []
        bar = tqdm(dataloader)
        for i, batch in enumerate(bar):
            image_inputs, text_inputs = batch
            image_inputs, text_inputs = image_inputs.to(device), text_inputs.to(device)

            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features @ w.T

            losses = [
                loss_func(model, image_features, text_features)
                for loss_func in loss_funcs
            ]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs[epoch].append(
                {"loss": loss.item(), "losses": [loss.item() for loss in losses]}
            )
            bar.set_description(f"Epoch {epoch}/{end_epoch}, Loss: {logs[epoch][i]}")

        torch.save(model.state_dict(), f"{model_path}/model_epoch_{epoch}.pt")

        epoch_loss = np.mean([item["loss"] for item in logs[epoch]])
        epoch_losses = [
            np.mean([item["losses"][i] for item in logs[epoch]])
            for i in range(len(loss_funcs))
        ]
        print(f"Epoch {epoch}: loss = {epoch_loss:.4f}, losses = {epoch_losses}")
    return model, logs


def train_clip_toy(
    model,
    dataset,
    model_path,
    batch_size=32,
    start_epoch=0,
    end_epoch=10,
    loss_funcs=[ce_loss],
):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    device = get_device()

    if start_epoch == 0:
        print("Training original model...")
        torch.save(model.state_dict(), f"{model_path}/model_epoch_{start_epoch}.pt")
    else:
        print(f"Loading model from {model_path} and continue training...")
        assert os.path.exists(f"{model_path}/model_epoch_{start_epoch}.pt")
        model.load_state_dict(torch.load(f"{model_path}/model_epoch_{start_epoch}.pt"))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    logs = {}
    for epoch in range(start_epoch + 1, end_epoch + 1):
        logs[epoch] = []
        bar = tqdm(dataloader)
        for i, batch in enumerate(bar):
            image_inputs, text_inputs = batch
            image_inputs, text_inputs = image_inputs.to(device), text_inputs.to(device)

            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            losses = [
                loss_func(model, image_features, text_features)
                for loss_func in loss_funcs
            ]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs[epoch].append(
                {"loss": loss.item(), "losses": [loss.item() for loss in losses]}
            )
            bar.set_description(f"Epoch {epoch}/{end_epoch}, Loss: {logs[epoch][i]}")

        torch.save(model.state_dict(), f"{model_path}/model_epoch_{epoch}.pt")

        epoch_loss = np.mean([item["loss"] for item in logs[epoch]])
        epoch_losses = [
            np.mean([item["losses"][i] for item in logs[epoch]])
            for i in range(len(loss_funcs))
        ]
        print(f"Epoch {epoch}: loss = {epoch_loss:.4f}, losses = {epoch_losses}")
    return model, logs


def encode_clip_classification(
    model, dataset, prompt="a photo of a {}.", batch_size=32
):
    device = get_device()

    text_inputs = torch.cat(
        [clip.tokenize(prompt.format(c)) for c in dataset.data.classes]
    ).to(device)
    with torch.no_grad():
        all_text_features = model.encode_text(text_inputs).cpu()
        all_text_features /= all_text_features.norm(dim=-1, keepdim=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
    )

    all_image_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_inputs, labels = batch
            image_inputs = image_inputs.to(device)

            image_features = model.encode_image(image_inputs).cpu()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)

        all_image_features = torch.cat(all_image_features, dim=0)

    return all_image_features, all_text_features


def svd(X, n_components=2, return_singular_values=False):
    U, S, Vt = np.linalg.svd(X)
    X_reduce = U[:, :n_components] * S[:n_components]
    if return_singular_values:
        return X_reduce, S
    return X_reduce


def visualize_2d(clusters, colors=None, labels=None, connection=False):
    assert isinstance(clusters, list)
    for cluster in clusters:
        assert isinstance(cluster, np.ndarray)
        assert cluster.shape[1] == 2

    fig = plt.figure(figsize=(5, 5))
    if colors is None:
        colors = ["r" for i in range(len(clusters))]
    if labels is None:
        labels = [f"cluster_{i}" for i in range(len(clusters))]
    for cluster, color, label in zip(clusters, colors, labels):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color, label=label, alpha=0.9)

    if connection:
        # assert len(clusters) == 2 and len(clusters[0]) == len(clusters[1])
        for i in range(len(clusters[0])):
            plt.plot(
                [clusters[0][i, 0], clusters[1][i, 0]],
                [clusters[0][i, 1], clusters[1][i, 1]],
                c="k",
                alpha=0.1,
            )
        for i in range(len(clusters[2])):
            plt.plot(
                [clusters[2][i, 0], clusters[3][i, 0]],
                [clusters[2][i, 1], clusters[3][i, 1]],
                c="k",
                alpha=0.1,
            )
    
    plt.savefig('./vis.png')

def visualize_2d_multi(clusters, colors=None, labels=None, connection=False,method="umap"):
    assert isinstance(clusters, list)
    for cluster in clusters:
        assert isinstance(cluster, np.ndarray)
        assert cluster.shape[1] == 2

    def my_norm(x):
        return x/np.linalg.norm(x, axis=-1, keepdims=True)

    # clusters[0] = my_norm(clusters[0])
    # clusters[1] = my_norm(clusters[1])
    # clusters[2] = my_norm(clusters[2])
    # clusters[3] = my_norm(clusters[3])

    fig = plt.figure(figsize=(5,5),dpi=500)
    plt.style.use('seaborn-v0_8')

    if labels is None:
        labels = [f"cluster_{i}" for i in range(len(clusters))]
    for cluster, color, label in zip(clusters, colors, labels):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color, label=label, alpha=0.6)

    if connection:
        for i in range(len(clusters[0])):
            plt.plot(
                [clusters[0][i, 0], clusters[2][i, 0]],
                [clusters[0][i, 1], clusters[2][i, 1]],
                c="k",
                alpha=0.1,
            )
        for i in range(len(clusters[1])):
            plt.plot(
                [clusters[1][i, 0], clusters[3][i, 0]],
                [clusters[1][i, 1], clusters[3][i, 1]],
                c="k",
                alpha=0.1,
            )
    
    plt.savefig('./vis_{}.png'.format(method))
    plt.close()
    
    
    modality_gap1 = clusters[0].mean(axis=0) - clusters[2].mean(axis=0)
    modality_gap2 = clusters[1].mean(axis=0) - clusters[3].mean(axis=0)
    delta1 = np.linalg.norm(modality_gap1)
    delta2 = np.linalg.norm(modality_gap2)
    mean_delta = (delta1+delta2)/2
    print('method: {}, gap1: {}, delta1: {}, gap2: {}, delta2: {}, mean delta: {}'.format(method, str(modality_gap1),str(delta1),str(modality_gap2),str(delta2),str(mean_delta)))


def visualize_3d(clusters, colors=None, labels=None, connection=False):
    assert isinstance(clusters, list)
    assert connection == False
    for cluster in clusters:
        assert isinstance(cluster, np.ndarray)
        assert cluster.shape[1] == 3

    fig = plt.figure()
    ax = Axes3D(fig)
    if colors is None:
        colors = ["r" for i in range(len(clusters))]
    if labels is None:
        labels = [f"cluster_{i}" for i in range(len(clusters))]
    for cluster, color, label in zip(clusters, colors, labels):
        ax.scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], c=color, label=label, alpha=0.2
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.add_axes(ax)
    plt.show()


def dim_reduce(features, n_dim=2, methods=["svd", "pca", "tsne", "umap"]):
    assert isinstance(features, np.ndarray)

    features_reduce = {}
    for method in methods:
        if method == "svd":
            features_reduce[method] = svd(features, n_dim)
        # if method == "umap":
        #     projector = UMAP(n_neighbors=30,n_components=n_dim)
        #     features_reduce[method] = projector.fit_transform(features)
        else:
            projector = eval(method.upper())(n_components=n_dim)
            features_reduce[method] = projector.fit_transform(features)
    return features_reduce


def reduce_and_visualize(
    image_features,
    text_features,
    n_dim=2,
    methods=["svd", "pca", "tsne", "umap"],
    connection=False,
):
    assert isinstance(image_features, np.ndarray) and isinstance(
        text_features, np.ndarray
    )
    assert n_dim in [2, 3]

    features = np.concatenate([image_features, text_features], axis=0)
    features_reduce = dim_reduce(features, n_dim=n_dim, methods=methods)

    for i, method in enumerate(methods):
        image_features_reduce = features_reduce[method][: len(image_features)]
        text_features_reduce = features_reduce[method][len(image_features) :]
        
        eval(f"visualize_{n_dim}d")(
            [image_features_reduce, text_features_reduce],
            colors=["r", "b"],
            connection=connection,
        )
        # import pdb;pdb.set_trace()

def reduce_and_visualize_multi(
    fea_list,
    n_dim=2,
    methods=["svd", "pca", "tsne", "umap"],
    connection=False,
):

    assert n_dim in [2, 3]

    features = np.concatenate(fea_list, axis=0)
    features_reduce = dim_reduce(features, n_dim=n_dim, methods=methods)

    for i, method in enumerate(methods):
        image_NC_features_reduce = features_reduce[method][: fea_list[0].shape[0]]
        image_AD_features_reduce = features_reduce[method][fea_list[0].shape[0] :fea_list[0].shape[0]+fea_list[1].shape[0]]
        text_NC_features_reduce = features_reduce[method][fea_list[0].shape[0]+fea_list[1].shape[0] :fea_list[0].shape[0]+fea_list[1].shape[0]+fea_list[2].shape[0]]
        text_AD_features_reduce = features_reduce[method][fea_list[0].shape[0]+fea_list[1].shape[0]+fea_list[2].shape[0]:]
        
        eval(f"visualize_{n_dim}d_multi")(
            [image_NC_features_reduce, image_AD_features_reduce,text_NC_features_reduce,text_AD_features_reduce],
            colors=["r", "b", "y", "g"],
            connection=connection,
            method=method,
        )


def convert_image_to_rgb(image):
    return image.convert("RGB")


def estimate_density(image_features, text_features):
    x_plot = np.linspace(-1.2, 1.2, 100)
    y_plot = np.linspace(-1.2, 1.2, 100)
    xy_plot = np.array(np.meshgrid(x_plot, y_plot)).reshape(2, -1).T

    kde_image = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(image_features)
    image_density = np.exp(kde_image.score_samples(xy_plot))

    kde_text = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(text_features)
    text_density = np.exp(kde_text.score_samples(xy_plot))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(
        image_density.reshape(100, 100),
        extent=(-1.2, 1.2, -1.2, 1.2),
        origin="lower",
        cmap="Reds",
        alpha=0.5,
        vmin=min([image_density.min(), text_density.min()]),
        vmax=max([image_density.max(), text_density.max()]),
    )
    plt.scatter(image_features[:, 0], image_features[:, 1], c="red", alpha=0.05)

    plt.subplot(1, 2, 2)
    plt.imshow(
        text_density.reshape(100, 100),
        extent=(-1.2, 1.2, -1.2, 1.2),
        origin="lower",
        cmap="Blues",
        alpha=0.5,
        vmin=min([image_density.min(), text_density.min()]),
        vmax=max([image_density.max(), text_density.max()]),
    )
    plt.scatter(text_features[:, 0], text_features[:, 1], c="blue", alpha=0.05)

    print(
        text_density.min(),
        text_density.max(),
        text_density.mean(),
        image_density.min(),
        image_density.max(),
        image_density.mean(),
    )


def estimate_angle_density(image_features, text_features):
    image_features_angle = [
        np.arctan2(image_features[i, 1], image_features[i, 0]).item()
        for i in range(len(image_features))
    ]
    text_features_angle = [
        np.arctan2(text_features[i, 1], text_features[i, 0]).item()
        for i in range(len(text_features))
    ]

    kappa = 25
    kde_image = VonMisesKDE(image_features_angle, weights=[], kappa=kappa)
    kde_text = VonMisesKDE(text_features_angle, weights=[], kappa=kappa)

    test_x = np.linspace(-math.pi, math.pi, 100)

    # # Display individual distributions
    # for i in np.arange(0, len(text_features_angle)):
    #     sample = text_features_angle[i]
    #     test_y = kde_text.vonMisesPDF(test_x, sample)
    #     test_y = test_y / test_y.sum()
    #     plt.plot(test_x, test_y, color='gray', alpha=0.5)

    # Display posterior estimate
    plt.figure(figsize=(10, 1))

    plt.subplot(1, 2, 1)
    plt.plot(test_x, kde_image.evaluate(test_x), zorder=20, color="red", alpha=0.5)
    plt.fill_between(
        test_x, kde_image.evaluate(test_x), step="pre", alpha=0.2, color="red"
    )
    plt.xlim(-math.pi, math.pi)
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.plot(test_x, kde_text.evaluate(test_x), zorder=20, color="blue", alpha=0.5)
    plt.fill_between(
        test_x, kde_text.evaluate(test_x), step="pre", alpha=0.2, color="blue"
    )
    plt.xlim(-math.pi, math.pi)
    plt.ylim(0, 1)


if __name__ == "__main__":
    ##### Test svd() #####
    X = np.arange(100).reshape(10, 10)
    X_2d = svd(X)
    assert X_2d.shape == (10, 2)