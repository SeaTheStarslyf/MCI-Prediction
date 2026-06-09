import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from medblip.modeling_medblip import MedBLIPModel
from dataset.dataset_RMI import get_dataloader

# 设置设备
def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 诊断对比损失的脚本
class ContrastiveLossDiagnoser:
    def __init__(self, config):
        self.config = config
        self.device = setup_device()
        self.model = None
        self.dataloader = None
    
    def load_model_and_data(self):
        """加载模型和数据"""
        # 创建模型
        self.model = MedBLIPModel(
            max_txt_len=config['txt_len'],
            num_numerical_features=config['txt_len'],
            training_mode='teacher',
            modalities='both'
        )
        self.model.to(self.device)
        
        # 加载数据
        print("加载训练数据集...")
        # 当val_split_ratio=0.0时，直接使用全部数据作为训练集
        trainloader = get_dataloader(
            datalist=config['train_datalist'],
            batch_size=config['batch_size_train'],
            txt_len=config['txt_len'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True,
            val_split_ratio=None  # 不划分验证集
        )
        self.dataloader = trainloader
        print(f"数据加载完成，批次大小: {trainloader.batch_size}")
    
    def analyze_positive_samples(self, num_batches=5):
        """分析正样本分布"""
        print("\n=== 分析正样本分布 ===")
        
        total_positive_pairs = 0
        total_possible_pairs = 0
        positive_ratios = []
        
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= num_batches:
                break
            
            image, text, label, id = batch
            batch_size = len(id)
            
            # 创建正样本掩码
            id_matrix = (id.unsqueeze(1) == id.unsqueeze(0))
            label_matrix = (label.unsqueeze(1) == label.unsqueeze(0))
            pos_mask = (id_matrix & label_matrix)
            
            # 计算正样本数量（排除自身）
            positive_pairs = torch.sum(pos_mask.float()) - batch_size
            possible_pairs = batch_size * (batch_size - 1)
            positive_ratio = positive_pairs / possible_pairs if possible_pairs > 0 else 0
            
            total_positive_pairs += positive_pairs
            total_possible_pairs += possible_pairs
            positive_ratios.append(positive_ratio.item())
            
            print(f"批次 {batch_idx+1}: 正样本对数: {positive_pairs}, 总可能对数: {possible_pairs}, 正样本比例: {positive_ratio:.4f}")
        
        avg_positive_ratio = total_positive_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
        print(f"\n平均正样本比例: {avg_positive_ratio:.4f}")
        print(f"正样本比例范围: {min(positive_ratios):.4f} - {max(positive_ratios):.4f}")
        
        if avg_positive_ratio < 0.01:
            print("⚠️  警告: 正样本比例过低，可能导致对比损失难以下降")
        elif avg_positive_ratio > 0.5:
            print("⚠️  警告: 正样本比例过高，可能导致对比损失过于简单")
        
        return avg_positive_ratio
    
    def analyze_feature_distribution(self, num_batches=5):
        """分析特征分布"""
        print("\n=== 分析特征分布 ===")
        
        all_image_feats = []
        all_text_feats = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= num_batches:
                    break
                
                image, text, label, id = batch
                image = image.unsqueeze(1).to(self.device)
                text = text.to(self.device)
                label = label.to(self.device)
                
                # 提取特征
                features = self.model._extract_features(image, text, augment=False)
                image_feat = features['image_feat']
                text_feat = features['text_feat']
                
                all_image_feats.append(image_feat.cpu().numpy())
                all_text_feats.append(text_feat.cpu().numpy())
                all_labels.append(label.cpu().numpy())
        
        # 合并数据
        all_image_feats = np.concatenate(all_image_feats, axis=0)
        all_text_feats = np.concatenate(all_text_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 分析特征统计信息
        print(f"图像特征形状: {all_image_feats.shape}")
        print(f"文本特征形状: {all_text_feats.shape}")
        
        # 计算特征范数
        image_norms = np.linalg.norm(all_image_feats, axis=1)
        text_norms = np.linalg.norm(all_text_feats, axis=1)
        
        print(f"图像特征范数均值: {np.mean(image_norms):.4f}, 标准差: {np.std(image_norms):.4f}")
        print(f"文本特征范数均值: {np.mean(text_norms):.4f}, 标准差: {np.std(text_norms):.4f}")
        
        # 计算特征相似度矩阵
        sim_matrix = np.dot(all_image_feats, all_text_feats.T)
        avg_similarity = np.mean(sim_matrix)
        std_similarity = np.std(sim_matrix)
        
        print(f"跨模态特征平均相似度: {avg_similarity:.4f}, 标准差: {std_similarity:.4f}")
        
        # 可视化特征分布
        self.visualize_features(all_image_feats, all_text_feats, all_labels)
        
        return avg_similarity
    
    def analyze_temperature_effect(self, num_batches=3):
        """分析温度参数对损失的影响"""
        print("\n=== 分析温度参数影响 ===")
        
        temperatures = [0.01, 0.03, 0.07, 0.1, 0.3, 0.5, 1.0]
        loss_results = {}
        
        with torch.no_grad():
            for temp in temperatures:
                total_loss = 0
                batch_count = 0
                
                for batch_idx, batch in enumerate(self.dataloader):
                    if batch_idx >= num_batches:
                        break
                    
                    image, text, label, id = batch
                    image = image.unsqueeze(1).to(self.device)
                    text = text.to(self.device)
                    label = label.to(self.device)
                    id = id.to(self.device)
                    
                    # 提取特征
                    features = self.model._extract_features(image, text, augment=False)
                    image_feat = features['image_feat']
                    text_feat = features['text_feat']
                    
                    # 计算不同温度下的损失
                    from medblip.utils import itc_loss
                    loss = itc_loss(image_feat, text_feat, id, label, temp=torch.tensor(temp))
                    total_loss += loss.item()
                    batch_count += 1
                
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                loss_results[temp] = avg_loss
                print(f"温度 {temp}: 平均对比损失 = {avg_loss:.4f}")
        
        # 绘制温度-损失曲线
        self.plot_temperature_curve(loss_results)
        
        return loss_results
    
    def analyze_batch_size_effect(self, batch_sizes=[8, 16, 32]):
        """分析批次大小对损失的影响"""
        print("\n=== 分析批次大小影响 ===")
        
        loss_results = {}
        
        for batch_size in batch_sizes:
            # 创建对应批次大小的数据加载器
            # 当不需要验证集时，直接使用全部数据作为训练集
            trainloader = get_dataloader(
                datalist=config['train_datalist'],
                batch_size=batch_size,
                txt_len=config['txt_len'],
                shuffle=True,
                num_workers=config['num_workers'],
                drop_last=True,
                val_split_ratio=None
            )
            
            total_loss = 0
            batch_count = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(trainloader):
                    if batch_idx >= 3:  # 每个批次大小测试3个批次
                        break
                    
                    image, text, label, id = batch
                    image = image.unsqueeze(1).to(self.device)
                    text = text.to(self.device)
                    label = label.to(self.device)
                    id = id.to(self.device)
                    
                    # 提取特征
                    features = self.model._extract_features(image, text, augment=False)
                    image_feat = features['image_feat']
                    text_feat = features['text_feat']
                    
                    # 计算损失
                    from medblip.utils import itc_loss
                    loss = itc_loss(image_feat, text_feat, id, label, temp=self.model.itc_temp)
                    total_loss += loss.item()
                    batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            loss_results[batch_size] = avg_loss
            print(f"批次大小 {batch_size}: 平均对比损失 = {avg_loss:.4f}")
        
        return loss_results
    
    def visualize_features(self, image_feats, text_feats, labels):
        """可视化特征分布"""
        print("\n=== 可视化特征分布 ===")
        
        # 使用t-SNE降维
        combined_feats = np.concatenate([image_feats, text_feats], axis=0)
        labels_combined = np.concatenate([labels, labels], axis=0)
        modality_labels = np.concatenate([np.zeros(len(labels)), np.ones(len(labels))], axis=0)
        
        print("进行t-SNE降维...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(combined_feats)
        
        # 绘制散点图
        plt.figure(figsize=(12, 8))
        
        # 按模态和类别绘制
        for modality in [0, 1]:
            for label in np.unique(labels):
                mask = (modality_labels == modality) & (labels_combined == label)
                color = 'blue' if modality == 0 else 'red'
                marker = 'o' if modality == 0 else 's'
                plt.scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=color,
                    marker=marker,
                    label=f'{"Image" if modality == 0 else "Text"} - Class {label}',
                    alpha=0.6
                )
        
        plt.title('Feature Distribution (t-SNE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存可视化结果
        os.makedirs('diagnosis_results', exist_ok=True)
        plt.savefig('diagnosis_results/feature_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("特征分布可视化已保存到 diagnosis_results/feature_distribution.png")
    
    def plot_temperature_curve(self, loss_results):
        """绘制温度-损失曲线"""
        plt.figure(figsize=(10, 6))
        
        temps = sorted(loss_results.keys())
        losses = [loss_results[temp] for temp in temps]
        
        plt.plot(temps, losses, 'o-', linewidth=2, markersize=8)
        plt.title('Temperature vs Contrastive Loss')
        plt.xlabel('Temperature')
        plt.ylabel('Average Contrastive Loss')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # 保存结果
        os.makedirs('diagnosis_results', exist_ok=True)
        plt.savefig('diagnosis_results/temperature_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("温度-损失曲线已保存到 diagnosis_results/temperature_curve.png")
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("=" * 60)
        print("        对比损失诊断工具")
        print("=" * 60)
        
        # 1. 加载模型和数据
        self.load_model_and_data()
        
        # 2. 分析正样本分布
        self.analyze_positive_samples()
        
        # 3. 分析特征分布
        self.analyze_feature_distribution()
        
        # 4. 分析温度参数影响
        self.analyze_temperature_effect()
        
        # 5. 分析批次大小影响
        self.analyze_batch_size_effect()
        
        print("\n" + "=" * 60)
        print("        诊断完成")
        print("=" * 60)
        print("请查看 diagnosis_results 目录下的可视化结果")

if __name__ == "__main__":
    # 配置参数
    config = {
        'txt_len': 209,  # 语音pca维度
        'batch_size_train': 16,
        'num_workers': 2,
        'train_datalist': ['RMT_MRI-train'],
    }
    
    # 创建诊断器并运行
    diagnoser = ContrastiveLossDiagnoser(config)
    diagnoser.run_full_diagnosis()