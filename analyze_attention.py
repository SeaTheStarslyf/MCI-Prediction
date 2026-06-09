import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from medblip.modeling_medblip import MedBLIPModel
from dataset.dataset_RMI import get_dataloader
import argparse

# 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class AttentionAnalyzer:
    def __init__(self, model_path, config):
        """
        初始化注意力分析器
        Args:
            model_path: 模型权重路径
            config: 模型配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        """
        加载模型
        Args:
            model_path: 模型权重路径
        Returns:
            加载好的模型
        """
        model = MedBLIPModel(
            max_txt_len=self.config['txt_len'],
            num_numerical_features=self.config['txt_len'],
            num_cog_features=self.config['cog_len'],
            use_ft_transformer=True,
            use_positional_embedding=self.config['use_position_embedding'],
            use_cross_attention=self.config['use_cross_attention'],
            training_mode='teacher',
            modalities=self.config['modalities']
        )
        
        # 加载模型权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型权重: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            sys.exit(1)
        
        model.to(self.device)
        return model
    
    def get_attention_weights(self, batch):
        """
        获取注意力权重
        Args:
            batch: 输入批次数据
        Returns:
            注意力权重字典
        """
        # 解包批次数据
        if len(batch) == 5:
            image, text, cog, label, id = batch
        else:
            image, text, label, id = batch
            cog = torch.zeros_like(text)
        
        # 移动到设备
        image = image.unsqueeze(1).to(self.device)
        text = text.to(self.device)
        cog = cog.to(self.device)
        
        # 提取特征
        features = self.model._extract_features(image, text, cog, augment=False)
        
        # 存储注意力权重
        attention_weights = {}
        
        # 1. 获取FT-Transformer中的自注意力权重
        if hasattr(self.model, 'text_restorer') and hasattr(self.model.text_restorer, 'ft_transformer'):
            ft_transformer = self.model.text_restorer.ft_transformer
            if hasattr(ft_transformer.encoder, 'layers'):
                for i, layer in enumerate(ft_transformer.encoder.layers):
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn_probs') and layer.self_attn.attn_probs is not None:
                        attention_weights[f'text_self_attn_layer_{i}'] = layer.self_attn.attn_probs.cpu().numpy()
                    if hasattr(layer, 'cross_attn') and hasattr(layer.cross_attn, 'attn_probs') and layer.cross_attn.attn_probs is not None:
                        attention_weights[f'text_cross_attn_layer_{i}'] = layer.cross_attn.attn_probs.cpu().numpy()
        
        # 2. 获取认知量表特征的注意力权重
        if hasattr(self.model, 'cog_restorer') and hasattr(self.model.cog_restorer, 'ft_transformer'):
            cog_transformer = self.model.cog_restorer.ft_transformer
            if hasattr(cog_transformer.encoder, 'layers'):
                for i, layer in enumerate(cog_transformer.encoder.layers):
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn_probs') and layer.self_attn.attn_probs is not None:
                        attention_weights[f'cog_self_attn_layer_{i}'] = layer.self_attn.attn_probs.cpu().numpy()
                    if hasattr(layer, 'cross_attn') and hasattr(layer.cross_attn, 'attn_probs') and layer.cross_attn.attn_probs is not None:
                        attention_weights[f'cog_cross_attn_layer_{i}'] = layer.cross_attn.attn_probs.cpu().numpy()
        
        # 3. 获取特征重要性（基于token嵌入的L2范数）
        if 'text_features' in features:
            text_features = features['text_features'].detach().cpu().numpy()
            # 计算每个特征的重要性（基于特征向量的L2范数）
            if len(text_features.shape) == 2:
                # 对于CLS token，我们需要获取所有token的特征
                # 注意：这里需要修改FTTransformer的forward方法，返回所有token的特征
                # 暂时使用简化的方法
                attention_weights['text_feature_importance'] = np.mean(np.abs(text_features), axis=0)
        
        if 'cog_features' in features:
            cog_features = features['cog_features'].detach().cpu().numpy()
            if len(cog_features.shape) == 2:
                attention_weights['cog_feature_importance'] = np.mean(np.abs(cog_features), axis=0)
        
        # 4. 获取跨模态注意力权重
        if hasattr(self.model, 'cross_attention') and hasattr(self.model.cross_attention, 'attn_probs') and self.model.cross_attention.attn_probs is not None:
            attention_weights['cross_modality_attention'] = self.model.cross_attention.attn_probs.cpu().numpy()
        
        return attention_weights
    
    def analyze_batch(self, dataloader, num_batches=5):
        """
        分析多个批次的数据
        Args:
            dataloader: 数据加载器
            num_batches: 分析的批次数
        Returns:
            平均注意力权重
        """
        all_attention = {}
        batch_count = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            attention = self.get_attention_weights(batch)
            
            # 累积注意力权重
            for key, value in attention.items():
                if key not in all_attention:
                    all_attention[key] = []
                all_attention[key].append(value)
            
            batch_count += 1
        
        # 计算平均值
        avg_attention = {}
        for key, values in all_attention.items():
            if key in ['text_feature_importance', 'cog_feature_importance']:
                # 对于特征重要性，计算每个特征的平均值
                stacked = np.stack(values)
                avg_attention[key] = np.mean(stacked, axis=0)
            else:
                # 对于注意力权重，计算每个位置的平均值
                stacked = np.stack(values)
                avg_attention[key] = np.mean(stacked, axis=0)
        
        return avg_attention
    
    def visualize_feature_importance(self, importance, feature_names=None, title='特征重要性', save_path=None):
        """
        可视化特征重要性
        Args:
            importance: 特征重要性数组
            feature_names: 特征名称列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        
        # 如果没有提供特征名称，使用索引
        if feature_names is None:
            feature_names = [f'特征{i+1}' for i in range(len(importance))]
        
        # 排序特征重要性
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # 只显示前20个最重要的特征
        top_k = min(20, len(sorted_importance))
        sorted_importance = sorted_importance[:top_k]
        sorted_names = sorted_names[:top_k]
        
        # 绘制柱状图
        plt.bar(range(top_k), sorted_importance)
        plt.xticks(range(top_k), sorted_names, rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"特征重要性图已保存到: {save_path}")
        else:
            plt.show()
    
    def visualize_attention_heatmap(self, attention, title='注意力热力图', save_path=None):
        """
        可视化注意力热力图
        Args:
            attention: 注意力权重矩阵
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 如果注意力是多头的，取平均值
        if len(attention.shape) == 4:  # [batch, heads, tokens, tokens]
            attention = np.mean(attention, axis=1)  # 平均多头注意力
        if len(attention.shape) == 3:  # [batch, tokens, tokens]
            attention = np.mean(attention, axis=0)  # 平均批次
        
        # 只显示前20个token
        n_tokens = min(20, attention.shape[0])
        attention = attention[:n_tokens, :n_tokens]
        
        # 绘制热力图
        sns.heatmap(attention, cmap='viridis')
        plt.title(title)
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"注意力热力图已保存到: {save_path}")
        else:
            plt.show()
    
    def run_analysis(self, dataloader, output_dir='attention_analysis'):
        """
        运行完整的注意力分析
        Args:
            dataloader: 数据加载器
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析注意力
        avg_attention = self.analyze_batch(dataloader)
        
        # 可视化特征重要性
        if 'text_feature_importance' in avg_attention:
            self.visualize_feature_importance(
                avg_attention['text_feature_importance'],
                title='语音特征重要性',
                save_path=os.path.join(output_dir, 'speech_feature_importance.png')
            )
        
        if 'cog_feature_importance' in avg_attention:
            self.visualize_feature_importance(
                avg_attention['cog_feature_importance'],
                title='认知量表特征重要性',
                save_path=os.path.join(output_dir, 'cog_feature_importance.png')
            )
        
        # 可视化注意力热力图
        for key, attention in avg_attention.items():
            if 'self_attn' in key:
                self.visualize_attention_heatmap(
                    attention,
                    title=f'{key} 注意力热力图',
                    save_path=os.path.join(output_dir, f'{key}_heatmap.png')
                )
        
        print(f"注意力分析完成，结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='注意力分析脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--modalities', type=str, default='speech_cog', help='使用的模态')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=2, help='工作线程数')
    parser.add_argument('--output_dir', type=str, default='attention_analysis', help='输出目录')
    args = parser.parse_args()
    
    # 配置
    config = {
        'txt_len': 209,  # 语音pca维度
        'cog_len': 80,  # 认知量表特征数量
        'batch_size_train': args.batch_size,
        'num_workers': args.num_workers,
        'use_position_embedding': True,
        'use_cross_attention': True,
        'modalities': args.modalities
    }
    
    # 加载数据
    print("加载数据...")
    try:
        dataloader = get_dataloader(
            datalist=['RMT_MRI-test'],
            batch_size=args.batch_size,
            txt_len=config['txt_len'],
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )
        print(f"成功创建数据加载器，批次数量: {len(dataloader)}")
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        sys.exit(1)
    
    # 动态计算认知量表特征的维度
    for batch in dataloader:
        if len(batch) == 5:
            _, _, cog_features, _, _ = batch
            config['cog_len'] = cog_features.shape[1]
            print(f"动态更新认知量表特征维度: {config['cog_len']}")
            break
    
    # 创建分析器
    analyzer = AttentionAnalyzer(args.model_path, config)
    
    # 运行分析
    analyzer.run_analysis(dataloader, args.output_dir)

if __name__ == "__main__":
    main()
