import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleTableRestorer(nn.Module):
    def __init__(self, table_dim=512, hidden_dim=768):
        super().__init__()
        
        # 简单的MLP架构
        self.restore_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1024),
            nn.LayerNorm(1024),  # 保持稳定
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, table_dim)  # 输出维度匹配原始表格
        )
        
        # 可选：添加一个小的特征交互层
        self.feature_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
    
    def forward(self, original_feat, masked_embeddings, image_embeddings, mask):
        """
        original_feat: [batch_size, table_dim] 原始表格数值
        masked_embeddings: [batch_size, 1, hidden_dim] 投影后的masked嵌入
        image_embeddings: [batch_size, seq_len, hidden_dim] 图像嵌入
        mask: [batch_size, table_dim] mask位置
        """
        batch_size = original_feat.shape[0]
        
        # # 方法1：直接使用特征（最简单）
        # text_context = masked_embeddings.squeeze(1)  # [batch_size, hidden_dim]
        # image_context = image_embeddings[:, 0, :]    # [batch_size, hidden_dim]
        
        # 方法2：轻微的特征交互（可选）
        text_context = self.feature_interaction(masked_embeddings.squeeze(1))
        image_context = self.feature_interaction(image_embeddings[:, 0, :])
        
        # 融合特征
        fused = torch.cat([text_context, image_context], dim=1)  # [batch_size, 2*hidden_dim]
        
        # 重构表格特征
        restored_feat = self.restore_mlp(fused)  # [batch_size, table_dim]
        
        # 计算损失
        if mask.sum() > 0:
             loss = F.smooth_l1_loss(restored_feat[mask], original_feat[mask])
        else:
            loss = torch.tensor(0.0, device=original_feat.device)
        
        return loss