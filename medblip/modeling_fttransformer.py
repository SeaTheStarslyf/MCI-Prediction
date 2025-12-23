import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from medblip.mae_vit import CrossAttention

class FeatureTokenizer(nn.Module):
    """
    特征分词器：将表格特征转换为token嵌入
    处理两种类型的特征：数值型和类别型
    """
    def __init__(self, num_numerical_features: int, categorical_cardinalities: List[int], d_token: int):
        super().__init__()
        
        # 数值特征处理
        self.numerical_embeddings = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_numerical_features)
        ])
        
        # 类别特征处理
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, d_token) for cat_size in categorical_cardinalities
        ])
        
        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities)
        self.d_token = d_token
    
    def forward(self, numerical_features: Optional[torch.Tensor] = None, 
                categorical_features: Optional[torch.Tensor] = None):
        tokens = []
        
        # 处理数值特征
        if numerical_features is not None and self.num_numerical > 0:
            for i in range(self.num_numerical):
                feat = numerical_features[:, i].unsqueeze(1)  # [B, 1]
                token = self.numerical_embeddings[i](feat)  # [B, d_token]
                tokens.append(token)
        
        # 处理类别特征
        if categorical_features is not None and self.num_categorical > 0:
            for i in range(self.num_categorical):
                feat = categorical_features[:, i]  # [B]
                token = self.categorical_embeddings[i](feat)  # [B, d_token]
                tokens.append(token)
        
        # 将所有token拼接起来
        tokens = torch.stack(tokens, dim=1)  # [B, num_features, d_token]
        return tokens

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 确保d_model能被num_heads整除
        assert d_model % num_heads == 0
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x: [B, N, d_model], where N is the number of tokens
        batch_size, num_tokens, _ = x.shape
        
        # 应用层归一化
        residual = x
        x = self.layer_norm(x)
        
        # 线性投影并分割为多个头
        q = self.W_q(x).view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, N, d_k]
        k = self.W_k(x).view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, N, d_k]
        v = self.W_v(x).view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, N, d_k]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, num_heads, N, N]
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # 应用softmax和dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 计算加权和
        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, N, d_k]
        
        # 合并多个头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_model)  # [B, N, d_model]
        
        # 应用输出投影并添加残差连接
        output = self.W_o(attn_output) + residual  # [B, N, d_model]
        
        return output

class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor):
        # x: [B, N, d_model]
        
        # 应用层归一化
        residual = x
        x = self.layer_norm(x)
        
        # 应用前馈网络
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        
        # 添加残差连接
        output = x + residual  # [B, N, d_model]
        
        return output

class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttention(d_model, num_heads, dropout),
                FeedForward(d_model, d_ff, dropout)
            ]) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x: [B, N, d_model]
        
        for attn, ffn in self.layers:
            x = attn(x, attention_mask)
            x = ffn(x)
        
        return x

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data
    """
    def __init__(self, 
                 num_numerical_features: int,
                 categorical_cardinalities: List[int] = None,
                 d_token: int = 192,
                 num_heads: int = 8,
                 d_ff: int = 768,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 output_dim: int = 768):
        super().__init__()
        
        if categorical_cardinalities is None:
            categorical_cardinalities = []
            
        # 特征分词器
        self.feature_tokenizer = FeatureTokenizer(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=categorical_cardinalities,
            d_token=d_token
        )
        
        # 是否使用CLS token
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer编码器
        self.encoder = TransformerEncoder(
            d_model=d_token,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 输出投影层（用于匹配MedBLIP模型的维度）
        self.output_projection = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Linear(d_token * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, numerical_features: Optional[torch.Tensor] = None, 
                categorical_features: Optional[torch.Tensor] = None):
        # 对特征进行分词
        tokens = self.feature_tokenizer(numerical_features, categorical_features)
        batch_size = tokens.shape[0]
        
        # 添加CLS token（如果使用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 通过Transformer编码器
        encoder_output = self.encoder(tokens)
        
        # 如果使用CLS token，只使用CLS token的输出
        if self.use_cls_token:
            output = encoder_output[:, 0]
        else:
            # 否则，对所有token的输出进行平均池化
            output = encoder_output.mean(dim=1)
        
        # 通过输出投影层
        output = self.output_projection(output)
        
        return output

class TableFTTRestorer(nn.Module):
    """
    使用FT-Transformer处理表格数据，并提供与SimpleTableRestorer兼容的接口
    采用与图像处理相同的单向交叉注意力机制
    """
    def __init__(self, 
                 num_numerical_features: int,
                 categorical_cardinalities: List[int] = None,
                 hidden_dim: int = 768,
                 dropout: float = 0.1,
                 num_cross_attn_heads: int = 8):
        super().__init__()
        
        # FT-Transformer用于表格特征处理
        self.ft_transformer = FTTransformer(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=categorical_cardinalities,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 添加单向Cross-Attention层：与图像处理一致，表格特征作为query，图像特征作为key和value
        self.cross_attn = CrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attn_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # LayerNorm层，与图像处理保持一致
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 重构层：尝试重构表格特征
        self.reconstruction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_numerical_features)
        )
    
    def forward(self, original_feat, masked_embeddings, image_embeddings, mask, compute_loss=True):
        """
        与SimpleTableRestorer兼容的接口，使用与图像处理相同的单向交叉注意力机制
        
        Args:
            original_feat: 原始表格特征
            masked_embeddings: 掩码后的表格嵌入
            image_embeddings: 图像嵌入
            mask: 掩码位置
            compute_loss: 是否计算重构损失
        
        Returns:
            如果compute_loss为True: (loss, table_embedding)
            如果compute_loss为False: (0, table_embedding)
        """
        # 使用FT-Transformer处理表格特征
        # 这里假设original_feat是数值型特征
        table_embedding = self.ft_transformer(numerical_features=original_feat)
        
        # 准备用于Cross-Attention的特征格式
        # 表格特征需要增加一个维度以匹配cross attention的输入格式 [B, N, C]
        table_embedding_reshaped = table_embedding.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 如果提供了图像嵌入，则进行跨模态交互
        # 与图像处理一致，只进行单向注意力：表格特征关注图像特征
        if image_embeddings is not None:
            # 应用LayerNorm，与图像处理保持一致
            normed_table_emb = self.norm1(table_embedding_reshaped)
            normed_image_emb = self.norm2(image_embeddings)
            
            # 表格特征关注图像特征，与Block类中的实现一致
            attended_table_emb = self.cross_attn(normed_table_emb, normed_image_emb)
            
            # 添加残差连接，与图像处理保持一致
            enhanced_table_emb = table_embedding_reshaped + attended_table_emb
            
            # 获取增强后的表格特征 (去掉多余维度)
            enhanced_table_emb = enhanced_table_emb.squeeze(1)  # [B, hidden_dim]
            
            # 如果不需要计算损失，可以直接返回
            loss = torch.tensor(0.0, device=original_feat.device)
            
            if compute_loss:
                # 重构表格特征
                restored_feat = self.reconstruction_layer(enhanced_table_emb)
                
                # 计算损失
                if mask is not None and mask.sum() > 0:
                    loss = F.smooth_l1_loss(restored_feat[mask], original_feat[mask])
            
            # 返回增强后的表格嵌入，用于后续的多模态融合
            return loss, enhanced_table_emb
        else:
            # 如果没有图像嵌入，则使用原始表格特征
            # 如果不需要计算损失，可以直接返回
            loss = torch.tensor(0.0, device=original_feat.device)
            return loss, table_embedding