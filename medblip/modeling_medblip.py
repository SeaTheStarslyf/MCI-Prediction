import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F


from medblip.mae_vit import mae_vit_base_patch16
from medblip.mae_bert import BertConfig, BertForMaskedLM
from medblip.modeling_table import SimpleTableRestorer
from medblip.modeling_fttransformer import TableFTTRestorer, FTTransformer

from transformers import BertTokenizer
from sklearn.metrics import accuracy_score,roc_auc_score

import numpy as np
import cv2
import matplotlib.pyplot as plt
from medblip.utils import itc_loss, distillation_loss


# 简化版的FT-Transformer，用于学生模型
class StudentFTTransformer(nn.Module):
    """
    简化版的Feature Tokenizer Transformer，用于学生模型的蒸馏
    相比原版FTTransformer，减少了层数、注意力头数和前馈网络维度
    """
    def __init__(self, 
                 num_numerical_features: int,
                 categorical_cardinalities: List[int] = None,
                 d_token: int = 192,
                 num_heads: int = 4,  # 减少注意力头数
                 d_ff: int = 384,     # 减少前馈网络维度
                 num_layers: int = 2,  # 减少层数
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 output_dim: int = 768):
        super().__init__()
        
        if categorical_cardinalities is None:
            categorical_cardinalities = []
            
        # 复用原版FTTransformer的FeatureTokenizer
        self.feature_tokenizer = FTTransformer(num_numerical_features=num_numerical_features, 
                                              categorical_cardinalities=categorical_cardinalities, 
                                              d_token=d_token).feature_tokenizer
        
        # 是否使用CLS token
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # 简化的Transformer编码器
        from medblip.modeling_fttransformer import TransformerEncoder
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
        
        # 通过简化的Transformer编码器
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

label_map = {
    'AD':1, 
    'MCI':1,
    'CN':0,
}

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MedBLIPModel(nn.Module):
    def __init__(
        self,
        max_txt_len=60,
        hidden_size=768,
        num_class = 2,
        num_numerical_features=60,  # 语音特征中的数值特征数量
        use_ft_transformer=True,    # 是否使用FT-Transformer替代SimpleTableRestorer
        freeze_teacher=False,       # 是否冻结教师模型参数
        kd_temp=2.0,                # 蒸馏温度
        kd_weight=0.5,              # 蒸馏损失权重
        student_cls_weight=1.0,     # 学生分类损失权重
        feat_distill_weight=0.5,    # 特征级蒸馏权重
        disable_student=False       # 禁用学生模型（用于纯教师模型训练）
    ):
        super().__init__()

        # 蒸馏相关配置参数
        self.freeze_teacher = freeze_teacher
        self.distillation_temp = nn.Parameter(torch.tensor(kd_temp))
        self.kd_weight = kd_weight
        self.student_cls_weight = student_cls_weight
        self.image_mask_ratio = 0.75  # 图像掩码率，用于图像重构损失
        self.text_res_weight = 1.0  # 文本恢复损失权重，可配置
        self.feat_distill_weight = feat_distill_weight  # 特征级蒸馏权重
        self.disable_student = disable_student  # 是否禁用学生模型
        
        # 添加缺失的属性
        self.itc_temp = 0.07  # 对比学习温度
        self.criterion = nn.CrossEntropyLoss()  # 分类损失函数

        self.vision_transformer = mae_vit_base_patch16()

        config = BertConfig.from_pretrained("/data-pool/data/data2/qiuhui/tokenizer")
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 24
        
        # 使用FT-Transformer替代简单的表格处理器
        self.use_ft_transformer = use_ft_transformer
        if use_ft_transformer:
            # 使用FT-Transformer处理表格特征（教师模型）
            self.text_restorer = TableFTTRestorer(
                num_numerical_features=num_numerical_features,
                categorical_cardinalities=None,  # 假设语音特征都是数值型的
                hidden_dim=hidden_size,
                dropout=0.1,
                num_cross_attn_heads=config.num_attention_heads  # 使用与视觉编码器相同的头数
            )
            # 设置表格特征维度
            self.table_feature_dim = num_numerical_features
            
            # 简化的FT-Transformer用于学生模型
            self.student_ft_transformer = StudentFTTransformer(
                num_numerical_features=num_numerical_features,
                categorical_cardinalities=None,
                output_dim=hidden_size,
                dropout=0.1
            )
        else:
            # 保持原有实现作为备选
            self.text_restorer = SimpleTableRestorer(num_numerical_features, hidden_size)
        
        # 保存参数
        self.max_txt_len = max_txt_len

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_transformer.dim, hidden_size),
            nn.LayerNorm(hidden_size)  
        )
        self.text_proj = nn.Sequential(
            nn.Linear(max_txt_len, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 多模态融合分类器 [image_feat, speech_feat] - 教师模型
        self.mlp = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )
        # === 语音单模态头（student）===
        self.mlp_speech = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )
        
        # 特征级蒸馏损失
        self.feat_criterion = nn.MSELoss()

         # ===局部-全局对齐投影头 ===
        self.local_global_proj_image = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        self.local_global_proj_text = nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        self.itc_temp = nn.Parameter(0.07 * torch.ones([]))  # 对比学习温度
        self.max_txt_len = max_txt_len
        self.mlm_probability = 0.5
        self.criterion = nn.CrossEntropyLoss()
        
        # 根据训练模式设置参数的可训练性
        if self.disable_student:
            # 禁用学生模型模式：训练教师模型参数，完全禁用学生模型相关组件
            if hasattr(self, 'student_ft_transformer'):
                for param in self.student_ft_transformer.parameters():
                    param.requires_grad = False
            if hasattr(self, 'mlp_speech'):
                for param in self.mlp_speech.parameters():
                    param.requires_grad = False
            print(f"模型初始化完成 - 训练模式: 仅教师模型训练 (禁用学生模型)")
        elif self.freeze_teacher:
            # 学生模型蒸馏阶段：冻结教师模型参数，训练学生模型参数
            self._freeze_teacher_parameters()
        else:
            # 教师模型训练阶段：训练教师模型参数，冻结学生模型参数
            if hasattr(self, 'student_ft_transformer'):
                for param in self.student_ft_transformer.parameters():
                    param.requires_grad = False
            if hasattr(self, 'mlp_speech'):
                for param in self.mlp_speech.parameters():
                    param.requires_grad = False
            
        # 打印训练模式信息
        print(f"模型初始化完成 - 训练模式: {'学生模型蒸馏训练' if freeze_teacher else '教师模型训练'}")
        print(f"使用的表格处理器: {'FT-Transformer' if use_ft_transformer else 'SimpleTableRestorer'}")
        print(f"蒸馏参数: temperature={kd_temp}, kd_weight={kd_weight}, student_cls_weight={student_cls_weight}")
            
    def _freeze_teacher_parameters(self):
        """冻结教师模型参数，确保蒸馏过程中不更新教师模型"""
        # 添加静态标志位避免重复打印
        if not hasattr(self, '_has_printed_params_status'):
            self._has_printed_params_status = False
            
        # 冻结教师模型相关组件
        frozen_layers = []
        
        # 冻结视觉编码器参数
        for param in self.vision_transformer.parameters():
            param.requires_grad = False
        frozen_layers.append('vision_transformer')
        
        # 冻结视觉投影层参数
        for param in self.vision_proj.parameters():
            param.requires_grad = False
        frozen_layers.append('vision_proj')
            
        # 冻结多模态融合分类器参数（教师模型）
        for param in self.mlp.parameters():
            param.requires_grad = False
        frozen_layers.append('mlp')
            
        # 冻结文本处理相关参数 - 仅冻结用于教师模型的部分
        if hasattr(self, 'text_restorer'):
            for param in self.text_restorer.parameters():
                param.requires_grad = False
            frozen_layers.append('text_restorer')
            
        # 学生模型可训练参数
        trainable_layers = []
        
        # 确保学生FT-Transformer可训练
        if hasattr(self, 'student_ft_transformer'):
            for param in self.student_ft_transformer.parameters():
                param.requires_grad = True
            trainable_layers.append('student_ft_transformer')
            
        # 确保学生模型分类头可训练
        if hasattr(self, 'mlp_speech'):
            for param in self.mlp_speech.parameters():
                param.requires_grad = True
            trainable_layers.append('mlp_speech')
        
        # 只打印一次参数状态摘要
        if not self._has_printed_params_status:
            print(f"[模型参数状态] 已冻结: {', '.join(frozen_layers)}")
            print(f"[模型参数状态] 可训练: {', '.join(trainable_layers)}")
            print(f"[模型参数状态] 学生模型现在使用简化的FT-Transformer进行特征提取")
            
            # 确认冻结状态 - 打印参数数量统计
            total_params = sum(p.numel() for p in self.parameters())
            frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"[参数统计] 总参数: {total_params:,} 个")
            print(f"[参数统计] 冻结参数: {frozen_params:,} 个 ({frozen_params/total_params*100:.1f}%)")
            print(f"[参数统计] 可训练参数: {trainable_params:,} 个 ({trainable_params/total_params*100:.1f}%)")
            print(f"[参数统计] 学生模型训练约 {trainable_params/1000:.0f}K 个参数")
            
            # 标记已打印
            self._has_printed_params_status = True

    def augment_image(self, image):
        """MAE专用的简单图像增强"""
        aug_image = image.clone()
        device = aug_image.device
        
        # 60%概率增强
        if torch.rand(1, device=device) > 0.4:
            # 1. 高斯噪声 - 对MAE有意义
            if torch.rand(1, device=device) > 0.5:
                noise_std = 0.05 * torch.std(aug_image)
                noise = torch.randn_like(aug_image) * noise_std
                aug_image = aug_image + noise
            
            # 2. 对比度调整 - 对MAE有意义
            if torch.rand(1, device=device) > 0.5:
                contrast = 0.9 + 0.2 * torch.rand(1, device=device)  # [0.9, 1.1] 更保守
                mean_val = aug_image.mean()
                aug_image = contrast * (aug_image - mean_val) + mean_val
        
        return aug_image

    def augment_text(self, text):
        """简单有效的表格增强"""
        aug_text = text.clone()
        device = aug_text.device
        
        # 60%概率增强
        if torch.rand(1, device=device) > 0.4:
            # 1. 高斯噪声
            if torch.rand(1, device=device) > 0.5:
                noise_std = 0.05 * torch.std(aug_text, dim=0, keepdim=True)
                noise = torch.randn_like(aug_text) * noise_std
                aug_text = aug_text + noise
            
            # 2. 特征丢弃
            if torch.rand(1, device=device) > 0.5:
                drop_rate = 0.1  # 丢弃10%的特征
                mask = torch.rand_like(aug_text) > drop_rate
                aug_text = aug_text * mask.float()
        
        return aug_text

    def select_patches_by_attention(self,image_embeds, attns):
        """
        使用注意力权重选择重要patch
        attns: 来自vision transformer的注意力图列表
        """
        batch_size, num_tokens, hidden_dim = image_embeds.shape
        patch_embeds = image_embeds[:, 1:, :]  # 去掉CLS token
        
        # 使用最后一层的CLS token注意力权重
        last_layer_attn = attns[-1]  # [batch, heads, num_tokens, num_tokens]
        cls_attention = last_layer_attn[:, :, 0, 1:]  # CLS token对其他patch的注意力 [batch, heads, 512]
        cls_attention = cls_attention.mean(dim=1)  # [batch, 512]
        
        # 选择每个batch中注意力最高的patch
        num_key_patches = min(16, patch_embeds.shape[1])
        selected_patches = []
        
        for i in range(batch_size):
            _, topk_indices = torch.topk(cls_attention[i], num_key_patches)
            selected_patches.append(patch_embeds[i, topk_indices])
        
        return torch.stack(selected_patches)  # [batch, 16, 768]


    def mask_table_features(self, table_features, mask_ratio=0.2, targets=None, masked_indices=None):
        """
        表格数据的mask策略
        Args:
            table_features: [batch_size, feature_dim] 表格特征
            mask_ratio: mask比例
            targets: 可选的目标标签
            masked_indices: 可选的预定义mask位置
        """
        batch_size, feature_dim = table_features.shape
        device = table_features.device
        
        # 1. 创建mask位置
        if masked_indices is None:
            # 随机选择mask位置，每个样本独立
            probability_matrix = torch.full((batch_size, feature_dim), mask_ratio, device=device)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # 确保每个样本至少有一个特征被mask
            for i in range(batch_size):
                if not masked_indices[i].any():
                    # 随机选择一个特征mask
                    rand_idx = torch.randint(0, feature_dim, (1,), device=device)
                    masked_indices[i, rand_idx] = True
        
        # 2. 创建被mask的特征（将masked位置置0）
        masked_features = table_features.clone()
        masked_features[masked_indices] = 0
        
        # 3. 如果提供了targets，只计算被mask位置的loss
        if targets is not None:
            # 对于表格数据，targets通常是原始特征本身
            targets = targets.clone()
            targets[~masked_indices] = -100  # 忽略未被mask的位置
        
        if targets is not None:
            return masked_features, targets, masked_indices
        else:
            return masked_features, masked_indices

    
    def forward(self, samples):
        # 解包样本
        image, text, label, id = samples #image text已经归一化
        image = image.unsqueeze(1).cuda()  # [bs, 1, 128, 128, 128]
        text = text.cuda()
        label = label.cuda()
        id = id.cuda()
        
        # 禁用学生模型模式 - 只计算教师模型
        if self.disable_student:
            # 直接进入教师模型训练模式
            # ========== 数据增强 ==========
            aug_image = self.augment_image(image)    # 增强后的图像
            aug_text = self.augment_text(text)       # 增强后的表格

            # 使用增强数据提取特征（用于教师模型训练）
            aug_image_embeds, aug_attns, _, _ = self.vision_transformer.forward_encoder(
                img=aug_image, text_emb=None, mask_ratio=0)
            aug_image_embeds = self.vision_proj(aug_image_embeds)
            aug_image_feat = F.normalize(aug_image_embeds[:,0,:], dim=-1)

            # 表格特征处理
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 使用FT-Transformer处理表格特征
                masked_aug_text, mask_indices = self.mask_table_features(aug_text, mask_ratio=0.2)
                masked_aug_text_embeds = masked_aug_text.unsqueeze(1)
                
                text_res_loss, aug_text_features = self.text_restorer(
                    original_feat=aug_text,
                    masked_embeddings=masked_aug_text_embeds,
                    image_embeddings=aug_image_embeds,  # 传入图像嵌入以实现跨模态交互
                    mask=mask_indices
                )
                aug_text_embeds = aug_text_features.unsqueeze(1)
                aug_text_feat = F.normalize(aug_text_features, dim=-1)
            else:
                # 原始处理方式
                aug_text_embeds = self.text_proj(aug_text).unsqueeze(1)
                aug_text_feat = F.normalize(aug_text_embeds[:,0,:], dim=-1)
                
                # 表格重构
                masked_aug_text, mask_indices = self.mask_table_features(aug_text, mask_ratio=0.2)
                masked_aug_text_embeds_proj = self.text_proj(masked_aug_text).unsqueeze(1)

                text_res_loss = self.text_restorer(
                    original_feat=aug_text,
                    masked_embeddings=masked_aug_text_embeds_proj,
                    image_embeddings=aug_image_embeds,
                    mask=mask_indices
                )

            # 对比学习损失
            loss_itc = itc_loss(aug_image_feat, aug_text_feat, id, label, temp=self.itc_temp)

            # 图像重构损失
            image_res_loss, _, _ = self.vision_transformer(
                aug_image, text_emb=aug_text_embeds, mask_ratio=self.image_mask_ratio
            )

            # 局部对齐损失
            selected_patches = self.select_patches_by_attention(aug_image_embeds, aug_attns)
            local_image_feats = self.local_global_proj_image(selected_patches)
            global_text_feats = self.local_global_proj_text(aug_text_embeds)
            global_text_feats = global_text_feats.expand(-1, selected_patches.shape[1], -1)
            loss_local = F.mse_loss(local_image_feats, global_text_feats)

            # 教师模型推理
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                h_concat = torch.cat([aug_image_embeds[:,0], aug_text_features], dim=-1)
            else:
                h_concat = torch.cat([aug_image_embeds[:,0], aug_text_embeds[:,0]], dim=-1)
            
            y_hat_teacher = self.mlp(h_concat)
            loss_cls_tea = self.criterion(y_hat_teacher, label)

            # 组合所有损失 - 只包含教师模型相关损失
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                loss_itc +      # 对比学习损失
                text_res_loss + # 文本恢复损失
                image_res_loss +# 图像恢复损失
                loss_local      # 局部对齐损失
            )

            # 返回所有损失值和预测结果
            return {
                "loss": total_loss,
                'loss_itc': loss_itc,
                'loss_text_res': text_res_loss,
                'loss_image_res': image_res_loss,
                'loss_cls': loss_cls_tea,
                'loss_cls_teacher': loss_cls_tea,
                'loss_cls_student': 0.0,
                'loss_kl': 0.0,
                'loss_feat': 0.0,
                'loss_local': loss_local,
                'y_hat_teacher': y_hat_teacher,
                'y_hat_student': None
            }

        # ===== 教师模型训练阶段 =====
        if not self.freeze_teacher:
            # 教师模型训练模式 - 专注于教师模型的完整训练
            
            # ========== 数据增强 ==========
            aug_image = self.augment_image(image)    # 增强后的图像
            aug_text = self.augment_text(text)       # 增强后的表格

            # 使用增强数据提取特征（用于教师模型训练）
            aug_image_embeds, aug_attns, _, _ = self.vision_transformer.forward_encoder(
                img=aug_image, text_emb=None, mask_ratio=0)
            aug_image_embeds = self.vision_proj(aug_image_embeds)
            aug_image_feat = F.normalize(aug_image_embeds[:,0,:], dim=-1)

            # 表格特征处理
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 使用FT-Transformer处理表格特征
                masked_aug_text, mask_indices = self.mask_table_features(aug_text, mask_ratio=0.2)
                masked_aug_text_embeds = masked_aug_text.unsqueeze(1)
                
                text_res_loss, aug_text_features = self.text_restorer(
                    original_feat=aug_text,
                    masked_embeddings=masked_aug_text_embeds,
                    image_embeddings=aug_image_embeds,  # 传入图像嵌入以实现跨模态交互
                    mask=mask_indices
                )
                aug_text_embeds = aug_text_features.unsqueeze(1)
                aug_text_feat = F.normalize(aug_text_features, dim=-1)
            else:
                # 原始处理方式
                aug_text_embeds = self.text_proj(aug_text).unsqueeze(1)
                aug_text_feat = F.normalize(aug_text_embeds[:,0,:], dim=-1)
                
                # 表格重构
                masked_aug_text, mask_indices = self.mask_table_features(aug_text, mask_ratio=0.2)
                masked_aug_text_embeds_proj = self.text_proj(masked_aug_text).unsqueeze(1)

                text_res_loss = self.text_restorer(
                    original_feat=aug_text,
                    masked_embeddings=masked_aug_text_embeds_proj,
                    image_embeddings=aug_image_embeds,
                    mask=mask_indices
                )

            # 对比学习损失
            loss_itc = itc_loss(aug_image_feat, aug_text_feat, id, label, temp=self.itc_temp)

            # 图像重构损失
            image_res_loss, _, _ = self.vision_transformer(
                aug_image, text_emb=aug_text_embeds, mask_ratio=self.image_mask_ratio
            )

            # 局部对齐损失
            selected_patches = self.select_patches_by_attention(aug_image_embeds, aug_attns)
            local_image_feats = self.local_global_proj_image(selected_patches)
            global_text_feats = self.local_global_proj_text(aug_text_embeds)
            global_text_feats = global_text_feats.expand(-1, selected_patches.shape[1], -1)
            loss_local = F.mse_loss(local_image_feats, global_text_feats)

            # 教师模型推理
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                h_concat = torch.cat([aug_image_embeds[:,0], aug_text_features], dim=-1)
            else:
                h_concat = torch.cat([aug_image_embeds[:,0], aug_text_embeds[:,0]], dim=-1)
            
            y_hat_teacher = self.mlp(h_concat)
            loss_cls_tea = self.criterion(y_hat_teacher, label)

            # 教师训练阶段不训练学生模型
            
            # 组合所有损失 - 教师训练阶段只包含教师模型相关损失
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                loss_itc +      # 对比学习损失
                text_res_loss + # 文本恢复损失
                image_res_loss +# 图像恢复损失
                loss_local      # 局部对齐损失
            )

        # ===== 学生模型蒸馏训练阶段 =====
        else:
            # 学生模型蒸馏模式 - 只训练学生模型相关参数
            
            # 确保教师模型参数已冻结
            self._freeze_teacher_parameters()
            
            # ========== 数据准备 ==========
            aug_text = self.augment_text(text)  # 增强后的表格

            # 教师模型 - 使用原始数据生成软目标 (禁用梯度计算)
            with torch.no_grad():
                # 教师模型特征提取
                image_embeds_tea, _, _, _ = self.vision_transformer.forward_encoder(
                    img=image, text_emb=None, mask_ratio=0)
                image_embeds_tea = self.vision_proj(image_embeds_tea)
                
                # 文本特征处理
                if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                    _, text_features_tea = self.text_restorer(
                        original_feat=text,
                        masked_embeddings=text.unsqueeze(1),
                        image_embeddings=image_embeds_tea,
                        mask=None,
                        compute_loss=False
                    )
                    h_concat_tea = torch.cat([image_embeds_tea[:,0], text_features_tea], dim=-1)
                else:
                    text_embeds_tea = self.text_proj(text).unsqueeze(1)
                    h_concat_tea = torch.cat([image_embeds_tea[:,0], text_embeds_tea[:,0]], dim=-1)
                
                # 教师模型推理，生成软目标
                y_hat_teacher = self.mlp(h_concat_tea)
            
            # 教师模型特征提取和处理
            with torch.no_grad():
                if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                    # 使用教师模型的FT-Transformer处理文本特征
                    _, teacher_text_features = self.text_restorer(
                        original_feat=aug_text,
                        masked_embeddings=aug_text.unsqueeze(1),
                        image_embeddings=image_embeds_tea,
                        mask=None,
                        compute_loss=False
                    )
                else:
                    # 原始处理方式
                    teacher_text_embeds = self.text_proj(aug_text).unsqueeze(1)
                    teacher_text_features = teacher_text_embeds[:,0]
            
            # 学生模型 - 使用简化的FT-Transformer处理语音特征
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 使用简化的FT-Transformer处理增强的表格特征
                masked_aug_text, mask_indices = self.mask_table_features(aug_text, mask_ratio=0.2)
                
                # 使用学生的简化FT-Transformer
                student_text_features = self.student_ft_transformer(numerical_features=aug_text)
                
                # 表格重构损失（可选）
                text_res_loss = torch.tensor(0.0, device=aug_text.device)
                
                y_hat_student = self.mlp_speech(student_text_features)
            else:
                # 原始处理方式（备用）
                aug_text_embeds = self.text_proj(aug_text).unsqueeze(1)
                student_text_features = aug_text_embeds[:,0]
                
                # 表格重构
                masked_aug_text, mask_indices = self.mask_table_features(aug_text, mask_ratio=0.2)
                masked_aug_text_embeds_proj = self.text_proj(masked_aug_text).unsqueeze(1)

                text_res_loss = self.text_restorer(
                    original_feat=aug_text,
                    masked_embeddings=masked_aug_text_embeds_proj,
                    image_embeddings=None,  # 学生模型不使用图像特征
                    mask=mask_indices
                )
                
                y_hat_student = self.mlp_speech(student_text_features)
            
            # 计算学生模型分类损失 (硬目标监督)
            loss_cls_stu = self.criterion(y_hat_student, label) * self.student_cls_weight
            
            # 计算蒸馏损失 (软目标监督)
            loss_kd = distillation_loss(y_hat_teacher, y_hat_student, T=self.distillation_temp) * self.kd_weight
            
            # 计算特征级蒸馏损失 (中间特征对齐)
            loss_feat = self.feat_criterion(student_text_features, teacher_text_features) * self.feat_distill_weight
            
            # 组合总损失 - 学生模型只关注分类和蒸馏
            total_loss = loss_cls_stu + loss_kd + loss_feat

        # 返回所有损失值和预测结果
        return {
            "loss": total_loss,
            'loss_itc': loss_itc if 'loss_itc' in locals() else 0.0,
            'loss_text_res': text_res_loss,
            'loss_image_res': image_res_loss if 'image_res_loss' in locals() else 0.0,
            'loss_cls': (loss_cls_tea if 'loss_cls_tea' in locals() else 0.0) + (loss_cls_stu if 'loss_cls_stu' in locals() else 0.0) + (loss_kd if 'loss_kd' in locals() else 0.0),
            'loss_cls_teacher': loss_cls_tea if 'loss_cls_tea' in locals() else 0.0,
            'loss_cls_student': loss_cls_stu if 'loss_cls_stu' in locals() else 0.0,
            'loss_kl': loss_kd if 'loss_kd' in locals() else 0.0,
            'loss_feat': loss_feat if 'loss_feat' in locals() else 0.0,
            'loss_local': loss_local if 'loss_local' in locals() else 0.0,
            'y_hat_teacher': y_hat_teacher if 'y_hat_teacher' in locals() else None,
            'y_hat_student': y_hat_student if 'y_hat_student' in locals() else None
        }

        
    def predict(self, samples):
        """预测函数，同时返回教师模型和学生模型的预测结果"""
        # 支持字典形式输入以增强灵活性
        if isinstance(samples, dict):
            image = samples.get('image', None)
            text = samples.get('text', None)
            label = samples.get('label', None)
            id = samples.get('id', None)
            
            if image is not None:
                image = image.unsqueeze(1).cuda()
            if text is not None:
                text = text.cuda()
            if label is not None:
                label = label.cuda()
        else:
            # 保持原有接口兼容性
            image,text,label,id = samples
            image = image.unsqueeze(1).cuda() # bs c 128 128 128
            text = text.cuda()
            label = label.cuda()
        # id = id.cuda()
        
        # 教师模型推理
        image_embeds,attns, mask,ids_restore = self.vision_transformer.forward_encoder(
            img = image,
            text_emb=None,
            mask_ratio=0)
        image_embeds = self.vision_proj(image_embeds) #[24, 513, 768]

        def visualize_feature_map_on_image_qh(image, feature_map, path):
            # Resize the feature map to match the size of the raw image
            feature_map = cv2.resize(feature_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            attn = feature_map
            attn = (attn - attn.min())/(attn.max() - attn.min())
            
            attn = cv2.GaussianBlur(attn, (9, 9), 0)
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    if image[x,y]==0: # <=some value
                        attn[x,y]=0

            plt.imshow(image, cmap="gray")
            plt.imshow(attn, cmap="jet", alpha=0.7)
            plt.savefig(path)

        def visualize_image_qh(image,path):
            plt.imshow(image, cmap="gray")
            plt.savefig(path)

        # # # Example usage
        # bi = 2
        # image_3d = image[bi,0].cpu().numpy() # 128 128 128
        # attns = torch.stack(attns,-1) # bs 12 513 513 24
        # attns = attns[bi].mean(0) # 513 513 24
        # feature_map_3d = attns[0,1:,-1].view(8,8,8) # 8 8 8
        # feature_map_3d = feature_map_3d.cpu().numpy() # numpy

        # todo_list = [[24,1],[40,2],[46, 2], [56,3],[64,4],[72,4],[88,5]]

        # for i in range(5):
        #     aa, bb = todo_list[i][0],todo_list[i][1]

        #     # 0 plane
        #     image_2d = image_3d[aa,:,:]
        #     feature_map_2d = feature_map_3d[bb,:,:]
        #     visualize_image_qh(image_2d, './vis/0_img_{}.png'.format(aa))
        #     visualize_feature_map_on_image_qh(image_2d, feature_map_2d,'./vis/0_attn_{}.png'.format(aa))

        #     # 1 plane
        #     image_2d = image_3d[:,aa,:]
        #     feature_map_2d = feature_map_3d[:,bb,:]
        #     visualize_image_qh(image_2d, './vis/1_img_{}.png'.format(aa))
        #     visualize_feature_map_on_image_qh(image_2d, feature_map_2d,'./vis/1_attn_{}.png'.format(aa))

        #     # 2 plane
        #     image_2d = image_3d[:,:,aa]
        #     feature_map_2d = feature_map_3d[:,:,bb]
        #     visualize_image_qh(image_2d, './vis/2_img_{}.png'.format(aa))
        #     visualize_feature_map_on_image_qh(image_2d, feature_map_2d,'./vis/2_attn_{}.png'.format(aa))

        # import pdb;pdb.set_trace()

    

        # text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # inputs_ids: bs 70
        # encoded_layers, _, self_attn, cross_attn = self.text_transformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     output_attentions=True)
        
        # text_embeds = self.text_proj(encoded_layers[-1]) # bs txt_len 768 
        
        # 根据是否使用FT-Transformer来处理文本特征
        if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
            # 教师模型处理
            text_embeds_proj_teacher = self.text_restorer.ft_transformer(numerical_features=text)
            text_embeds_teacher = text_embeds_proj_teacher.unsqueeze(1)  # 添加一个维度以保持与原有接口兼容
            
            # 学生模型处理
            text_embeds_proj_student = self.student_ft_transformer(numerical_features=text)
        else:
            # 原始处理方式
            text_embeds_teacher = self.text_proj(text).unsqueeze(1)  # bs txt_len 768 
            text_embeds_proj_student = text_embeds_teacher[:,0]
        
        # 教师模型预测
        h_concat_teacher = torch.cat([image_embeds[:,0], text_embeds_proj_teacher], dim=-1)
        y_hat = self.mlp(h_concat_teacher)
        
        # 学生模型预测
        y_hat_speech = self.mlp_speech(text_embeds_proj_student)

        return y_hat, y_hat_speech, image_embeds[:,0], text_embeds_teacher[:,0]

    def tsne(self, samples):
        image = samples["image"].cuda()
        text = samples["text"]
        label = samples["label"].cuda()

        image_embeds = self.vision_proj(self.ln_vision(self.visual_encoder(image))) # bs img_len 768
        image_feat = F.normalize(image_embeds[:,0,:], dim=-1) # bs 768

        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device)
        encoded_layers, _, self_attn, cross_attn = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            output_attentions=True,)
        text_embeds = self.text_proj(encoded_layers[-1]) # bs txt_len 768
        text_feat = F.normalize(text_embeds[:, 0, :], dim=-1)


        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        
        encoder_input_ids = text_tokens.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        encoded_layers, _, self_attn, cross_attn = self.Qformer.bert(
            encoder_input_ids,
            attention_mask=text_tokens.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
        )
        output_itm = encoded_layers[-1]
        multi_feat = F.normalize(output_itm[:,0,:], dim=-1)


       
        return image_feat, text_feat, multi_feat, label

    def forward_bert(self, samples):
        text = samples["text"]
        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to('cuda:0')

        encoded_layers, _, self_attn, cross_attn = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            output_attentions=True,
        )
        output_logits = encoded_layers[-1]

        l_embeddings = output_logits[:, 0, :]
        cls_logits = self.cls_head(l_embeddings)
       
        return cls_logits, self_attn, cross_attn

    def forward_bert_tsne(self, samples):
        text = samples["text"]
        label = samples["label"].cuda()
        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to('cuda:0')

        encoded_layers, _, self_attn, cross_attn = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            output_attentions=True,
        )
        output_logits = encoded_layers[-1]

        l_embeddings = output_logits[:, 0, :]
       
        return l_embeddings,label

    def forward_vit(self, samples):
        image = samples["image"].cuda()
        cls_labels = samples["label"].cuda()

        image_embeds = self.vision_proj(self.ln_vision(self.visual_encoder(image))) # bs img_len 768

        v_embeddings = image_embeds[:, 0, :]
        cls_logits = self.cls_head(v_embeddings)
       
        loss_cls = F.cross_entropy(cls_logits, cls_labels)
        # print(' loss_cls: ', loss_cls.item())

        gts = cls_labels.cpu()
        preds = cls_logits.cpu()
        nonlabel_indices = torch.nonzero(gts==-100).squeeze()
        gts = torch.index_select(gts, 0, torch.tensor([i for i in range(gts.shape[0]) if i not in nonlabel_indices]))
        preds = torch.index_select(preds, 0, torch.tensor([i for i in range(preds.shape[0]) if i not in nonlabel_indices]))
        

        gts_one_hot = torch.nn.functional.one_hot(gts, num_classes=3)
        preds_ont_hot = torch.nn.functional.one_hot(preds.argmax(-1), num_classes=3)
        acc = accuracy_score(preds.argmax(-1).cpu(),gts)
        auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())

        print('loss cls: {:.6f}, acc: {:.6f}, auc: {:.6f}'.format(loss_cls , acc, auc))


        return {"loss": loss_cls}
    

def load_pretrained_vision_encoder(model, pretrained_path):
    """加载三分类预训练的视觉部分到二分类模型"""
    
    # 加载预训练权重
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    
    # 只保留视觉编码器相关的权重
    vision_keys = []
    for k in pretrained_dict.keys():
        if k.startswith('vision_transformer.') or k.startswith('vision_proj.'):
            vision_keys.append(k)
    
    # 更新当前模型的视觉部分权重
    update_count = 0
    for k in vision_keys:
        if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape:
            model_dict[k] = pretrained_dict[k]
            update_count += 1
        else:
            print(f"⚠️  跳过: {k} - 形状不匹配")
    
    # 加载权重（strict=False允许分类层不匹配）
    model.load_state_dict(model_dict, strict=False)
    
    print(f"✅ 视觉预训练权重加载完成")
    print(f"📊 更新了 {update_count}/{len(vision_keys)} 个视觉层")
    print(f"🎯 分类层保持二分类随机初始化")

    #冻结预训练权重
    print(f"✅ 视觉预训练权重冻结")
    for name, param in model.named_parameters():
        if "vision_transformer" in name or "vision_proj" in name:
            param.requires_grad = False
    
    return model

# 添加教师模型权重加载方法到MedBLIPModel类
import types

def load_teacher_weights(self, teacher_model_path):
    """从保存的教师模型中加载权重，支持分步训练策略"""
    print(f"从 {teacher_model_path} 加载教师模型权重...")
    
    # 加载教师模型权重
    checkpoint = torch.load(teacher_model_path, map_location='cpu')
    
    # 处理可能的checkpoint结构（如包含'model'键）
    if 'model' in checkpoint:
        teacher_dict = checkpoint['model']
    else:
        teacher_dict = checkpoint
    
    model_dict = self.state_dict()
    
    # 过滤并复制匹配的权重
    update_count = 0
    skip_count = 0
    for k, v in teacher_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            update_count += 1
        else:
            skip_count += 1
    
    # 加载权重
    self.load_state_dict(model_dict, strict=False)
    
    print(f"✅ 教师模型权重加载完成")
    print(f"📊 成功更新: {update_count} 个参数")
    print(f"⚠️  跳过不匹配: {skip_count} 个参数")
    
    # 如果设置了冻结教师模型，则重新冻结参数
    if self.freeze_teacher:
        self._freeze_teacher_parameters()
        print("✅ 教师模型参数已重新冻结")
    
    return self

# 动态添加方法到MedBLIPModel类
MedBLIPModel.load_teacher_weights = load_teacher_weights