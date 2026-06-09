import torch
import os

# 加载预训练权重
pretrained_path = 'pretrained/latest.pth'
print(f"加载预训练权重: {pretrained_path}")

if not os.path.exists(pretrained_path):
    print(f"错误: 找不到预训练权重文件 {pretrained_path}")
    exit(1)

# 加载权重
pretrained_dict = torch.load(pretrained_path, map_location='cpu')
print(f"预训练权重加载成功，包含 {len(pretrained_dict)} 个键值对")

# 分析权重结构
print("\n=== 权重结构分析 ===")

# 按模块分类
modules = {}
for key in pretrained_dict.keys():
    # 提取模块名
    if '.' in key:
        module_name = key.split('.')[0]
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append(key)
    else:
        if 'other' not in modules:
            modules['other'] = []
        modules['other'].append(key)

# 打印每个模块的权重
for module_name, keys in modules.items():
    print(f"\n模块: {module_name} ({len(keys)} 个权重)")
    for key in keys[:10]:  # 每个模块最多显示10个权重
        shape = pretrained_dict[key].shape
        print(f"  - {key}: {shape}")
    if len(keys) > 10:
        print(f"  ... 等 {len(keys) - 10} 个权重")

# 特别分析视觉相关的权重
print("\n=== 视觉相关权重详细分析 ===")
vision_keys = []
for key in pretrained_dict.keys():
    if key.startswith('vision_transformer.') or key.startswith('vision_proj.'):
        vision_keys.append(key)
        shape = pretrained_dict[key].shape
        print(f"- {key}: {shape}")

print(f"\n视觉相关权重总数: {len(vision_keys)}")

# 分析vision_proj的具体形状
if 'vision_proj.weight' in pretrained_dict:
    print(f"\nvision_proj.weight 形状: {pretrained_dict['vision_proj.weight'].shape}")
if 'vision_proj.bias' in pretrained_dict:
    print(f"vision_proj.bias 形状: {pretrained_dict['vision_proj.bias'].shape}")

# 分析vision_transformer的输出维度
if 'vision_transformer.cls_token' in pretrained_dict:
    print(f"\nvision_transformer.cls_token 形状: {pretrained_dict['vision_transformer.cls_token'].shape}")
if 'vision_transformer.pos_embed' in pretrained_dict:
    print(f"vision_transformer.pos_embed 形状: {pretrained_dict['vision_transformer.pos_embed'].shape}")