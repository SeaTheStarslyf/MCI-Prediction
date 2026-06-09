import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset_RMI import get_dataloader

# 测试数据加载
try:
    print("测试数据加载...")
    dataloader = get_dataloader(
        datalist=['RMT_MRI-train'],
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    print("数据加载成功！")
    
    # 查看第一个批次的数据
    for batch in dataloader:
        image, features, cog_features, label, id = batch
        print(f"图像形状: {image.shape}")
        print(f"特征形状: {features.shape}")
        print(f"认知量表特征形状: {cog_features.shape}")
        print(f"标签: {label}")
        print(f"ID: {id}")
        break
    
    print("测试成功！新数据已成功合并到认知量表模态中。")
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()