import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 删除PCA导入

label_map = {
    'AD': 1,
    'MCI': 1,
    'CN': 0,
}

class Dataset(torch.utils.data.Dataset):
    """
    Loads data and corresponding label and returns pytorch float tensor.
    """
    def __init__(self, data):
        self.files = data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Read data and label and return them.
        """
        img_path = self.files[idx]['image_path']

        # 使用SimpleITK
        image_sitk = sitk.ReadImage(img_path)

        # 1. 标准化
        normalizer = sitk.NormalizeImageFilter()
        normalized_image = normalizer.Execute(image_sitk)

        # 2. 直接重采样到目标尺寸，避免后续padding
        original_size = normalized_image.GetSize()
        target_size = [128, 128, 128]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        # 保持原有的物理间距比例
        resampler.SetOutputSpacing([orig_sz * target_sp / target_sz 
                                for orig_sz, target_sz, target_sp in 
                                zip(original_size, target_size, normalized_image.GetSpacing())])
        resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值
        resampled_image = resampler.Execute(normalized_image)

        # 3. 直接转换为tensor，无需padding和resize
        data = sitk.GetArrayFromImage(resampled_image).astype(np.float32)
        image = torch.FloatTensor(data)  # 已经是128x128x128

        features = self.files[idx]['features']
        label = self.files[idx]['label']
        id = self.files[idx]['id']

        return image, features, label, id


def get_dataloader(datalist=['ADNI-train'], batch_size=1, txt_len=60, shuffle=False, num_workers=12, drop_last=False):
    all_features = []
    all_img_paths = []
    all_labels = []
    
    # 第一步：收集所有数据
    for data in datalist:
        filename = f'Alifuse_bibm/local_data/{data}.csv'
        print('load data from', filename)
        
        df = pd.read_csv(filename, header=0)
        
        # 提取特征列
        feature_columns = [col for col in df.columns if df.columns.get_loc(col) >= df.columns.get_loc('SCREENER_SCORE')]
        print(f"使用 {len(feature_columns)} 个特征列")
        
        # 批量提取所有特征
        features_batch = df[feature_columns].values.astype(np.float32)
        features_batch = np.nan_to_num(features_batch)  # 处理缺失值
        
        # 收集数据
        all_features.append(features_batch)
        all_img_paths.extend(df['MRI_Path'].tolist())
        all_labels.extend([label_map.get(group, -1) for group in df['Group']])
        all_ids = [abs(hash(path)) for path in all_img_paths]
    
    # 合并所有特征
    all_features = np.vstack(all_features)
    print(f"总样本数: {len(all_features)}")
    print(f"特征矩阵形状: {all_features.shape}")
    
    # 第二步：在整个数据集上进行标准化（去掉PCA降维）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    print("标准化完成")
    
    # 不再进行PCA降维，直接使用所有特征
    # 注意：txt_len参数现在不再使用，保留是为了兼容性
    if txt_len is not None and txt_len < features_scaled.shape[1]:
        print(f"警告：txt_len参数({txt_len})小于特征数({features_scaled.shape[1]})，但不再进行降维")
        print(f"使用所有{features_scaled.shape[1]}个特征")
    
    # 第三步：创建数据列表
    files = []
    for i in range(len(all_img_paths)):
        files.append({
            'image_path': all_img_paths[i],
            'features': torch.FloatTensor(features_scaled[i]),  # 使用标准化后的所有特征
            'label': all_labels[i],
            'id': all_ids[i]
        })
    
    # 创建数据集和数据加载器
    dataset = Dataset(data=files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           num_workers=num_workers, drop_last=drop_last)

    return dataloader