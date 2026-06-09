import pandas as pd
import os

# 定义文件路径
data_dir = 'c:\\Users\\Lenovo\\Desktop\\deep\\local_data'

# 读取现有的train和test数据集，提取PTID
train_file = os.path.join(data_dir, 'RMT_MRI-train.csv')
test_file = os.path.join(data_dir, 'RMT_MRI-test.csv')

# 读取train和test数据并提取PTID
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_ptids = set(train_data['PTID'].tolist())
test_ptids = set(test_data['PTID'].tolist())

print(f"Train set size: {len(train_ptids)}")
print(f"Test set size: {len(test_ptids)}")

# 处理MMSE数据集
mmse_file = os.path.join(data_dir, 'MMSE_matched_to_RMT_MRI_split.csv')
mmse_data = pd.read_csv(mmse_file)

# 划分MMSE数据集
mmse_train = mmse_data[mmse_data['PTID'].isin(train_ptids)]
mmse_test = mmse_data[mmse_data['PTID'].isin(test_ptids)]

print(f"MMSE train size: {len(mmse_train)}")
print(f"MMSE test size: {len(mmse_test)}")

# 保存划分后的MMSE数据集
mmse_train.to_csv(os.path.join(data_dir, 'MMSE-train.csv'), index=False)
mmse_test.to_csv(os.path.join(data_dir, 'MMSE-test.csv'), index=False)

# 处理MOCA数据集
moca_file = os.path.join(data_dir, 'MOCA_RMT_MRI_split.csv')
moca_data = pd.read_csv(moca_file)

# 划分MOCA数据集
moca_train = moca_data[moca_data['PTID'].isin(train_ptids)]
moca_test = moca_data[moca_data['PTID'].isin(test_ptids)]

print(f"MOCA train size: {len(moca_train)}")
print(f"MOCA test size: {len(moca_test)}")

# 保存划分后的MOCA数据集
moca_train.to_csv(os.path.join(data_dir, 'MOCA-train.csv'), index=False)
moca_test.to_csv(os.path.join(data_dir, 'MOCA-test.csv'), index=False)

print("数据集划分完成！")
print(f"MMSE训练集保存在: {os.path.join(data_dir, 'MMSE-train.csv')}")
print(f"MMSE测试集保存在: {os.path.join(data_dir, 'MMSE-test.csv')}")
print(f"MOCA训练集保存在: {os.path.join(data_dir, 'MOCA-train.csv')}")
print(f"MOCA测试集保存在: {os.path.join(data_dir, 'MOCA-test.csv')}")