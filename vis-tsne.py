import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 生成示例数据
np.random.seed(42)
data = np.random.randn(100, 5)

# 创建一个TSNE对象
tsne = TSNE(n_components=2)

# 计算嵌入
embedded_data = tsne.fit_transform(data) # 传入所需的embedding：emb.detach().cpu().numpy()

# 可视化嵌入
plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
plt.title('t-SNE Visualization')
plt.savefig('./vis-tsne.jpg')