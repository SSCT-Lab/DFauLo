import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 指定字体路径
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = fm.FontProperties(fname=font_path, size=24)  # 调整字体大小

# 输入层np.vstack((data1[0][2], data2[0][2], data3[0][2]))
# 中间层np.vstack((data1, data2, data3))
# 生成示例数据
def drawPCA(data1, data2, data3):
    print(data1.shape)
    print(data2.shape)
    print(data3.shape)
    # 合并数据集
    data_combined = np.vstack((data1, data2, data3))

    # 标准化数据
    scaler = StandardScaler()
    data_combined_scaled = scaler.fit_transform(data_combined)

    # PCA降维到3D
    pca = PCA(n_components=3)
    data_combined_pca = pca.fit_transform(data_combined_scaled)

    # 分开降维后的数据
    data1_pca = data_combined_pca[:1]
    data2_pca = data_combined_pca[1:2]
    data3_pca = data_combined_pca[2:]

    # 绘制3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制数据点
    ax.scatter(data1_pca[:, 0], data1_pca[:, 1], data1_pca[:, 2], label='正常数据-1', alpha=0.7, edgecolors='w', s=100)
    ax.scatter(data2_pca[:, 0], data2_pca[:, 1], data2_pca[:, 2], label='正常数据-2', alpha=0.7, edgecolors='w', s=100)
    ax.scatter(data3_pca[:, 0], data3_pca[:, 1], data3_pca[:, 2], label='缺陷数据', alpha=0.7, edgecolors='w', s=100)

    # 添加图例和标签
    ax.set_title('输出层内部异常行为与缺陷触发关系', fontsize=24, fontproperties=font_prop)
    ax.set_xlabel('Principal Component 1', fontsize=18, fontproperties=font_prop)
    ax.set_ylabel('Principal Component 2', fontsize=18, fontproperties=font_prop)
    ax.set_zlabel('Principal Component 3', fontsize=18, fontproperties=font_prop)
    ax.legend(prop=font_prop)
    plt.savefig('PCA3D.png', dpi=300)

# 示例调用
# drawPCA(data1, data2, data3)
