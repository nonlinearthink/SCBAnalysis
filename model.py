# %%
"""库函数导入、定义函数与类"""
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings
import os
from IPython.display import display

# 解决matplotlib无法正常显示中文的问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['STHeiti']
# 默认字体是mac用户的，windows使用下面的这条代码
# plt.rcParams['font.sans-serif'] = ['Simhei']


def comparision(vec: list):
    """
    生成比较矩阵
    CopyRight:
        https://blog.csdn.net/aBIT_Tu/article/details/84029849
    """
    n = len(vec)
    F = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                F[i, j] = 1
            else:
                F[i, j] = vec[i] / vec[j]
    return F


class AHP:
    """AHP类
    提供AHP算法支持
    CopyRight:
        https://www.guofei.site/2020/01/05/ahp.html
    """
    def __init__(self, criteria, b):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria
        self.b = b
        self.num_criteria = criteria.shape[0]
        self.num_project = b[0].shape[0]

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, '不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反互对称矩阵')

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n]
        return max_eigen, CR, eigen

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        print('准则层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(
            max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_eigen))

        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(eigen)

        pd_print = pd.DataFrame(
            eigen_list,
            index=['准则' + str(i) for i in range(self.num_criteria)],
            columns=['方案' + str(i) for i in range(self.num_project)],
        )
        pd_print.loc[:, '最大特征值'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1
        print('方案层')
        print(pd_print)

        # 目标层
        obj = np.dot(criteria_eigen.reshape(1, -1), np.array(eigen_list))
        print('\n目标层', obj)
        print('最优选择是方案{}'.format(np.argmax(obj)))
        return obj


# %%
"""准备模型所需的特征数据集"""
amount = pd.read_csv(os.path.join('data', 'consume.csv'))

amount = amount[amount.Dept.str.contains('食')
                | amount.Dept.str.contains('超市')]  # 筛选出所有的非公共消费，有部分消费是水电费等消费

# 创建模型评价所需的特征数据集
dict_list = []
for name, group in amount.groupby(amount['CardNo']):
    money = group['Money'].agg(['sum', 'mean', 'max'])
    dict_t = {
        'CardNo': name,
        'AvgSurplus': format(group['Surplus'].agg(['mean'])[0], '.2f'),
        'TotalConsume': format(money[0], '.2f'),
        'Freq': group.shape[0]
    }
    dict_list.append(dict_t)

# 导出成DataFrame
features = pd.DataFrame(dict_list[0:])
features = features[features.Freq >= 30]  # 筛选掉所有消费频率低于30次的数据，认为是不正常的
display(features)

# %%
"""通过Elbow Method求K-Means最佳k值"""
training_data = features.loc[0:, ['AvgSurplus', 'TotalConsume',
                                  'Freq']]  # K-means训练集

# 当k小于真实聚类数时，由于k的增大会大幅增加每个簇的聚合程度，故 SSE 的下降幅度会很大。
# 而当k到达真实聚类数时，再增加k所得到的聚合程度回报会迅速变小，所以 SSE 的下降幅度会骤减，
# 随着k值的继续增大而趋于平缓。
# 将每个簇的质点与簇内样本点的平方距离误差和称为畸变程度(distortions)，
# 那么，对于一个簇，它的畸变程度越低，代表簇内成员越紧密，
# 畸变程度越高，代表簇内结构越松散。
# 畸变程度会随着类别的增加而降低，
# 但对于有一定区分度的数据，在达到某个临界点时畸变程度会得到极大改善，之后缓慢下降，
# 这个临界点就可以考虑为聚类性能较好的点。

SSE = []  # 存放每次结果的误差平方和
for k in range(2, 20):
    km = KMeans(n_clusters=k)  # 构造聚类器
    scaler = StandardScaler()
    pipline = make_pipeline(scaler, km)
    pipline.fit(training_data)
    SSE.append(km.inertia_)  # inertia_:样本到其最近聚类中心的平方距离之和

plt.figure(figsize=(10, 5))
plt.plot(range(2, 20), SSE, 'o-')
plt.xticks(range(0, 22, 1))
plt.grid(linestyle='--')
plt.xlabel("分类数量K")
plt.ylabel('误差平方和(SSE)')
plt.savefig(os.path.join('img', '2-1.png'), dpi=300)

# %%
"""创建K-means模型"""
# 由上图可以认为最佳的K值在5左右，我们选取n_clusters为5。
km = KMeans(n_clusters=6, algorithm='elkan')
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
pipline = make_pipeline(scaler, km)  # 训练kmeans模型

pipline.fit(training_data)

features['Labels'] = pipline.predict(training_data)  # 预测数据的Labels

display(features)

# %%
"""基于DFM(Deposit、Frequency、Monetary)模型的K-Means分类"""
fig = plt.figure(figsize=(8, 6))  # figsize-画布大小

ax = Axes3D(fig)  # 创建3D图
colors = ['c', 'darkorange', 'lime', 'red', 'fuchsia', 'yellow']
for name, group in features.groupby(features['Labels']):
    std = MinMaxScaler(feature_range=(0, 1))  # 标准化，不然画出来比较难看
    x = std.fit_transform(group['AvgSurplus'].values.reshape(-1, 1))
    y = std.fit_transform(group['TotalConsume'].values.reshape(-1, 1))
    z = std.fit_transform(group['Freq'].values.reshape(-1, 1))
    print('drawing label {}'.format(name))
    ax.scatter3D(x, y, z, c=colors[name], label=name, s=4)
ax.view_init(30, 45)  # 旋转视图的角度
ax.set_xlabel('平均存储系数')
ax.set_ylabel('总消费系数')
ax.set_zlabel('消费频率系数')
plt.legend()
plt.savefig(os.path.join('img','2-2.png'), dpi=300)

# %%
"""对上图模型的验证"""
# 根据上图请自行调整，因为K-Means分类的label值不稳定，所以每次运行完前面的无比更改这个值
# 这个类是上图中所有项都最低的那一个类

group_label = 0
deep_model = features[features.Labels == group_label]
deep_model['AvgSurplus'] = deep_model['AvgSurplus'].astype("float64")
deep_model['TotalConsume'] = deep_model['TotalConsume'].astype("float64")
display(deep_model)

# %%
"""基于AHP算法细分模型"""

criteria = comparision([1, 5, 3])
b = [
    comparision(deep_model['AvgSurplus'].values),
    comparision(deep_model['TotalConsume'].values),
    comparision(deep_model['Freq'].values)
]

results = AHP(criteria, b).run()

# %%
""""""
deep_model['Rank'] = results[0]
deep_model.sort_values(by='Rank')


# %%
