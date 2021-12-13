import pandas as pd
import numpy as np
import datetime
import sklearn.preprocessing
import sklearn.cluster
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# 读取数据
air_data_path = 'D:/Download/基于聚类算法完成航空客户价值分析任务-数据集/datasets/air_data.csv'
air_data = pd.read_csv(air_data_path)

# 展示每列数据的类型
print(air_data.dtypes)

# 预览前 5 条数据
print(air_data.head(5))

# 使用 pandas 中的 DataFrame 的 describe() 函数表述数据的基本统计信息
print(air_data.describe().T)

# 检查数据中是否有重复的会员ID
dup = air_data[air_data['MEMBER_NO'].duplicated()]
if len(dup) != 0:
    print("There are duplication in the data:")
    print(dup)

# 统计数据集中缺失值情况
print(air_data.isnull().any())

# 对于存在缺失值的属性调用 notnull() 函数
boolean_filter = air_data['SUM_YR_1'].notnull() & air_data['SUM_YR_2'].notnull()
air_data = air_data[boolean_filter]

filter_1 = air_data['SUM_YR_1'] != 0
filter_2 = air_data['SUM_YR_2'] != 0
air_data = air_data[filter_1 | filter_2]

print(air_data.shape)

# 在对于客户价值分析的一个经典模型RFM模型的基础上，建立航空行业的LRFMC模型。
# Length of Relationship:客户关系时长，反映可能的活跃时长。
# Recency:最近消费时间间隔，反映当前的活跃状态。
# Frequency:客户消费频率，反映客户的忠诚度。
# Mileage:客户总飞行里程，反映客户对乘机的依赖性。
# Coefficient of Discount:客户所享受的平均折扣率，侧面反映客户价值高低。

# 针对LRFMC模型，增加一个表示关系长度(L)的属性(列) L = LOAD_YIME - FFP_DATE
load_time = datetime.datetime.strptime('2014/03/31', '%Y/%m/%d')
ffp_dates = [datetime.datetime.strptime(ffp_date, '%Y/%m/%d') for ffp_date in air_data['FFP_DATE']]
length_of_relationship = [(load_time - ffp_date).days for ffp_date in ffp_dates]
air_data['LEN_REL'] = length_of_relationship

# 移除我们不关心的属性(列)，即只保留LRFMC模型需要的属性
features = ['LEN_REL', 'FLIGHT_COUNT', 'avg_discount', 'SEG_KM_SUM', 'LAST_TO_END']
data = air_data[features]
features = ['L', 'F', 'C', 'M', 'R']
data.columns = features
print(data)

# 预览前 5 行数据，并查看数据的元数据。
print(data.head(5))
print(data.describe().T)

# 对特征进行标准化，使得各特征的均值为0，方差为1.
ss = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)  # 标准化
data = ss.fit_transform(data)  # 数据转换
data = pd.DataFrame(data, columns=features)

data_db = data.copy()

# 标准化后的数据的元数据
print(data.describe().T)

# 模型训练与对数据的预测
# 使用 k-means 聚类算法来分析数据
num_clusters = 5  # 设置类别为5
km = sklearn.cluster.KMeans(n_clusters=num_clusters)  # 模型加载
km.fit(data)  # 模型训练

# 查看模型学习出来的5个群体的中心，以及5个群体所包含的样本个数
r1 = pd.Series(km.labels_).value_counts()
r2 = pd.DataFrame(km.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
r.columns = list(data.columns) + ['counts']
print(r)

# 查看模型对每个样本预测的群体标签
print(km.labels_)

# 尝试使用RFM模型
data_rfm = data[['R', 'F', 'M']]
print(data_rfm.head(5))

# 模型对只包含rfm数据集训练
km.fit(data_rfm)
print(km.labels_)
r1 = pd.Series(km.labels_).value_counts()
r2 = pd.DataFrame(km.cluster_centers_)
rr = pd.concat([r2, r1], axis=1)
rr = pd.DataFrame(ss.fit_transform(rr))
rr.columns = list(data_rfm.columns) + ['counts']
print(rr)


# 利用雷达图对模型学习出的 5 个群体的特征进行可视化分析。


def radar_factory(num_vars, frame='circle'):
    # 计算得到evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # 使用1条线段连接指定点
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 旋转绘图，使第一个轴位于顶部
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """覆盖填充，以便默认情况下关闭该行"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """覆盖填充，以便默认情况下关闭该行"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: x[0], y[0] 处的标记加倍
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # 轴必须以（0.5，0.5）为中心并且半径为0.5
            # 在轴坐标中。
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type 必须是'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon 给出以1为中心的半径为1的多边形
                # （0，0），但我们希望以（0.5，
                #   0.5）的坐标轴。
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# LCRFM模型做图
N = num_clusters
theta = radar_factory(N, frame='polygon')
data = r.to_numpy()
fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                       subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
# 去掉最后一列
case_data = data[:, :-1]
# 设置纵坐标不可见
ax.get_yaxis().set_visible(False)
# 图片标题
title = "Radar Chart for Different Means"
ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
             horizontalalignment='center', verticalalignment='center')
for d in case_data:
    # 画边
    ax.plot(theta, d)
    # 填充颜色
    ax.fill(theta, d, alpha=0.05)
# 设置纵坐标名称
ax.set_varlabels(features)
# 添加图例
labels = ["CustomerCluster_" + str(i) for i in range(1, 6)]
legend = ax.legend(labels, loc=(0.9, .75), labelspacing=0.1)
plt.show()

# 对RFM模型做图
theta = radar_factory(3, frame='polygon')
data = rr.to_numpy()
fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                       subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
# 去掉最后一列
case_data = data[:, :-1]
# 设置纵坐标不可见
ax.get_yaxis().set_visible(False)
# 图片标题
title = "Radar Chart for Different Means"
ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
             horizontalalignment='center', verticalalignment='center')
for d in case_data:
    # 画边
    ax.plot(theta, d)
    # 填充颜色
    ax.fill(theta, d, alpha=0.05)
# 设置纵坐标名称
ax.set_varlabels(['R', 'F', 'M'])
# 添加图例
labels = ["CustomerCluster_" + str(i) for i in range(1, 6)]
legend = ax.legend(labels, loc=(0.9, .75), labelspacing=0.1)
plt.show()

# 用DBSCAN模型对LCRFM特征进行计算
from sklearn.cluster import DBSCAN

# db = DBSCAN(eps=10,min_samples=2).fit(data_db)

# Kagging debug
db = DBSCAN(eps=10, min_samples=2).fit(data_db.sample(10000))

DBSCAN_labels = db.labels_
print(DBSCAN_labels)
