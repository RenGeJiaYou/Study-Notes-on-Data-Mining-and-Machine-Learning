"""
Q2. 西瓜数据集有17个样本，
有两个特征（X1:密度，X2:含糖率），
最后一列是类別，是否是好瓜（是:1，否:-1)，
使用SVM算法，画出决策边界，并且计算错误率。
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sys.path.append('..')  # ..表示上级目录,将上级目录添加到 python 解释器查找 包/模块 路径的列表中

# pandas 读取文本数据
data = pd.read_table('watermelon30a.txt', delimiter=',')  # 按逗号分隔行内每列;默认按换行符分隔每行

x = pd.DataFrame({'密度': data['密度'], '含糖率': data['含糖率']})  # 用 Series 字典对象生成 DataFrame, key 将作为列名,而 value 填充到数组
x = x.values.tolist()  # 只取列数据而去除列名,并转为 Python list 对象

encoder = LabelEncoder()  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
y = encoder.fit_transform(data['好瓜']).tolist()  # '好瓜'属性列只有'是'和'否'两个取值,因此 y 只有[0,1]两个数据
x, y = np.array(x), np.array(y)

fig = plt.figure(figsize=[15, 10])  # 生成图形对象
# ------------------------------------------------------------------------------------------------
fig.add_subplot(1, 2, 1)
clf = SVC(kernel='poly', gamma=2, C=10)  # ※ 多项式核函数
clf.fit(x, y)  # ※ 开始拟合. x:特征向量; y:目标向量
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow', s=40)  # 要求绘制散点图. x[:, 0], x[:, 1]表示两列特征值.s 决定展示每个数据点的大小
ax = plt.gca()  # get current axis :获取当前轴,即 Axes 对象
xLim = ax.get_xlim()  # 返回 x 轴视图限制
yLim = ax.get_ylim()  # 返回 y 轴视图限制
xx = np.linspace(xLim[0], xLim[1], 30)  # 在该上下限内划分30个左闭右闭的点,实际分作 29 个均等的段.返回这些点组成的列表
yy = np.linspace(yLim[0], yLim[1], 30)
yy, xx = np.meshgrid(yy,
                     xx)  # 将两个数组复制拓展为两个矩阵(),长度为 m 的向量拓展至 n 倍;长度为 n 的向量拓展至 m 倍.两矩阵中每一对同位元素就是网格的[x,y] https://www.cnblogs.com/lemonbit/p/7593898.html
xy = np.vstack([xx.ravel(), yy.ravel()]).T  # ravel() 将多维数组拼贴成一维 -> 将两个数组拼为 2 行 -> 转置为两列
z = clf.decision_function(xy).reshape(xx.shape)
plt.contour(xx, yy, z, colors='g', levels=[-1, 0, 1], alpha=0.3,
            linestyles=['--', '-', '--'])  # colors 决定颜色; alpha 决定线条颜色的浓淡
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
            linewidths=1, facecolors='none', edgecolors='k', cmap='rainbow')  # 绘制支持向量
plt.title('poly')
# ------------------------------------------------------------------------------------------------
fig.add_subplot(1, 2, 2)
clf = SVC(kernel='linear', gamma=2, C=1)  # ※ 线性核函数
clf.fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow', s=50)
ax = plt.gca()
xLim = ax.get_xlim()
yLim = ax.get_ylim()
xx = np.linspace(xLim[0], xLim[1], 30)
yy = np.linspace(yLim[0], yLim[1], 30)
yy, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
z = clf.decision_function(xy).reshape(xx.shape)
plt.contour(xx, yy, z, colors='g', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
            linewidths=1, facecolors='none', edgecolors='k', cmap='rainbow')
plt.title('linear')
plt.show()

"""
https://jackcui.blog.csdn.net/article/details/78158354
"""