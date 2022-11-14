"""
Q2. 西瓜数据集有17个样本，
有两个特征（X1:密度，X2:含糖率），
最后一列是类別，是否是好瓜（是:1，否:-1)，
使用SVM算法，画出决策边界，并且计算错误率。
"""
import sys
from decisionTree.shannon import createDataSet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sys.path.append('..')  # ..表示上级目录,将上级目录添加到 python 解释器查找 包/模块 路径的列表中

# 获取西瓜数据集
dataSet, labels = createDataSet()

# pandas 读取文本数据
data = pd.read_table('watermelon30a.txt', delimiter=',')

x = pd.DataFrame({'密度': data['密度'], '含糖率': data['含糖率']})
x = x.values.tolist()

encoder = LabelEncoder()
y = encoder.fit_transform(data['好瓜']).tolist()
x, y = np.array(x), np.array(y)

fig = plt.figure(figsize=[15, 10])
fig.add_subplot(1, 2, 1)
clf = SVC(kernel='poly', gamma=2, C=1000)    # ※ 多项式核函数
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
plt.title('rbf')

fig.add_subplot(1, 2, 2)
clf = SVC(kernel='linear', gamma=2, C=1000) # ※ 线性核函数
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