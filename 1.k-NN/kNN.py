from numpy import *  # ’import numpy‘也可以，不过之后调用就要写numpy.array(...)
import operator

"""
 Describe: 创建数据集和标签
 Params:
    无
 Returns:
    group:  数据集
    labels: 标签集
"""
def createDataSet():
    group = array([[1, 101], [5, 89], [108, 5], [115, 8]])  # 四组二维特征
    labels = ['爱情片', '爱情片', '动作片', '动作片']  # 四组特征的标签
    return group, labels


# k-NN 分类算法
def classify0(inX,dataSet,labels,k):


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()

    # 打印数据集
    print(group)
    print(labels)