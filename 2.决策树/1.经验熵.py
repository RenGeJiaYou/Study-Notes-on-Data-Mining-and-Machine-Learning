from math import log

"""
经验熵：每一条记录拥有若干个「特征」项，而已经得知的「决策」变量来源于现实经验

在编写代码之前，我们先对数据集进行属性标注。

年龄：         0代表青年，1代表中年，2代表老年；
有工作：       0代表否，1代表是；
有自己的房子：  0代表否，1代表是；
信贷情况：      0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。
"""

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]

    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性


"""
香农熵计算函数
Params:
    dataSet 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


def calcShannonEnt(dataSet):
    numEntiries = len(dataSet)

    # 统计不同标签各自的总量
    labelCounts = {}  # 创建一个字典，key:标签名，value:该标签的样本数
    for featVec in dataSet:  # 遍历每一条记录
        currentLabel = featVec[-1]  # 最后一列是用于分类的标签值
        if currentLabel not in labelCounts.keys():  # 如果当前标签还没存进去，就新建一个空的k-v
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 否则+1

    # 开始统计香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntiries  # 计算每个分类的样本被选中的概率 p(xi)
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))  # 0.9709505944546686