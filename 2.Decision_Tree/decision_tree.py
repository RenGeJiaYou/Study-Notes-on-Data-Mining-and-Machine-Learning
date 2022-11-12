import operator

from feature import chooseBestFeature, spliceDataSet
from shannon import createDataSet

"""
函数说明:统计classList中出现此处最多的元素(类标签)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)

"""


def majorityCnt(classList):
    classCount = {}  # 通过一个 dict 找出当前子集中占比最大的特征
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount += 1
    # 1.sort() 只能排序 list, sorted() 可以排任何可排的数据结构，包括 dict、set
    # 2.iteritems() 返回的不是原 dict 的列表化拷贝（一次性复制所有对象），而是原 dict 的迭代器对象（一次调用只会复制一个对象）。优点是省内存。
    # 3.operator.getitem(a,b)返回 a 中索引为 b 的值。
    sortedClassCount = sorted(classCount.iteritems(), key=operator.getitem(1), reverse=True)

    return sortedClassCount[0][0]  # 返回最多的特征的名字


"""
函数说明:创建决策树
    递归地创建决策树，终止条件为：
        1.当前子集所有元素的标签一致，返回该标签
        2.当前已划分完了所有的特征，返回数量最多的那个标签

Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树

"""


def createTree(dataSet, labels):
    classList = [item[-1] for item in dataSet]  # 取目标变量标签
    if classList.count(classList[0]) == len(classList):  # 终止条件①：判定当前最多的特征项
        return classList[0]
    if len(dataSet[0]) == 1:  # 终止条件②：划分完所有特征
        return majorityCnt(classList)

    bestFeat = chooseBestFeature(dataSet)  # 获得当前最佳特征项的下标
    bestFeatLabel = labels[bestFeat]  # 获得当前最佳特征项
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 当前已找到最佳特征，删除之

    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueFeatValues = set(featValues)  # 去重
    for value in uniqueFeatValues:  # 根据不重复的属性值创建子节点
        subLabels = labels[:]  # 列表是传址调用，不复制一份，直接传本体的话，内层递归会意外修改原数组
        myTree[bestFeatLabel][value] = createTree(spliceDataSet(dataSet, bestFeat, value), subLabels)

    return myTree  # 从递归返回


"""
↑ 关于 myTree[bestFeatLabel][value] 的写法含义，实际是一个嵌套 dict 
myTree 是一个 dict
其内部的一个键 bestFeatLabel 的值也是一个 dict
 myTree[bestFeatLabel]              -> 表示取 myTree 内 bestFeatLabel 的值，这个值也是一个 dict 
 myTree[bestFeatLabel][value]       -> 表示修改这个子 dict ：新增一个键值对，key 为遍历的属性值, value 为 [根据当前最优特征的其中一个属性值划分的] 子集
"""

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    print(myTree)
    print(type(str(myTree)))