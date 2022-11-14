from shannon import createDataSet, calcShannonEnt

"""
功能描述：返回数据集D中，根据[某个特征项的每一个枚举值]划分的不同 Di 子集

Params:
    dataSet:    原有整体数据集 D，一个二维数组
    axis:       指定来划分 D 的特征项，一个下标
    value:      指定特征项的枚举值，一个数字，本例中取值仅在{0，1，2}
Returns:
    retDataSet 按指定特征划分的数据集Di 。该特征项能有几个不同的取值，i 就是几
"""
def spliceDataSet(dataSet, axis, value):
    retDataSet = []  # 待返回的子集,也是一个二维数组.行：满足 axis == value 的所有记录；列：去除 axis 特征行

    for featVec in dataSet:  # featVec 是原集 D 的一条记录,feature vector
        if featVec[axis] == value:  # 将符合指定特征值的记录 去掉该特征项
            reduceVec = featVec[:axis]
            reduceVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceVec)

    return retDataSet


"""
功能描述:选出最优特征

Params:
    dataSet:        原有整体数据集 D，一个二维数组
Returns:
    bestFeature:    信息增益最大的(最优)特征的索引值
"""
def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征项数量
    baseEnt = calcShannonEnt(dataSet)  # 计算整体的香农熵 H（D）
    bestInfoGain = 0.0  # 信息增益 g(D,A),将轮流计算每个特征下的增益信息，并保留最佳增益值
    bestFeature = -1  # 保留最佳增益的特征项的下标（ bestInfoGain 和 bestFeature 将采用贪心算法，迭代时更新最大值）

    # todo:把 python 几种迭代方式好好看看
    for i in range(numFeatures):  # 遍历所有特征项,即求H(D|A1)、H(D|A2)、H(D|A3)...
        # featList = []
        # for featVec in dataSet:
        #     featList.append(featVec[i])

        featList = [featVec[i] for featVec in dataSet]  # ←使用列表生成式，将 3 行简化为 1 行：
        uniqueFeatList = set(featList)  # 只保留互不相同的项
        curConEnt = 0.0  # 当前特征的经验条件熵

        for value in uniqueFeatList:  # 根据set()长度，遍历 Di。计算各个 p(xi) * H(Di) 并求期望之和
            subDataSet = spliceDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))  # 计算当前特征项下的各个不同特征值：|Di| / |D|
            curConEnt += prob * calcShannonEnt(subDataSet)  # 计算当前 H（D|A）

        curInfoGain = baseEnt - curConEnt  # 计算出当前特征项的信息增益

        # print("特征 %s 的增益为%.3f" % (labels[i], curInfoGain))

        # 有更大值时更新 best 变量
        if curInfoGain > bestInfoGain:
            bestInfoGain = curInfoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值

    return bestFeature


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print("最优特征索引值:" + labels[chooseBestFeature(dataSet)])