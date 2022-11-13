import matplotlib
from matplotlib import pyplot as plt
from shannon import createDataSet
from decision_tree import createTree

zhFontPath = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf")  # 引入中文

"""
树的字符串表达形式：
{'纹理': {0: 0, 1: {'触感': {0: 0, 1: 1}}, 2: {'根蒂': {0: 1, 1: {'色泽': {1: 1, 2: {'触感': {0: 1, 1: 0}}}}, 2: 0}}}}
"""


# 函数说明:获取叶节点的数量，才能确定 figure 的 X 轴长度。
#
# Parameters:
#     myTree - 决策树
# Returns:
#     numLeafs - 决策树的叶子结点的数目
def getNumLeafs(myTree):  # 运行1000次的时间为：0.037899017333984375 s
    numLeafs = 0  # 初始化叶子
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 函数说明:获取决策树的层数,才能确定 figure 的 Y 轴长度。寻找'{'所及的最大深度，即为树层数
#
# Parameters:
#     myTree - 决策树
# Returns:
#     maxDepth - 决策树的层数
def getTreeDepth(myTree):  # 运行1000次的时间为：0.0438 s
    maxDepth = 0  # 初始化决策树深度
    firstStr = next(iter(
        myTree))  # python3中 myTree.keys()返回的是dict_keys,而非 list ,所以不能使用 myTree.keys()[0] 的方法获取 dict 的 key 名，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth  # 更新层数
    return maxDepth


# 树的配置参数
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 函数说明:调用annotate() ，完成实际的绘图工作
# Parameters:
#     nodeTxt  - 结点文本
#     centerPt - 箭头位置
#     parentPt - 箭尾位置
#     nodeType - 箭头样式
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=zhFontPath)


# 函数说明:在父子节点间的中点填充文本信息（即图中的0，1，2等 表示按照不同的特征值划分的子类）
# Parameters:
#     centerPt   - 箭头位置
#     parentPt   - 箭尾位置
#     txtString  - 箭头中点的标注文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 确定 x 轴中点
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 确定 y 轴中点
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)  # 在父子结点的中点处标注文本


# 函数说明:完成绘制工作.调用 plotMidText() 绘制中点标注信息
# Parameters:
#     myTree   - dict 类型的树
#     parentPt - 箭尾位置
#     nodeTxt  - 箭头中点的标注文本
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 计算图片的宽
    depth = getTreeDepth(myTree)  # 计算图片的高

    firstStr = next(iter(myTree))  # 获取 key
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 按照叶节点数量，将 x 轴分为等大的区域
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)

    secondDict = myTree[firstStr]  # 获取 value
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 当前结点 y 轴位置应该在父节点之下
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # 非叶节点
            plotTree(secondDict[key], cntrPt,
                     str(key))  # 深度遍历子结点，完成绘制工作。str(key)指的是当前特征值，可以在 shannon.py 中找到这些特征值对应的物理特征
        else:  # 叶节点，打印
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  #
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 函数说明: ※核心函数。调用 plotTree()
# Parameters:
#     myTree   - dict 类型的树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    # plotTree是即将调用的函数，也是一个对象，可以通过点运算符（.）添加自己的变量
    plotTree.totalW = float(getNumLeafs(inTree))  # plotTree.totalW 存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # plotTree.totalW 存储树的深度
    plotTree.xOff = -0.5 / plotTree.totalW  # xoff、 yoff 追踪已经绘制的结点位置。x轴和y轴范围都是[0.0 , 1.0]
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')

    plt.show()


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    createPlot(myTree)