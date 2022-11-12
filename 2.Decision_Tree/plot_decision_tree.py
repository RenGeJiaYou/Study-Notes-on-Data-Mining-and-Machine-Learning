import matplotlib
from matplotlib import pyplot as plt

zh_font_path = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf")  # 引入中文

# 树的字符串表达形式：
# {'纹理': {0: 0, 1: {'触感': {0: 0, 1: 1}}, 2: {'根蒂': {0: 1, 1: {'色泽': {1: 1, 2: {'触感': {0: 1, 1: 0}}}}, 2: 0}}}}

"""
函数说明:获取叶节点的数量，才能确定 figure 的 X 轴长度

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""


def getNumLeafs(myTree):
    pass


"""
函数说明:获取决策树的层数,才能确定 figure 的 Y 轴长度。寻找'{'所及的最大深度，即为树层数

Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""


def getTreeDepth(myTree):
    num = max_depth = 0
    str_tree = str(myTree)  # 将字典类型转化为字符串，便于搜索
    for i, v in enumerate(str_tree):
        if v == '{':
            num += 1
            if num > max_depth:
                max_depth = num
        elif v == '}':
            num -= 1
    return max_depth


# 树的配置参数
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 调用annotate() ，完成实际的绘图工作
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=zh_font_path)


def createPlot():
    fig = plt.figure(1, facecolor='white')  # 当前 figure 的唯一标识号，背景色.返回值是一个 Figure 图形对象
    fig.clf()  # 清除当前图形
    createPlot.ax1 = plt.subplot(111, frameon=False)  # ax1 定义了一个绘图区。而且是全局变量，因此 createPlot.ax1.annotate(...) 可以直接调用
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


if __name__ == '__main__':
    createPlot()
    depth = getTreeDepth(
        "{'纹理': {0: 0, 1: {'触感': {0: 0, 1: 1}}, 2: {'根蒂': {0: 1, 1: {'色泽': {1: 1, 2: {'触感': {0: 1, 1: 0}}}}, 2: 0}}}}")
    print(depth)