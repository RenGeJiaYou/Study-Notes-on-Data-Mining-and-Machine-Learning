import matplotlib
from matplotlib import pyplot as plt

# 初步了解pyplot的使用

zh_font_path = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf")  # 引入中文

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
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # 清除当前图形
    createPlot.ax1 = plt.subplot(111, frameon=False)  # ax1 定义了一个绘图区。而且是全局变量，因此 createPlot.ax1.annotate(...) 可以直接调用
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


if __name__ == '__main__':
    createPlot()