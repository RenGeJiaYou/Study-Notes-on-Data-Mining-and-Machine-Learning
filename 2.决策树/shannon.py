from math import log

"""
经验熵：每一条记录拥有若干个「特征」项，而已经得知的「决策」变量来源于现实经验。熵越大，不确定性就越大

属性标注：
特征变量：
    色泽：0代表浅白  1代表青绿  2代表乌黑
    根蒂：0代表蜷缩  1代表稍蜷  2代表硬挺
    敲声：0代表沉闷  1代表浊响  2代表清脆
    纹理：0代表模糊  1代表稍糊  2代表清晰
    脐部：0代表凹陷  1代表稍凹  2代表平坦
    触感：0代表硬滑  1代表软粘  

目标变量：
    好瓜：0代表否    1代表是
    
※ 注意：标注的属性值具体是多少对结果毫无影响，只是做了不同值的区分。决定熵值的是每个不同的属性值在总体中的比例
"""

"""
功能描述：返回数据集和标签集

Params:
    无
Returns:
    数据集，标签集
"""
def createDataSet():
    dataSet = [[1, 0, 1, 2, 0, 0, 1],
               [2, 0, 0, 2, 0, 0, 1],
               [2, 0, 1, 2, 0, 0, 1],
               [1, 0, 0, 2, 0, 0, 1],
               [0, 0, 1, 2, 0, 0, 1],
               [1, 1, 1, 2, 1, 1, 1],
               [2, 1, 1, 1, 1, 1, 1],
               [2, 1, 1, 2, 1, 0, 1],
               [2, 1, 0, 1, 1, 0, 0],
               [1, 2, 2, 2, 2, 1, 0],
               [0, 2, 2, 0, 2, 0, 0],
               [0, 0, 1, 0, 2, 1, 0],
               [1, 1, 1, 1, 0, 0, 0],
               [0, 1, 0, 1, 0, 0, 0],
               [2, 1, 1, 2, 1, 1, 0],
               [0, 0, 1, 0, 2, 0, 0],
               [1, 0, 0, 1, 1, 0, 0]]  # 机器学习 西瓜数据集

    labels = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "好瓜"]  # 标签汇总
    return dataSet, labels


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
    print(calcShannonEnt(dataSet))