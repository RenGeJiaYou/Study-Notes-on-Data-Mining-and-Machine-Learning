import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# 通过这种方式实现中文输出
zhfontpath = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf")

x = np.arange(1, 11)  # 返回[1,11)的等差数列
y = x ** 3 - 5  # 函数 y(x)

plt.title("demo", fontproperties=zhfontpath)
plt.xlabel("X轴", fontproperties=zhfontpath)
plt.ylabel("Y轴", fontproperties=zhfontpath)

plt.plot(x, y, 'vr')
plt.show()
##########################################
y1 = 3 * (x ** 2) + 2
y2 = -5 * (x ** 2) - 4
# 1
plt.subplot(1, 2, 1)  # 1行2列第1个
plt.plot(x, y1, "-y")
plt.title("A")
# 2
plt.subplot(1, 2, 2)  # 1行2列第2个
plt.plot(x, y2, ".r")
plt.title("B")
plt.show()
"""
作为线性图的替代，可以通过向 plot() 函数添加格式字符串来显示离散值。 可以使用以下格式化字符。
字符	描述
'-'	        实线样式
'--'	    短横线样式
'-.'	    点划线样式
':'	        虚线样式
'.'	        点标记
','	        像素标记
'o'	        圆标记
'v'	        倒三角标记
'^'	        正三角标记
'&lt;'	    左三角标记
'&gt;'	    右三角标记
'1'	        下箭头标记
'2'	        上箭头标记
'3'	        左箭头标记
'4'	        右箭头标记
's'	        正方形标记
'p'	        五边形标记
'*'	        星形标记
'h'	        六边形标记 1
'H'	        六边形标记 2
'+'	        加号标记
'x'	        X 标记
'D'	        菱形标记
'd'	        窄菱形标记
'&#124;'	竖直线标记
'_'	        水平线标记

以下是颜色的缩写：

字符	        颜色
'b'	        蓝色
'g'	        绿色
'r'	        红色
'c'	        青色
'm'	        品红色
'y'	        黄色
'k'	        黑色
'w'	        白色
"""