import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = fm.FontProperties(fname=font_path, size=10)  # 调整字体大小
# 数据
data = [3.0482065677642822, 2.9238059520721436, 3.843790292739868]

# 设置图形大小
plt.figure(figsize=(10, 2))

# 绘制数据点
# for i, value in enumerate(data):
plt.plot(data[0], 1, 'o', label=f'正常数据-1')
plt.plot(data[1], 1, 'o', label=f'正常数据-2')
plt.plot(data[2], 1, 'o', label=f'缺陷数据')

# 设置 x 轴的范围
plt.xlim(min(data) - 0.5, max(data) + 0.5)

# 去掉 y 轴
plt.yticks([])

# 添加网格线
plt.grid(axis='x')
plt.legend(prop=font_prop)
# 添加图例


# 设置标题
plt.title('输出层内部异常行为与缺陷触发关系',fontproperties=font_prop)

# 显示图形
plt.savefig('plot.png')
