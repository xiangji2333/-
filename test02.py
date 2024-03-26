import os
#数据读取
import pandas as pd
import torch

# 创建文件夹，创建文件house_tiny.csv
os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
# csv：CSV文件以纯文本形式保存表格数据，其中每行表示表格中的一行，而每个字段则通过逗号（或其他分隔符，如制表符）来分隔
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 打印表格
data = pd.read_csv(data_file)
print(data)

# 处理缺失数据，常见的如插值、删除，这里使用的是插值
# 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 它将分类变量转换为哑变量（dummy variables），并且对缺失值（NaN）也进行编码。dummy_na=True：这表示在进行独热编码时，会为缺失值（NaN）创建一个额外的哑变量列
# 独热编码是将分类变量转换为二进制的过程，其中每个可能的类别值都被表示为一个新的二进制特征（列），并且在每个样本中，只有一个特征是激活的（1），其他特征都是非激活的（0）。这种编码方式在机器学习中经常用于处理分类变量，因为它可以使得分类变量的值对于算法更易于理解和处理。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 在进行转换后inputs和outputs所有条目都是数值类型，可以将他们转换为张量格式
x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x,y)

