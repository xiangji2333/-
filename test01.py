import torch
import numpy
# 张量表示一个数值组成的数组，这个数组可以有多个维度
x = torch.arange(12)
print(x)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 输出张量的形状
print(x.shape)
# torch.Size([12])

# 输出张量的元素总数
print(x.numel())
# 12

# 修改张量元素数量和元素值
x = x.reshape(3,4)
print(x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(x.shape)
# torch.Size([3, 4])

# 使用全0、全1或其他常量或从特定分布中随机采样的数字
x = torch.zeros((2, 3, 4))
print(x)
# tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#
#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

x = torch.ones((2, 3, 4))
print(x)

# 提供包含数值的python列表（或嵌套列表）来为张量中的每个元素赋值
x = torch.tensor([[1, 2, 3],[3, 2, 3]])
print(x)
# tensor([[1, 2, 3],
#         [3, 2, 3]])

# 常见标准算数运算符可以被升级为按蒜素运算
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y,x-y,x*y,x/y,x**y)
# tensor([ 3.,  4.,  6., 10.]) tensor([-1.,  0.,  2.,  6.]) tensor([ 2.,  4.,  8., 16.]) tensor([0.5000, 1.0000, 2.0000, 4.0000]) tensor([ 1.,  4., 16., 64.])

# 将多个张量连接在一起
x = torch.arange(6,dtype=torch.float32).reshape((2,3))
y = torch.tensor([[0,3,3],[6,6,6]])
# (x,y)表示按元组输入
print(torch.cat((x,y),dim=0)) # 按第零维合并，即按行合并
print(torch.cat((x,y),dim=1)) # 按第一维合并，即按列合并
# tensor([[0., 1., 2.],
#         [3., 4., 5.],
#         [0., 3., 3.],
#         [6., 6., 6.]])
# tensor([[0., 1., 2., 0., 3., 3.],
#         [3., 4., 5., 6., 6., 6.]])

# 按逻辑运算符构建张量
print(x==y)
# tensor([[ True, False, False],
#         [False, False, False]])

# 对张量求和会产生只有一个元素的张量
print(x.sum())
# tensor(15.)

# 即使形状不同，可以调用广播机制来执行元素操作
# 广播机制前提是两个张量维度相同，同时两个变量有都有一个维度为1或者有一个维度相同只需要有一个维度为1即可
x = torch.arange(6).reshape((3,2))
y = torch.arange(2).reshape((1,2))
print(x+y)
# 即自动按行复制和自动按列复制
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])

# [-1]选择最后一个元素，[1,3]访问二三元素，区间左闭右开
# x[1,2]访问第二行第三个，x[0:2,:]访问一二行的所有列

# python进行一些操作会导致为新结果分配内存，当访问较大数据时需要避免导致内存问题
before = id(y)# id()与c++ *y相同，返回y所在内存的十进制标识
y = y + x # 这个操作本质上是产生了新的y
print(id(y)==before)
# False

# 原地操作
z = torch.zeros_like(y) # 创建一个和y形状相同的全零数组
print('id(z):',id(z))
z[:]=x+y
print('id(z):',id(z))
# z的地址不变
# 如果在后续操作没有重复使用y，可以通过以下操作来减少内存开销
y[:] += x # y += x
# 这样也可以避免分配内存，本质上是直接对元素进行改写而不改变地址

# 将转换numpy和张量
a = x.numpy()
b = torch.tensor(a)
print(type(a),type(b))
# <class 'numpy.ndarray'> <class 'torch.Tensor'>

#将大小为1的张量转换为python标量
a = torch.tensor([3.5])
#item为取出元素值同时不改变元素类型
print(a,a.item(),float(a),int(a))
# tensor([3.5000]) 3.5 3.5 3



