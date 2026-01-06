import torch
import torch.nn as nn

class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()  # Sigmoid 激活函数的实例
        self.e_lambda = e_lambda  # 正则化项的参数 lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)  # 返回表示模型的字符串，包括 lambda 参数的值
        return s

    @staticmethod
    def get_module_name():
        return "simam"  # 返回模块的名称，为 "simam"

    def forward(self, x):
        # 前向传播函数，对输入 x 进行处理并返回结果

        b, c, h, w = x.size()    # 获取输入张量的形状信息
        n          = w * h - 1   # 计算总像素数减1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # 计算每个像素与均值的差的平方
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 计算激活函数的输入 y，使用 SimAM 公式：x_minus_mu_square / (4 * (均值方差 + 正则化项)) + 0.5

        return x * self.activaton(y)  # 返回经过激活函数后的结果与输入张量 x 的乘积
