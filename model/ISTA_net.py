import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary


class BasicBlock1D(torch.nn.Module):
    def __init__(self):
        super(BasicBlock1D, self).__init__()
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter，并且会向宿主模型注册该参数，成为一部分。
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))  # 步长
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))  # 软阈值

        # _in_channels_, _out_channels_, _kernel_size_
        # init.xavier_normal_()：用正态分布填充输入张量，保证每一层输入输出方差相同
        # 卷积核的形状通常是 (out_channels, in_channels, kernel_height, kernel_width)
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        # r的更新不变: r(k) = x(k−1) − ρΦT(Φx(k−1)−y)
        # 在第k−1次迭代中，x(k−1)向保真项的梯度方向移动，步长为ρ，结果为r(k)；ΦT(Φx(k−1)−y)为保真项的梯度
        x = x - self.lambda_step * torch.matmul(x, PhiTPhi)  # 使用matmul处理1D数据矩阵乘法，x为行向量则x左乘PhiTPhi

        x = x + self.lambda_step * PhiTb
        # 为了使用conv1d，input.shape=[batch_size, in_channels, length] 在这之前为了矩阵运算，x.shape=[batch_size, length]
        x_input = x.view(-1, 1, x.size(-1))

        # x的更新：x(k) = argx min1/2||x−r(k)||2 + λ||Ψx||1
        # 线性变换Ψx用非线性变换的卷积网络ϝ来替换，省去人工设计线性变换矩阵Ψ的过程，模型容量变大了，且每一层网络的权重可以不同。
        x = F.conv1d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv1d(x, self.conv2_forward, padding=1)

        # 非线性变换：使用软阈值函数，relu创建了一个稀疏的特征图，只保留那些大于soft_thr的值；sign将结果符号化，数值稳定性或网络收敛性？？？
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        # 对称结构ϝ_：互为逆变换的特性会通过损失函数的设计来实现
        x = F.conv1d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv1d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, x.size(-1))

        # 计算对称结构loss：ϝ_(ϝ(x))-x
        x = F.conv1d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)  # 为了确保 x_est 是非负的
        x_est = F.conv1d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input  # 度量还原信号与原始信号之间的差异，通常用于损失函数的计算

        return [x_pred, symloss]


class ISTANet1D(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet1D, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo  # 网络层数

        for i in range(LayerNo):
            onelayer.append(BasicBlock1D())

        # 管理模块列表，每个层或模块都可以是一个独立的 nn.Module 对象
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.matmul(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.matmul(Phix, Phi)

        # 初始值x(0)=Qy
        x = torch.matmul(Phix, torch.transpose(Qinit, 0, 1))

        # ---在模型内使用，也需要将张量移动到GPU上---
        PhiTPhi = PhiTPhi.to(torch.device("cuda:0"), dtype=torch.float32)
        PhiTb = PhiTb.to(torch.device("cuda:0"), dtype=torch.float32)
        x = x.to(torch.device("cuda:0"), dtype=torch.float32)

        loss_layers_sym = []

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            loss_layers_sym.append(layer_sym)

        x_final = x

        return [x_final, loss_layers_sym]


if __name__ == "__main__":
    # 创建一个一维信号输入输出都为1的示例模型
    LayerNo = 10000  # 选择网络层数
    model = ISTANet1D(LayerNo)
    print(model)

    x = torch.randn(1, 101)  # 原信号，行向量 (batch_size, 101)
    Phi = torch.randn(46, 101)  # 采样矩阵，行向量为单个采样光谱
    # 行向量y=Phi*x
    Phix = torch.matmul(x, torch.transpose(Phi, 0, 1))  # 采样后的信号，行向量 (1, 46)
    # Q=XY^T(YY^T)^-1
    x = torch.transpose(x, 0, 1)  # 转为列向量，公式使用列向量推导 (101, 1)
    y = torch.matmul(Phi, x)  # 采样后的信号，列向量 (46, 1)
    y_yT = torch.matmul(y, torch.transpose(y, 0, 1))  # (46, 46)
    y_yT_inv = torch.inverse(y_yT)  # (46, 46)
    x_yT = torch.matmul(x, torch.transpose(y, 0, 1))  # (101, 46)
    Qinit = torch.matmul(x_yT, y_yT_inv)  # (101, 46)

    model.to(torch.device("cuda:0"), dtype=torch.float32)
    Phix.to(torch.device("cuda:0"), dtype=torch.float32)
    Phi.to(torch.device("cuda:0"), dtype=torch.float32)
    Qinit.to(torch.device("cuda:0"), dtype=torch.float32)

    [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
    print(x_output.shape)

    layer_num = len(model.fcs)
    print(layer_num)


