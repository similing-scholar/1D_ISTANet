针对ISTA_net代码的一些思考，记录从代码学习算法的一些笔记，目的是将原码改为自己能用的1维信号压缩感知重构网络，不深入研究数学问题。

> 改完的代码存放仓库：[GitHub仓库地址](https://github.com/similing-scholar/1D_ISTANet.git)、[gitee仓库地址](https://gitee.com/similing-scholar/1D_ISTANet.git)
>参考的贴子和代码：1. [jianzhangcs/ISTA-Net-PyTorch: ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing, CVPR2018 (PyTorch Code) (github.com)](https://github.com/jianzhangcs/ISTA-Net-PyTorch)  2. [【论文阅读笔记 2】ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/397243123)  3.[关于论文《ISTA-Net》的研究心得-CSDN博客](https://blog.csdn.net/qq_42201432/article/details/115263863)  4.[ISTANET FOR CS-MRI 从公式到代码的时间python pytorch。 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/516460564)  5.[笔记（二）——深度学习基本框架（层和块）_深度学习block_SaltyFish_Go的博客-CSDN博客](https://blog.csdn.net/weixin_45169380/article/details/122043686)  6.[[重温经典]深度解读ISTA-Net - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/648333661)

##### 一张原理图：
![ista_phase](https://github.com/jianzhangcs/ISTA-Net-PyTorch/blob/master/Figs/ista_phase.png?raw=true)
结合论文原图看代码，包括的元素有：$x$ 为待恢复原信号（训练的标签），$y$是$x$经过观测矩阵（传感矩阵？）编码的观测值（训练的输入），${x}^{(n)}$是迭代恢复的值（其中${x}^{(0)}$是一个有公式算出的初始化值），每个phase的${r}^{(k)}$（与ISTA公式中一致），对称的网络部分（代替ISTA算法${x}^{(k)}$的迭代）。

##### 俩个迭代公式：
$$ \mathbf{r}^{(k)}=\mathbf{x}^{(k)}−\mathbfρ\mathbf{Φ}^{T}(\mathbfΦ\mathbf{x}^{(k-1)}−\mathbf{y}) $$

$$ \mathbf{x}^{(k)}=\arg\min_{\mathbf{x}}\frac12\|\mathbf{x}-\mathbf{r}^{(k)}\|_2^2+\lambda\|\mathbf{\Psi}\mathbf{x}\|_1 $$

公式${r}^{(k)}$迭代的原码，与ISTA算法一致。
```python
x = x - self.lambda_step * torch.mm(x, PhiTPhi)  # torch.mm二维矩阵乘法  
x = x + self.lambda_step * PhiTb  # b为公式中的y
x_input = x.view(-1, 1, 33, 33)  # 33，33应该是压感之后的观测信号尺寸,这里需要根据实际更改
```
其中步长ρ和软阈值都是可训练参数
```python
self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
```

初始值${x}^{(0)}$的计算 
$$ \mathbf{Q}={XY^{T}(YY^{T})^{-1}} $$
$$ \mathbf{X}^{(0)}={Qy} $$
注意：这个公式是根据x为列向量推导的，实际代码里面需要转置为行向量
```python
if os.path.exists(Qinit_Name):  
    Qinit_data = sio.loadmat(Qinit_Name)  
    Qinit = Qinit_data['Qinit']  
  
else:  
    X_data = Training_labels.transpose()  # 转置为行向量
    Y_data = np.dot(Phi_input, X_data)  # 观测值
    Y_YT = np.dot(Y_data, Y_data.transpose())  
    X_YT = np.dot(X_data, Y_data.transpose())  
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))  
    del X_data, Y_data, X_YT, Y_YT  
    sio.savemat(Qinit_Name, {'Qinit': Qinit})
```

```python
x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
```

${x}^{(k)}$迭代$\mathbf{x}^{(k)}=\tilde{\mathcal{F}}\left(\mathrm{soft}\left(\mathcal{F}\left(\mathbf{r}^{(k)}\right),\theta\right)\right)$分三步：
1. 线性变换$Ψx$用非线性变换的卷积网络$F$来替换，省去人工设计线性变换矩阵（稀疏矩阵？？？）$Ψ$的过程，模型容量变大了，且每一层网络的权重可以不同。
```python
x = F.conv2d(x_input, self.conv1_forward, padding=1)  
x = F.relu(x)  
x_forward = F.conv2d(x, self.conv2_forward, padding=1)
```
2. 非线性变换：使用软阈值函数，relu创建了一个稀疏的特征图，只保留那些大于soft_thr的值；sign将结果符号化，数值稳定性或网络收敛性？？？
```python
x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))  # relu已经实现了保留大于0的功能，还需要sign？？？
```
3. $\tilde{\mathcal{F}}$过程：互为逆变换的特性会通过损失函数的设计来实现
```python
x = F.conv1d(x, self.conv1_backward, padding=1)  
x = F.relu(x)  
x_backward = F.conv1d(x, self.conv2_backward, padding=1)
```

源码中卷积层采用自定义写法
```python
self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))  
self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))  
self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))  
self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
```

```python
x = F.conv2d(x_input, self.conv1_forward, padding=1)  
x = F.relu(x)  
x_forward = F.conv2d(x, self.conv2_forward, padding=1)
```
需要注意这个写法中，卷积核的形状为(out_channels, in_channels, kernel_height, kernel_width)，参考官方文档区别F.conv2d和nn.conv2d[torch.nn.functional.conv2d — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)[Conv2d — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

对称结构的loss计算：
```python
x = F.conv1d(x_forward, self.conv1_backward)  
x = F.relu(x)  # 为了确保 x_est 是非负的  
x_est = F.conv1d(x, self.conv2_backward)  
symloss = x_est - x_input 
```


##### 一维信号的代码
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class BasicBlock1D(torch.nn.Module):
    def __init__(self):
        super(BasicBlock1D, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))  # 步长
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))  # 软阈值

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.matmul(x, PhiTPhi)  # 使用matmul处理1D数据矩阵乘法，x为行向量则x左乘PhiTPhi
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, x.size(-1))  # 为了使用conv1d，input.shape=[batch_size, in_channels, length]

        x = F.conv1d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv1d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv1d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv1d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, x.size(-1))

        x = F.conv1d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv1d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


class ISTANet1D(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet1D, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo  # 网络层数

        for i in range(LayerNo):
            onelayer.append(BasicBlock1D())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.matmul(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.matmul(Phix, Phi)

        x = torch.matmul(Phix, torch.transpose(Qinit, 0, 1))

        # 在模型内使用，也需要将张量移动到GPU上
        PhiTPhi = PhiTPhi.to(torch.device("cuda:0"), dtype=torch.float32)
        PhiTb = PhiTb.to(torch.device("cuda:0"), dtype=torch.float32)
        x = x.to(torch.device("cuda:0"), dtype=torch.float32)

        layers_sym = []

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
```