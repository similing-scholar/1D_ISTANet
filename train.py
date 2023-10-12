import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import wandb

from model.load_dataset_interface import Basic1DDataset, load_dataset
from utils.utils import load_h5, normalize_data
from model.ISTA_net import ISTANet1D
from model.train_model_interface import KerasModel
from utils.plot_interface import plot_metric


# ------------------------------------ step 0/5 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')

# ------------------------------------ step 1/5 : 加载数据------------------------------------
# 定义数据集文件路径
input_file = './dataset/1Dy[10000,42].h5'  # y观测值
mask_file = './dataset/mask[42,101].h5'
Qinit_file = './dataset/Q_ISTAnet[101,42].h5'
label_file = 'dataset/1Dspectrum[10000,101].h5'  # 原始信号x
# 创建数据集
my_dataset = Basic1DDataset(input_file=input_file, label_file=label_file, encoder='train')

# 定义划分比例和batch大小
train_ratio = 0.8
val_ratio = 0.2
batch_size = 128
# 加载数据集(观测值y和label)
train_loader, val_loader, test_loader = load_dataset(my_dataset, train_ratio, val_ratio, batch_size)
# 将Phi和Qinit转换为张量并归一化
Phi = torch.tensor(normalize_data(load_h5(mask_file)))
Qinit = torch.tensor(normalize_data(load_h5(Qinit_file)))
Phi = torch.tensor(load_h5(mask_file))
Qinit = torch.tensor(load_h5(Qinit_file))

# ------------------------------------ step 2/5 : 创建网络------------------------------------
LayerNo = 6  # 选择网络层数
net = ISTANet1D(LayerNo)  # 加载一个网络
net.to(device, dtype)  # 将网络移动到gpu/cpu，可以考虑.cuda()!!!
# 是否使用预训练权重
pre_training = True
pre_training_data = 'logs/2023_10_12_10_43_14_Lay6_checkpoint.pt'
if pre_training_data:
    net.load_state_dict(torch.load(pre_training_data))  # 加载预训练权重

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
model = KerasModel(device=device, dtype=dtype, net=net,
                   Phi=Phi, Qinit=Qinit,
                   loss_fn=nn.MSELoss(),
                   optimizer=optim.Adam(net.parameters(), lr=1e-5)
                   )

# ------------------------------------ step 4/5 : 训练模型并保存 --------------------------------------------------
time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
ckpt_path = f'logs/{time}_Lay{LayerNo}_checkpoint.pt'
model.fit(train_data=train_loader,
          val_data=val_loader,
          epochs=1000,
          ckpt_path=ckpt_path,
          patience=15,
          monitor='val_loss',
          mode='min')  # 监控的指标为验证集上的损失函数，模式为最小化

# ------------------------------------ step 5/5 : 绘制训练曲线评估模型 --------------------------------------------------
df_history = pd.DataFrame(model.history)  # 将训练过程中的损失和指标数据保存为DataFrame格式

fig = plot_metric(df_history, "loss")
fig_path = f'logs/{time}_Lay{LayerNo}_history.png'
fig.savefig(fig_path, dpi=300)
