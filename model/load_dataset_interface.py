import numpy as np
from torch.utils.data import Dataset, random_split
import torch
import os
import time

from utils.utils import load_h5, save_h5, normalize_data


class Basic1DDataset(Dataset):
    """
    传入不固定数据的数据集类，用于训练和预测
    """
    def __init__(self, input_file, mask_file=None, Q_file=None, label_file=None, encoder='train'):
        """
        对于训练和验证模型，需要传入Phix(y也是input), label，其中Phi, Qinit通过模型传入
        对于预测模型，只需要传入Phix(y也是input)
        """
        self.encoder = encoder  # encoder的值存储为实例变量

        self.input_data = load_h5(input_file)  # 观测值y input
        # self.Phi = load_h5(mask_file)  # 采样矩阵
        # self.Qinit = load_h5(Q_file)  # 初始值矩阵
        # 根据是否提供label_file来决定是否加载输出数据，label_file为本地变量
        if label_file is not None:
            self.label_data = load_h5(label_file)  # 原信号x 训练的label

    def __len__(self):
        return len(self.input_data)  # 数据集的长度是输入数据的行数

    def __getitem__(self, index):
        # 根据索引获取对应的输入行，并归一化
        input_row = normalize_data(self.input_data[index])
        input_row = self.input_data[index]
        # 将行数据转换为PyTorch张量，在这个网络中有举证运算所以不需要 .unsqueeze(0)增加通道数，在load_dataset中增加通道数
        input_tensor = torch.tensor(input_row, dtype=torch.float32)

        # Phi_tensor = torch.tensor(normalize_data(self.Phi), dtype=torch.float32)  # (46, 101)
        # Qinit_tensor = torch.tensor(normalize_data(self.Qinit), dtype=torch.float32)  # (101, 46)

        if self.encoder == 'train':  # 如果是训练模式，返回输入和标签对
            # 根据索引获取对应的输出行，并归一化
            label_row = normalize_data(self.label_data[index])
            label_row = self.label_data[index]
            label_tensor = torch.tensor(label_row, dtype=torch.float32)
            return input_tensor, label_tensor

        elif self.encoder == 'predict':  # 如果是预测模式，只返回输入
            return input_tensor


def load_dataset(my_dataset, train_ratio, val_ratio, batch_size):
    """加载数据集，划分数据集，返回数据加载器。
    train_ratio + val_ratio ≤ 1，否则给出报错提示
    """
    if train_ratio + val_ratio > 1.0:  # 检查train_ratio和val_ratio的合法性
        raise ValueError("train_ratio + val_ratio 必须小于等于 1.0")

    # 计算划分的样本数量
    total_samples = len(my_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # 使用random_split划分数据集
    train_dataset, val_dataset, test_dataset = random_split(my_dataset,
                                                            [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(0)
                                                            )

    # 只有当模型在GPU上进行训练时，pin_memory=True才会对数据加载效率产生积极影响。
    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)  # 多线程影响效率？？？？？？
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    mask_file = '../dataset/mask[42,101].h5'
    Q_file = '../dataset/Q_ISTAnet[101,42].h5'
    input_file = '../dataset/1Dspectrum[10000,101].h5'
    label_file = '../dataset/1Dy[10000,42].h5'

    # ----------实例化训练数据集----------
    train_dataset = Basic1DDataset(input_file, mask_file, Q_file, label_file)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, 0.8, 0.2, 64)

    # 遍历数据加载器以进行训练
    for input_batch, label_batch in train_loader:
        # 在这里进行模型的训练
        print(input_batch.shape, label_batch.shape)

    first_input_batch, first_label_batch = next(iter(train_loader))
    print("input:", first_input_batch[0])
    print("label:", first_label_batch[0])

    # # ----------实例化预测数据集----------
    # predict_dataset = Basic1DDataset(mask_file, '../dataset/1Dspectrum[100,42].h5', encoder='predict')
    # # 遍历数据集进行预测
    # for idx in range(len(predict_dataset)):
    #     input_data = predict_dataset[idx]  # 获取单条输入数据
    #     print(input_data.shape)