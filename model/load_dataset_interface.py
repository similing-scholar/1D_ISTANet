import h5py
import numpy
import numpy as np
from torch.utils.data import Dataset, random_split
import torch
import os
import time


def load_h5(file_name, dataset_name='data'):
    """从HDF5文件中加载数据并返回numpy.ndarray。
    """
    with h5py.File(file_name, 'r') as hf:
        loaded_dataset = hf[dataset_name][:]

    return loaded_dataset


def ISTAnet_encoder(spectrum_dataset, mask, encoder):
    """得到初始值矩阵Q
    训练时，x,y,mask都是已知的，采用ISTAnet论文中赋初值的方式
    预测时，y,mask已知，x未知，采用DeepInverse论文中赋初值的方式
    """
    if encoder == 'train':
        # 初始化Q矩阵：每一个对应尺寸为mask转置的大小 即(batch_size, 101, 46)
        Qinit = np.zeros((spectrum_dataset.shape[0], mask.shape[1], mask.shape[0]))
        # 对spectrum_dataset每一行(即x)做变换
        for i in range(spectrum_dataset.shape[0]):  # data_size
            x = spectrum_dataset[i, :].T  # (101, 1)
            y = np.dot(mask, x)  # (46, 101)·(101, 1) -> (46, 1)
            y_yT = np.dot(y, y.T)  # (46, 1)·(1, 46) -> (46, 46)
            x_yT = np.dot(x, y.T)  # (101, 1)·(1, 46) -> (101, 46)
            Qinit[i, :, :] = np.dot(x_yT, np.linalg.inv(y_yT))  # (101, 46)
        return Qinit

    elif encoder == 'predict':
        # 初始化一个列数是mask列数的recon_dataset
        recon_dataset = numpy.zeros((spectrum_dataset.shape[0], mask.shape[1]))
        for i in range(spectrum_dataset.shape[0]):
            reconstructed_value = np.dot(mask.T, spectrum_dataset[i, :])  # 42 -> 101
            recon_dataset[i, :] = reconstructed_value
        return recon_dataset

    else:
        raise ValueError("Invalid encoder mode. Use 'train' or 'predict'.")


def normalize_data(data):
    # 计算数据的最小值和最大值
    data_min = np.min(data)
    data_max = np.max(data)

    # 归一化数据到 [0, 1] 范围内
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data


class Basic1DDataset(Dataset):
    def __init__(self, mask_file, input_file, output_file=None, encoder='train'):
        """
        验证和实际预测中，编码方式会不同，因此需要encoder参数来区分。
        """
        self.mask = load_h5(mask_file)
        self.encoder = encoder  # encoder的值存储为实例变量
        self.input_data = ISTAnet_encoder(load_h5(input_file), self.mask, self.encoder)
        # 根据是否提供output_file来决定是否加载输出数据，output_file为本地变量
        if output_file is not None:
            self.output_data = load_h5(output_file)

    def __len__(self):
        return len(self.input_data)  # 数据集的长度是输入数据的行数

    def __getitem__(self, index):
        # 根据索引获取对应的输入行，并归一化
        input_row = normalize_data(self.input_data[index])
        # 将行数据转换为PyTorch张量，并增加一个通道数保证conv1d的输入格式
        input_tensor = torch.tensor(input_row, dtype=torch.float32).view(1, -1)  # 增加一个通道数 (1, 101) -> (1, 1, 101)

        if self.encoder == 'train':  # 如果是训练模式，返回输入和输出对
            # 根据索引获取对应的输出行，并归一化
            output_row = normalize_data(self.output_data[index])
            output_tensor = torch.tensor(output_row, dtype=torch.float32).view(1, -1)
            return input_tensor, output_tensor

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
    input_file = '../dataset/1Dspectrum[10000,101].h5'
    output_file = '../dataset/1Dspectrum[10000,101].h5'

    # ----------实例化训练数据集----------
    train_dataset = Basic1DDataset(mask_file, input_file, output_file)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, 0.8, 0.2, 64)

    # 遍历数据加载器以进行训练
    for input_batch, output_batch in train_loader:
        # 在这里进行模型的训练
        print(input_batch.shape, output_batch.shape)

    first_input_batch, first_output_batch = next(iter(train_loader))
    print("input:", first_input_batch[0])
    print("output:", first_output_batch[0])

    # # ----------实例化预测数据集----------
    # predict_dataset = Basic1DDataset(mask_file, '../dataset/1Dspectrum[100,42].h5', encoder='predict')
    # # 遍历数据集进行预测
    # for idx in range(len(predict_dataset)):
    #     input_data = predict_dataset[idx]  # 获取单条输入数据
    #     print(input_data.shape)