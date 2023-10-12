import h5py
import numpy as np


def load_h5(file_name, dataset_name='data'):
    """从HDF5文件中加载数据并返回numpy.ndarray。
    """
    with h5py.File(file_name, 'r') as hf:
        loaded_dataset = hf[dataset_name][:]

    return loaded_dataset

def save_h5(data, filename):
    """h5py格式会压缩，占用空间小，读取速度快，但是不能直接用文本编辑器打开"""
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('data', data=data)
    h5f.close()


def normalize_data(data):
    # 计算数据的最小值和最大值
    data_min = np.min(data)
    data_max = np.max(data)

    # 归一化数据到 [0, 1] 范围内
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data