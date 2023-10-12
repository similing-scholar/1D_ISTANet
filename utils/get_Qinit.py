import numpy as np

from utils import load_h5, save_h5


def Qinit_calculator(calculator, mask, spectrum_dataset=None, observed_dataset=None):
    """得到初始值矩阵Q,对观测值y进行编码，使得模型可以得到与输出值等尺寸的初始值x(0)=Qy
    因为Y=ΦX，Q其实为Φ^-1的近似，所以Q的尺寸为mask转置的大小，即(101, 46)
    当calculator='ISTAnet'，采用ISTAnet论文中赋初值的方式:使用所有训练样本的X.Y数据对计算最小二乘估计的Q
    当calculator='DeepInverse'，Q=Φ^T
    """
    if calculator == 'ISTAnet':
        if spectrum_dataset or mask is None:
            raise ValueError("spectrum_dataset & mask is required.")
        # 对所有的X,Y数据对来说，Y=ΦX，所以认为有X=QY的近似关系，可以最小二乘求Q=XY^T(YY^T)^-1
        X = spectrum_dataset.T  # (101, 1) 列向量
        Y = np.dot(mask, X)  # (46, 101)·(101, 1) -> (46, 1) 列向量 原文代码
        Y_YT = np.dot(Y, Y.T)  # (46, 1)·(1, 46) -> (46, 46)
        X_YT = np.dot(X, Y.T)  # (101, 1)·(1, 46) -> (101, 46)
        Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))  # (101, 46)
        del X, Y, X_YT, Y_YT

    elif calculator == 'ISTAnet_observed':
        if spectrum_dataset or observed_dataset is None:
            raise ValueError('spectrum_dataset & observed_dataset are required.')
        # 对所有的X,Y数据对来说，Y=ΦX，所以认为有X=QY的近似关系，可以最小二乘求Q=XY^T(YY^T)^-1
        X = spectrum_dataset.T  # (101, batch_size) 列向量
        Y = observed_dataset.T  # (46, batch_size) 列向量
        Y_YT = np.dot(Y, Y.T)  # (46, batch_size)·(batch_size, 46) -> (46, 46)
        X_YT = np.dot(X, Y.T)  # (101, batch_size)·(batch_size, 46) -> (101, 46)
        Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))  # (101, 46)
        del X, Y, X_YT, Y_YT

    elif calculator == 'DeepInverse':
        # Q=Φ^T
        Qinit = mask.T  # (46, 101) -> (101, 46)

    return Qinit


if __name__ == '__main__':
    # 从本地加载mask、spectrum_dataset、observed_dataset
    mask_file = '../dataset/mask[42,101].h5'
    input_file = '../dataset/1Dspectrum[10000,101].h5'
    output_file = '../dataset/1Dspectrum[10000,101].h5'

    mask = load_h5(mask_file)  # Φ
    observed_dataset = load_h5(input_file)  # 观测Y作为输入
    spectrum_dataset = load_h5(output_file)  # 原信号X作为输出

    # 计算Qinit并保存
    calculator = 'ISTAnet'
    Qinit = Qinit_calculator(calculator, mask, spectrum_dataset)

    Q_save_file = f'../dataset/Q_{calculator}[{mask.shape[1]},{mask.shape[0]}].h5'
    save_h5(Qinit, Q_save_file)
    print(f'Qinit file is saved in {Q_save_file}.')

