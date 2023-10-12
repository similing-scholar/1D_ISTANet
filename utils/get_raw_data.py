import numpy as np
import pandas as pd
import math


def get_array_filters(m, n):
    """获取滤光片光谱数据
    m：滤光片个数
    n：滤光片透过光谱通道个数,建议n的取值[0,19]+21,22,24,26,28,31,34,38,43,51,61,76,101,151,301
    """
    df = pd.read_excel(
        'D:/BIT课题研究/微型光谱成像仪/【数据】相机标定/透过法标定/20230703-ava-滤光片&光源/滤光片透过率.xlsx',
        header=0, index_col=0)  # 滤光片光谱数据【改文件地址参数】

    # dataframe转为array数据格式
    df_array = df.values

    # range(200, 501, 10)个通道实际波长对应的数据行为（400nm-701nm，10）
    waves_num = range(0, 201, math.ceil(201/n))  # 【改滤光片透过光谱通道个数n】
    # 选择行，即每一行为该波长下所有光谱的数据
    array_waves = df_array[waves_num]

    # range(0,30,1)选出30条光谱
    spec_num = []
    for i in range(0, m, 1):  # 【改滤光片个数m】
        spec_num.append(i)
    # 选择列，即每一列为一条光谱
    array_filters = array_waves[:, spec_num]
    # 转置成每一行为一条光谱数据
    array_filters = array_filters.T

    return array_filters


def generate_gaussian_vectors(num_vectors, vector_length, max_gaussians):
    """生成一组随机的高斯分布向量
    num_vectors：生成的向量数量
    vector_length：向量的长度
    max_gaussians：每个向量叠加的高斯分布数量
    """
    np.random.seed(0)  # 设置随机种子以确保结果可重复
    # 定义x轴的取值范围和分辨率
    x = np.linspace(-5, 5, vector_length)
    # 用于存储生成的向量的数组
    vector_array = np.zeros((num_vectors, vector_length))

    for i in range(num_vectors):
        # 随机确定每个向量叠加的高斯分布数量
        num_gaussians = np.random.randint(1, max_gaussians + 1)

        # 生成多个高斯分布的叠加曲线
        combined_curve = np.zeros(vector_length)
        for _ in range(num_gaussians):
            mean = np.random.uniform(-3, 3)  # 随机均值
            std_dev = np.random.uniform(0.1, 2)  # 随机标准差
            gaussian = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
            weight = np.random.uniform(0.1, 1)  # 随机权重
            combined_curve += weight * gaussian

        # 将生成的曲线保存到数组中
        vector_array[i, :] = combined_curve

    return vector_array