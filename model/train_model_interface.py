import sys
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

import torch
from torch import nn
from copy import deepcopy


def printlog(info):
    """用于在控制台上打印日志信息，包括当前时间戳和传入的信息。
    """
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")


class StepRunner:
    """用于执行模型训练或验证一步的类
    接收神经网络模型、损失函数、阶段（"train"或"val"）、指标字典、优化器和学习率调度器作为参数。
    """

    def __init__(self, device, dtype, net, loss_fn, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.device, self.dtype = device, dtype
        self.net, self.loss_fn = net, loss_fn
        self.metrics_dict, self.stage = metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler

    def __call__(self, Phix, Phi, Qinit, label):  # __call__使实例能够像函数一样被调用
        """用于执行一步训练或验证
        包括计算损失、反向传播、参数更新（仅在训练阶段）、计算指标，并返回损失和指标的字典。
        """
        # 移动数据到cuda
        Phix = Phix.to(self.device, self.dtype)
        Phi = Phi.to(self.device, self.dtype)
        Qinit = Qinit.to(self.device, self.dtype)
        label = label.to(self.device, self.dtype)

        # ---forward计算损失---
        pred, loss_layers_sym = self.net(Phix, Phi, Qinit)  # 前向传播

        # loss计算分为两步
        loss_discrepancy = torch.mean(torch.pow(pred - label, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(len(self.net.fcs)-1):  # len(self.net.fcs)为网络层数
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(self.device, self.dtype)  # 定值？？？

        loss = loss_discrepancy + torch.mul(gamma, loss_constraint)

        # ---backward反向传播和参数更新（仅在训练阶段执行）---
        if self.optimizer is not None and self.stage == "train":
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            if self.lr_scheduler is not None:  # 学习率更新
                self.lr_scheduler.step()
            self.optimizer.zero_grad()  # 清空梯度

        # ---计算指标---
        # step_metrics是一个字典，用于存储每个指标的名称与相应的指标值。这些指标值是从metric_fn(preds, labels)计算得到的。
        step_metrics = {self.stage + "_" + name: metric_fn(pred, label).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics  # 返回损失和指标的字典，.item()将Tensor转换为普通的数值类型


class EpochRunner:
    """用于执行整个训练或验证过程的类。
    接收一个StepRunner对象作为参数，并在__call__方法中遍历数据加载器（dataloader），执行多个步骤。
    在训练阶段，计算每个步骤的损失和指标，并在每个步骤结束时更新进度条。在整个周期结束时，计算并返回周期内的损失和指标的字典。
    """

    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()  # 设置阶段为训练或验证

    def __call__(self, dataloader):
        global epoch_log
        total_loss, step = 0.0, 0
        # 包装 enumerate(dataloader)，接收一个迭代器作为参数，并返回一个带有进度条的迭代器包装器。
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in loop:
            # ---执行一步训练或验证---
            if self.stage == "train":
                loss, step_metrics = self.steprunner(*batch)
            else:
                with torch.no_grad():  # 验证阶段不需要计算梯度
                    loss, step_metrics = self.steprunner(*batch)  # *于将一个可迭代对象（元组或列表）中的元素解包成多个独立的参数
            # 创建一个step日志字典，包含当前步骤的损失和指标
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)  # **将字典的键值对解包为关键字参数

            # ---累加总损失和步数---
            total_loss += loss
            step += 1
            # ---更新进度条---
            if i != len(dataloader) - 1:  # 如果不是最后一个批次，显示当前步骤的损失和指标
                loop.set_postfix(**step_log)
            else:  # 如果是最后一个批次，显示整个周期的损失和指标
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                # 创建一个epoch日志字典，包含整个周期的损失和指标
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)  # .set_postfix方法接受一个字典作为参数，用于更新tqdm进度条。

                # ---重置周期内的指标，以便下一个周期使用---
                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log  # 只返回整个epoch的损失和指标，不返回每个step的损失和指标


class KerasModel(torch.nn.Module):
    """一个包装神经网络模型、损失函数和训练相关参数的类，继承自torch.nn.Module。
    构造函数中初始化了模型、损失函数、指标字典、优化器和学习率调度器。
    forward方法用于执行模型的前向传播。
    """

    def __init__(self, device, dtype, net, loss_fn, metrics_dict=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.history = {}  # 用于记录训练和验证过程中的损失和指标历史数据

        self.device = device
        self.dtype = dtype
        self.net = net  # 神经网络模型
        self.loss_fn = loss_fn  # 损失函数
        self.metrics_dict = nn.ModuleDict(metrics_dict)  # 指标字典，用于评估模型性能

        # 优化器，如果没有提供，则使用默认的 Adam 优化器
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.parameters(), lr=1e-2)
        self.lr_scheduler = lr_scheduler  # 学习率调度器，可选

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError

    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min"):
        """训练模型的方法，包括训练、验证和早停等步骤。
        ckpt_path：用于保存最佳模型的路径，可选。??????
        patience：早停的等待次数，可选。在一定数量的训练周期内（连续周期）没有观察到性能的改善，就停止训练。
        monitor：用于早停的指标名称，可选。
        mode：用于早停的模式，可选，"min"或"max"。
        """
        for epoch in range(1, epochs + 1):
            printlog("Epoch {0} / {1}".format(epoch, epochs))

            # 1，train -------------------------------------------------
            train_step_runner = StepRunner(device=self.device, dtype=self.dtype, net=self.net, stage="train",
                                           loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                           optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
            train_epoch_runner = EpochRunner(train_step_runner)
            train_metrics = train_epoch_runner(train_data)

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]  # 记录训练阶段的损失和指标

            # 2，validate -------------------------------------------------
            if val_data:
                val_step_runner = StepRunner(device=self.device, dtype=self.dtype, net=self.net, stage="val",
                                             loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict))
                val_epoch_runner = EpochRunner(val_step_runner)
                with torch.no_grad():  # EpochRunner中已经包含了torch.no_grad()??????
                    val_metrics = val_epoch_runner(val_data)
                val_metrics["epoch"] = epoch  # 向 val_metrics 字典中添加一个键值对，其中键是 "epoch"
                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]  # 记录验证阶段的损失和指标

            # 3，early-stopping -------------------------------------------------
            if not val_data:  # 如果没有提供验证数据，则跳过早停逻辑，继续训练。
                continue
            arr_scores = self.history[monitor]  # 从记录训练和验证指标的历史数据中获取与监控指标（monitor）对应的指标值列表
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)  # 返回arr_scores索引
            if best_score_idx == len(arr_scores) - 1:  # 是否当前的性能指标是历史中的最佳值
                torch.save(self.net.state_dict(), ckpt_path)  # 会保存当前模型的状态字典
                # sys.stderr将消息输出到标准错误流中
                print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                                  arr_scores[best_score_idx]), file=sys.stderr)
            if len(arr_scores) - best_score_idx > patience:
                print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                    monitor, patience), file=sys.stderr)
                break

        self.net.load_state_dict(torch.load(ckpt_path))
        return pd.DataFrame(self.history)

    @torch.no_grad()  # 使用装饰器整个函数内禁用梯度计算
    def evaluate(self, val_data):
        val_step_runner = StepRunner(device=self.device, dtype=self.dtype, net=self.net, stage="val",
                                     loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict))
        val_epoch_runner = EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics

    @torch.no_grad()
    def predict(self, dataloader):
        # 在评估模式下，模型的行为有一些特殊的变化：Batch Normalization和Dropout通常会禁用随机性，梯度计算被禁用
        self.net.eval()
        # 将所有批次的模型输出连接成一个大的张量result
        result = torch.cat([self.forward(t[0]) for t in dataloader])
        # return result.dataset  返回一个没有梯度信息的张量
        return result.detach()

