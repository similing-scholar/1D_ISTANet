import torch

class ISTA_Loss(torch.nn.Module):

    def __init__(self, gamma=.01):
        super(ISTA_Loss, self).__init__()
        self.discrepancy_loss = torch.nn.MSELoss()
        self.gamma = gamma
        # self.constraint_loss = torch.nn.MSELoss()

    def forward(self, x, y, sym_loss):
        discrepancy_loss = self.discrepancy_loss(x, y)
        constraint_loss = torch.sum(torch.mean(torch.pow(sym_loss, 2), dim=1))
        # constraint_loss = torch.mean(torch.pow(sym_loss[0], 2))
        # for k in range(1, len(sym_loss)):
        #     constraint_loss += torch.mean(torch.pow(sym_loss[k], 2))
        total_loss = discrepancy_loss + self.gamma * constraint_loss
        return total_loss