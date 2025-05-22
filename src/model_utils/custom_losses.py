import torch
import torch.nn as nn



class LogCoshLoss(nn.Module):
    def __init__(self, p=0.02):
        super(LogCoshLoss, self).__init__()
        self.p = p

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        log_cosh_error = torch.log(torch.cosh(error))
        loss = torch.mean(log_cosh_error) *self.p
        return loss

