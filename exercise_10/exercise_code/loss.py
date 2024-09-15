import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# Custom loss function: combines MSE and L2 regularization
class Loss(nn.Module):
    def __init__(self,loss_func, reg=1e-4, **kwargs):
        super().__init__()
        self.base_loss=loss_func
        self.reg = reg

    def forward(self, outputs, targets, model):
        # Base loss (e.g., MSE)
        loss = self.base_loss(outputs, targets)

        # L2 regularization (sum of squared norms of weights)
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)**2

        # Combine the base loss with L2 regularization
        total_loss = loss + self.reg * l2_reg
        return total_loss
    
    def compute_loss_without_regularization(self, outputs, targets):
        # Compute the loss without regularization
        return self.base_loss(outputs, targets)
    
