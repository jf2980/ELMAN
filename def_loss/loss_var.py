


# def charbonnier_loss_color(pred, target, eps):
#     diff = torch.add(pred, -target)
#     diff_sq = diff * diff
#     diff_sq_color = torch.mean(diff_sq, 1, True)
#     error = torch.sqrt(diff_sq_color + eps)
#     loss = torch.mean(error)
#     return loss
#
#
# class CharbonnierLossColor(nn.Module):
#     """Charbonnier loss (one variant of Robust L1Loss, a differentiable
#     variant of L1Loss).
#
#     Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
#         Super-Resolution".
#
#     Args:
#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#         eps (float): A value used to control the curvature near zero. Default: 1e-12.
#     """
#
#     def __init__(self, loss_weight=2.0, reduction='mean', eps=1e-6):
#         super(CharbonnierLossColor, self).__init__()
#         # if reduction not in ['none', 'mean', 'sum']:
#         #     raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
#
#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.eps = eps
#
#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
#         """
#         return self.loss_weight * charbonnier_loss_color(pred, target,  eps=self.eps, )
# class CharbonnierLoss(torch.nn.Module):
#     """Charbonnier Loss (L1)"""
#     def __init__(self, eps=1e-6):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps
#
#     def forward(self, x, y):
#         b, c, h, w = y.size()
#         loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
#         return loss/(c*b*h*w)
# class CharbonnierLoss(torch.nn.Module):
#     def __init__(self, epsilon=1e-6):
#         super(CharbonnierLoss, self).__init__()
#         self.epsilon = epsilon
#
#     def forward(self, prediction, target):
#         return torch.mean(torch.sqrt((prediction - target) ** 2 + self.epsilon ** 2 ))
import torch.nn as nn
import torch

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return  self.criterion(pred_fft, target_fft)