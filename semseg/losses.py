import torch
from torch import nn, Tensor
from torch.nn import functional as F

def SSIM_(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

def SSIM(x, y, kernel_size=3):
    """
    Calculates the SSIM (Structural SIMilarity) loss

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    kernel_size : int
        Convolutional parameter

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM loss
    """
    ssim_value = SSIM_(x, y, kernel_size=kernel_size)
    return torch.clamp((1. - ssim_value) / 2., 0., 1.)

def calc_photometric_loss(t_est, images, ssim_loss_weight=0.85, clip_loss=0):
    """
    Calculates the photometric loss (L1 + SSIM)
    Parameters
    ----------
    t_est : list of torch.Tensor [B,3,H,W]
        List of warped reference images in multiple scales
    images : list of torch.Tensor [B,3,H,W]
        List of original images in multiple scales

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Photometric loss
    """
    # L1 loss
    l1_loss = [torch.abs(t_est[i] - images[i])
                for i in range(len(t_est))]
    # SSIM loss
    if ssim_loss_weight > 0.0:
        ssim_loss = [SSIM(t_est[i], images[i], kernel_size=3)
                        for i in range(len(t_est))]
        # Weighted Sum: alpha * ssim + (1 - alpha) * l1
        photometric_loss = [ssim_loss_weight * ssim_loss[i].mean(1, True) +
                            (1 - ssim_loss_weight) * l1_loss[i].mean(1, True)
                            for i in range(len(t_est))]
    else:
        photometric_loss = l1_loss
    # Clip loss
    if clip_loss > 0.0:
        for i in range(len(photometric_loss)):
            mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
            photometric_loss[i] = torch.clamp(
                photometric_loss[i], max=float(mean + clip_loss * std))
    # Return total photometric loss
    return photometric_loss

def reduce_photometric_loss(photometric_losses, photometric_reduce_op='min'):
    """
    Combine the photometric loss from all context images

    Parameters
    ----------
    photometric_losses : list of torch.Tensor [B,3,H,W]
        Pixel-wise photometric losses from the entire context

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Reduced photometric loss
    """
    # Reduce function
    def reduce_function(losses):
        if photometric_reduce_op == 'mean':
            return sum([l.mean() for l in losses]) / len(losses)
        elif photometric_reduce_op == 'min':
            return torch.cat(losses, 1).min(1, True)[0].mean()
        else:
            raise NotImplementedError(
                'Unknown photometric_reduce_op: {}'.format(photometric_reduce_op))
    # Reduce photometric loss
    photometric_loss = sum([reduce_function(photometric_losses[i])
                            for i in range(len(photometric_losses))]) / len(photometric_losses)
    return photometric_loss

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)