from torch import nn

from .bce import BCELoss
from .wbce import WBCELoss
from .focal_loss import BinaryFocalLoss
from .dice_loss import DiceLoss
from .combo_loss import ComboLoss
from .quality_focal_loss import QualityFocalLoss
from utils.utils import _sigmoid

class HeatmapLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        loss_name = cfg['loss']['sub_name']
        if loss_name=='mse':
            self._loss = nn.MSELoss()
        elif loss_name=='bce':
            #self._loss = nn.BCELoss()
            self._loss = BCELoss()
        elif loss_name=='wbce':
            auto_weight = cfg['loss']['auto_weight']
            scales      = cfg['loss']['scales']
            self._loss  = WBCELoss(auto_weight=auto_weight, scales=scales)
        elif loss_name=='focal':
            gamma      = cfg['loss']['gamma']
            auto_weight = cfg['loss']['auto_weight']
            scales      = cfg['loss']['scales']
            self._loss = BinaryFocalLoss(gamma=gamma, auto_weight=auto_weight, scales=scales)
        elif loss_name=='quality_focal':
            beta       = cfg['loss']['beta']
            self._loss = QualityFocalLoss(beta=beta)
        elif loss_name=='dice':
            epsilon    = cfg['loss']['epsilon']
            self._loss = DiceLoss(epsilon=epsilon)
        elif loss_name=='combo':
            epsilon    = cfg['loss']['epsilon']
            alpha      = cfg['loss']['alpha']
            self._loss = ComboLoss(alpha=alpha, epsilon=epsilon)
        else:
            raise KeyError('invalid loss: {}'.format(loss_name ))

    def forward(self, inputs, targets):
        # 对每个尺度先做 sigmoid
        inputs_s = {scale: _sigmoid(inp) for scale, inp in inputs.items()}

        # 若为 MSELoss，则期望张量，需按尺度逐个计算并累加
        if isinstance(self._loss, nn.MSELoss):
            loss_acc = 0.0
            for scale in inputs_s.keys():
                loss_acc = loss_acc + self._loss(inputs_s[scale], targets[scale])
            return loss_acc

        # 其他（如 BCE/WBCE/Focal/Dice/Combo）接受字典，保持原行为
        return self._loss(inputs_s, targets)

