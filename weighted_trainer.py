# weighted_trainer.py
import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

class WeightedDetectionLoss(v8DetectionLoss):
    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if class_weights is not None:
            # 强制转 float32，防止 amp=True 时精度报错
            safe_weights = class_weights.float().to(self.device)
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=safe_weights,
                reduction="none", 
            )
            if RANK in (-1, 0):
                print(f"✅ [SUCCESS] WeightedDetectionLoss loaded. Weights: {safe_weights.cpu().tolist()}")

class WeightedE2ELoss(E2ELoss):
    def __init__(self, model, class_weights=None):
        # 【加固 1】必须调用父类初始化！虽然会生成默认无权重的 loss，但能保证所有底层属性（如 device, task 等）正确设置
        super().__init__(model)
        
        # 【加固 2】直接强行覆盖父类生成的两个 loss
        def wloss(m, tal_topk=10, tal_topk2=None):
            return WeightedDetectionLoss(m, class_weights=class_weights, tal_topk=tal_topk, tal_topk2=tal_topk2)
        
        self.one2many = wloss(model, tal_topk=10, tal_topk2=None)
        self.one2one = wloss(model, tal_topk=10, tal_topk2=4)

class WeightedDetectionModel(DetectionModel):
    def __init__(self, *args, class_weights=None, **kwargs):
        self._class_weights = class_weights
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        return WeightedE2ELoss(self, class_weights=self._class_weights)

class WeightedTrainer(DetectionTrainer):
    # 【加固 3】绝对不要写 cfg=None！用 *args, **kwargs 兜底所有未知参数
    def __init__(self, *args, **kwargs):
        # 1. 从 kwargs 中安全地取出 overrides 字典
        overrides = kwargs.get("overrides", None) or {}
        # 2. 剥离 class_weights
        self._class_weights = overrides.pop("class_weights", None)
        # 3. 把干净后的字典放回 kwargs
        kwargs["overrides"] = overrides
        # 4. 原封不动透传给父类，让父类自己处理缺失的 cfg
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = WeightedDetectionModel(
            cfg, nc=self.data["nc"], class_weights=self._class_weights, verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

