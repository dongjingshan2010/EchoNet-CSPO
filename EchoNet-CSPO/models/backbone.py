"""帧级别 CNN 特征提取器, 直接复用 torchvision 的 ResNet 预训练权重。"""
import torch
import torch.nn as nn
import torchvision.models as tvm


_REGISTRY = {
    'resnet18': (tvm.resnet18, 'ResNet18_Weights', 512),
    'resnet34': (tvm.resnet34, 'ResNet34_Weights', 512),
    'resnet50': (tvm.resnet50, 'ResNet50_Weights', 2048),
}


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        if name not in _REGISTRY:
            raise ValueError(f'unsupported backbone: {name}')
        ctor, weights_cls_name, feat_dim = _REGISTRY[name]
        weights = None
        if pretrained:
            weights_cls = getattr(tvm, weights_cls_name)
            weights = weights_cls.DEFAULT
        m = ctor(weights=weights)
        self.feat_dim = feat_dim
        # 剥掉最后的 fc, 保留 GAP 输出
        self.stem = nn.Sequential(*list(m.children())[:-1])

    def set_frozen(self, frozen: bool):
        for p in self.stem.parameters():
            p.requires_grad = not frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,3,H,W) -> (B,T,feat_dim) 或 (B,3,H,W) -> (B,feat_dim)。"""
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            f = self.stem(x).flatten(1)
            return f.view(B, T, -1)
        return self.stem(x).flatten(1)
