"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

CNN Model Architectures for Looted Site Detection (ResNet, EfficientNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Attempt to import torchvision models; degrade gracefully if not available or incompatible
try:
    from torchvision import models  # noqa: F401
    _TORCHVISION_AVAILABLE = True
except Exception as e:  # Broad except to catch runtime errors during registration
    print(f"WARNING: torchvision import failed ({e}). Pretrained models will be disabled.")
    models = None  # type: ignore
    _TORCHVISION_AVAILABLE = False


def _weights_init(m):
    """Initialize weights using Kaiming normal initialization"""
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """Custom lambda layer for ResNet shortcuts"""
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2],
                                                            (0, 0, 0, 0, planes//4, planes//4),
                                                            "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone architecture"""
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        out = F.relu(self.bn1(self.conv1(x)))

        if mask is not None:
            # Ensure mask has the correct number of dimensions
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)  # Add a channel dimension
            elif len(mask.shape) != 4:
                raise ValueError(f"Mask must have 3 or 4 dimensions, but got {len(mask.shape)} dimensions")

            # Resize the mask to match the spatial dimensions of 'out'
            mask = F.interpolate(mask, size=out.shape[2:], mode='nearest')

            if mask.size(1) == 1:
                # Repeat the mask across the channel dimension to match the 'out' tensor
                mask = mask.repeat(1, out.size(1), 1, 1)
            elif mask.size(1) != out.size(1):
                raise ValueError(f"Mask channel dimension {mask.size(1)} does not match 'out' channel dimension {out.size(1)}")

            out = out * mask

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetWithMask(nn.Module):
    """Wrapper for torchvision ResNet models to support mask input"""
    def __init__(self, base_model):
        super(ResNetWithMask, self).__init__()
        self.base_model = base_model

    def forward(self, x, mask=None):
        if mask is not None:
            # Ensure mask has the correct number of dimensions
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)  # Add a channel dimension
            elif len(mask.shape) != 4:
                raise ValueError(f"Mask must have 3 or 4 dimensions, but got {len(mask.shape)} dimensions")

            # Resize the mask to match the spatial dimensions of 'x'
            mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')

            if mask.size(1) == 1:
                # Repeat the mask across the channel dimension to match the input tensor
                mask = mask.repeat(1, x.size(1), 1, 1)
            elif mask.size(1) != x.size(1):
                raise ValueError(f"Mask channel dimension {mask.size(1)} does not match input channel dimension {x.size(1)}")

            x = x * mask

        return self.base_model(x)




# Model factory functions
def resnet20(num_classes=2):
    """ResNet20 for binary classification"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet18(pretrained=False, num_classes=2):
    """ResNet18 with optional pretrained weights"""
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision not available; resnet18 disabled")
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception:
            print("Falling back to non-pretrained ResNet18 (weight load failed)")
            model = models.resnet18()
    else:
        model = models.resnet18()

    # Replace final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return ResNetWithMask(model)


def resnet34(pretrained=False, num_classes=2):
    """ResNet34 with optional pretrained weights"""
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision not available; resnet34 disabled")
    if pretrained:
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=weights)
        except Exception:
            print("Falling back to non-pretrained ResNet34 (weight load failed)")
            model = models.resnet34()
    else:
        model = models.resnet34()

    # Replace final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return ResNetWithMask(model)


def resnet50(pretrained=False, num_classes=2):
    """ResNet50 with optional pretrained weights"""
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision not available; resnet50 disabled")
    # Use legacy API; if fails fallback
    try:
        model = models.resnet50(pretrained=pretrained)
    except Exception:
        print("Falling back to non-pretrained ResNet50 (weight load failed)")
        model = models.resnet50(pretrained=False)

    # Replace final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return ResNetWithMask(model)


def efficientnet_b0(pretrained=False, num_classes=2):
    """EfficientNet-B0 with optional pretrained weights"""
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision not available; efficientnet_b0 disabled")
    try:
        model = models.efficientnet_b0(pretrained=pretrained)
    except Exception:
        print("Falling back to non-pretrained EfficientNet-B0 (weight load failed)")
        model = models.efficientnet_b0(pretrained=False)

    # Replace final layer for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return ResNetWithMask(model)


def efficientnet_b1(pretrained=False, num_classes=2):
    """EfficientNet-B1 with optional pretrained weights"""
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision not available; efficientnet_b1 disabled")
    try:
        model = models.efficientnet_b1(pretrained=pretrained)
    except Exception:
        print("Falling back to non-pretrained EfficientNet-B1 (weight load failed)")
        model = models.efficientnet_b1(pretrained=False)

    # Replace final layer for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return ResNetWithMask(model)


# (All UNet variants removed from the public release)


def _removed(*args, **kwargs):
    raise ValueError("UNet-based models are not available in this release.")


# Model registry for easy access
MODEL_REGISTRY = {
    'resnet20': resnet20,
}
if _TORCHVISION_AVAILABLE:
    MODEL_REGISTRY.update({
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'efficientnet_b0': efficientnet_b0,
        'efficientnet_b1': efficientnet_b1,
    })


def get_model(model_name: str, pretrained: bool = True, **kwargs):
    """
    Factory function to get model by name

    Args:
        model_name: Name of the model architecture
        pretrained: Whether to use pretrained weights (default: True for torchvision models)
        **kwargs: Additional arguments for model initialization

    Returns:
        PyTorch model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")

    model_fn = MODEL_REGISTRY[model_name]

    # Check if model supports pretrained weights
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1']:
        return model_fn(pretrained=pretrained, **kwargs)
    else:
        return model_fn(**kwargs)
