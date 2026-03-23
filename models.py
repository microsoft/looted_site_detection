"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
try:
    import xgboost as xgb  # type: ignore
except ImportError:
    xgb = None  # xgboost optional

# Import CNN models
try:
    from .cnn_models import get_model as _cnn_get_model, MODEL_REGISTRY as _CNN_MODEL_REGISTRY
    CNN_AVAILABLE = True
except Exception as e:  # Broad to also catch runtime errors from torchvision registration
    logging.warning(f"CNN model registry unavailable ({e}).")
    CNN_AVAILABLE = False
    _cnn_get_model = None
    _CNN_MODEL_REGISTRY = {}

@dataclass
class ModelSpec:
    name: str
    aggregation: str
    feature_type: str

class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        return self.fc(h)

class TorchWrapper:
    def __init__(self, model, lr: float = 1e-3, epochs: int = 20, device: Optional[str] = None):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.device = device or ('cuda' if torch and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    def fit(self, X: np.ndarray, y: np.ndarray):
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.long).to(self.device)
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(Xt)
            loss = self.criterion(out, yt)
            loss.backward()
            self.optimizer.step()
    def predict(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(Xt)
            return torch.argmax(logits, dim=1).cpu().numpy()
    def predict_proba(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

def build_model(name: str, aggregation: str, input_dim: int = None, **kwargs) -> object:
    """
    Build a model based on name.
    
    Traditional models (RF, LogReg, GB, XGB, MLP, GRU) expect pre-extracted features.
    CNN models (ResNet*, EfficientNet*) expect raw images and return PyTorch models.
    
    Args:
          name: Model name (rf, logreg, gb, xgb, mlp, gru, resnet20, resnet50,
              efficientnet_b0)
        aggregation: Aggregation method for traditional models (ignored for CNNs)
        input_dim: Input dimension for traditional models (ignored for CNNs)
        **kwargs: Additional arguments for CNN models:
            - num_classes: Number of output classes (default 2)
            - pretrained: Use pretrained weights for ResNet50/EfficientNet
            - mask_mode: 'multiply', 'concat', or 'none'
            - in_channels: Number of input channels (default 3 for RGB)
    
    Returns:
        Model instance (sklearn/xgboost model or PyTorch nn.Module)
    """
    name = name.lower()
    
    # Check if it's a CNN model
    # CNN path using new registry abstraction
    if name in _CNN_MODEL_REGISTRY:
        if not CNN_AVAILABLE or _cnn_get_model is None:
            raise RuntimeError(f'CNN model {name} requested but registry unavailable. Check earlier warnings.')
        num_classes = kwargs.get('num_classes', 2)
        pretrained = kwargs.get('pretrained', True)  # Default to True for ImageNet pretrained weights
        # Additional kwargs ignored for simplified registry; functions accept num_classes and optionally pretrained
        return _cnn_get_model(name, pretrained=pretrained, num_classes=num_classes)
    
    # Traditional models below
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    if name == 'logreg':
        # Increased iterations and class balancing; lbfgs generally fast for small-medium feature counts.
        # Users can switch to saga for very large concatenated feature spaces or elastic-net (not exposed here).
        return LogisticRegression(max_iter=10000, solver='lbfgs', class_weight='balanced', n_jobs=None)
    if name in {'gb','gboost','gradientboost','gradient_boost'}:
        return GradientBoostingClassifier(random_state=42)
    if name in {'xgb','xgboost'}:
        assert xgb is not None, 'xgboost package not installed. Please pip install xgboost.'
        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False
        )
    if name == 'gru':
        assert torch is not None, 'PyTorch not available for GRU model.'
        assert aggregation == 'none', 'Use aggregation=none for GRU (raw sequence).'
        return TorchWrapper(GRUClassifier(input_dim))
    if name == 'mlp':
        # MLP assumes normalized inputs. Recommend running with --normalize.
        # Early stopping to prevent overfitting; moderately sized architecture.
        return MLPClassifier(
            hidden_layer_sizes=(256,128),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            alpha=1e-4,
            batch_size='auto',
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.1,
            shuffle=True,
            random_state=42,
            verbose=False
        )
    raise ValueError(f'Unknown model name {name}.')


def is_cnn_model(model_name: str) -> bool:
    """Check if a model name refers to a CNN model (vs traditional ML model)."""
    cnn_names = {'resnet20', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet'}
    return model_name.lower() in cnn_names

