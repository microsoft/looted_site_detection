"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:,1])
    except Exception:
        metrics['roc_auc'] = float('nan')
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return metrics
