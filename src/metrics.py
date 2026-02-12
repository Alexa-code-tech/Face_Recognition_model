"""
Evaluation Metrics Module
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


class RecognitionMetrics:
    """Метрики для оценки системы распознавания"""
    
    @staticmethod
    def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Вычислить основные метрики
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Словарь метрик
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
    
    @staticmethod
    def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """Вычислить матрицу ошибок"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true: List[int], y_pred: List[int], 
                                 target_names: List[str] = None) -> str:
        """Получить подробный отчет классификации"""
        return classification_report(y_true, y_pred, target_names=target_names,
                                    zero_division=0)
    
    @staticmethod
    def compute_face_verification_metrics(distances: List[float],
                                         labels: List[int],
                                         threshold: float) -> Dict[str, float]:
        """
        Вычислить метрики верификации лиц
        
        Args:
            distances: Список расстояний
            labels: Список меток (1 - одно лицо, 0 - разные)
            threshold: Порог для верификации
            
        Returns:
            Словарь метрик
        """
        predictions = (np.array(distances) <= threshold).astype(int)
        labels_binary = np.array(labels)
        
        tp = np.sum((predictions == 1) & (labels_binary == 1))
        tn = np.sum((predictions == 0) & (labels_binary == 0))
        fp = np.sum((predictions == 1) & (labels_binary == 0))
        fn = np.sum((predictions == 0) & (labels_binary == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
        }
