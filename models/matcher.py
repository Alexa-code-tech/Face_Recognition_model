"""
Face Matching Module

Matches extracted embeddings against known faces
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cosine, euclidean
import logging

logger = logging.getLogger(__name__)


class Match:
    """Класс для хранения результата матчинга"""
    def __init__(self, name: str, distance: float, confidence: float):
        self.name = name
        self.distance = distance
        self.confidence = confidence
    
    def __repr__(self):
        return f"Match(name={self.name}, confidence={self.confidence:.3f})"


class FaceMatcher:
    """
    Матчинг лиц по эмбеддингам
    """
    
    def __init__(self, distance_metric: str = "euclidean", tolerance: float = 0.6):
        """
        Инициализация матчера
        
        Args:
            distance_metric: "euclidean" или "cosine"
            tolerance: Порог подобия
        """
        self.distance_metric = distance_metric
        self.tolerance = tolerance
        self.known_embeddings: Dict[str, List[np.ndarray]] = {}
        logger.info(f"Initialized FaceMatcher with {distance_metric} metric")
    
    def register_face(self, name: str, embeddings: np.ndarray) -> None:
        """
        Зарегистрировать известное лицо
        
        Args:
            name: Имя человека
            embeddings: Матрица эмбеддингов (N, embedding_size)
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if name not in self.known_embeddings:
            self.known_embeddings[name] = []
        
        self.known_embeddings[name].extend(embeddings)
        logger.info(f"Registered {len(embeddings)} embeddings for {name}")
    
    def match_face(self, embedding: np.ndarray, top_k: int = 1) -> List[Match]:
        """
        Найти совпадение для эмбеддинга
        
        Args:
            embedding: Вектор эмбеддинга
            top_k: Возвращать топ k результатов
            
        Returns:
            Список Match объектов отсортированных по расстоянию
        """
        if not self.known_embeddings:
            logger.warning("No known embeddings registered")
            return []
        
        distances = {}
        
        for name, embeddings_list in self.known_embeddings.items():
            min_distance = float('inf')
            
            for known_embedding in embeddings_list:
                distance = self._compute_distance(embedding, known_embedding)
                min_distance = min(min_distance, distance)
            
            distances[name] = min_distance
        
        # Сортировать по расстоянию
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        
        matches = []
        for name, distance in sorted_distances[:top_k]:
            confidence = self._distance_to_confidence(distance)
            
            if distance <= self.tolerance:
                matches.append(Match(name, distance, confidence))
        
        return matches
    
    def verify_face(self, embedding: np.ndarray, name: str) -> Tuple[bool, float]:
        """
        Проверить, принадлежит ли эмбеддинг конкретному человеку
        
        Args:
            embedding: Вектор эмбеддинга
            name: Имя человека
            
        Returns:
            (is_match, confidence)
        """
        if name not in self.known_embeddings:
            logger.warning(f"Unknown person: {name}")
            return False, 0.0
        
        min_distance = float('inf')
        
        for known_embedding in self.known_embeddings[name]:
            distance = self._compute_distance(embedding, known_embedding)
            min_distance = min(min_distance, distance)
        
        confidence = self._distance_to_confidence(min_distance)
        is_match = min_distance <= self.tolerance
        
        return is_match, confidence
    
    def _compute_distance(self, embedding1: np.ndarray, 
                         embedding2: np.ndarray) -> float:
        """Вычислить расстояние между эмбеддингами"""
        if self.distance_metric == "euclidean":
            return euclidean(embedding1, embedding2)
        elif self.distance_metric == "cosine":
            return cosine(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _distance_to_confidence(self, distance: float) -> float:
        """
        Конвертировать расстояние в уверенность
        
        distance -> confidence (0 to 1)
        0 -> 1.0 (идеальное совпадение)
        tolerance -> 0.5
        infinity -> 0.0
        """
        if distance <= 0:
            return 1.0
        elif distance >= self.tolerance * 2:
            return 0.0
        else:
            return 1.0 - (distance / (self.tolerance * 2))
    
    def get_known_people(self) -> List[str]:
        """Получить список известных людей"""
        return list(self.known_embeddings.keys())
    
    def get_stats(self) -> Dict:
        """Получить статистику"""
        return {
            "known_people": len(self.known_embeddings),
            "total_embeddings": sum(len(e) for e in self.known_embeddings.values()),
            "distance_metric": self.distance_metric,
            "tolerance": self.tolerance,
        }