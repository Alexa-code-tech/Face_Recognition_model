"""
Complete Face Recognition Pipeline

Combines detection, embedding, and matching
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .face_detector import FaceDetector, Face
from .face_embedder import FaceEmbedder
from .matcher import FaceMatcher, Match

logger = logging.getLogger(__name__)


class RecognitionPipeline:
    """
    Полный pipeline распознавания лиц
    """
    
    def __init__(self, 
                 detector_name: str = "mtcnn",
                 embedder_name: str = "facenet",
                 distance_metric: str = "euclidean",
                 tolerance: float = 0.6,
                 device: str = "cuda"):
        """
        Инициализация pipeline
        
        Args:
            detector_name: Детектор (mtcnn, retinaface, dlib)
            embedder_name: Эмбедер (facenet, vggface2, openface)
            distance_metric: Метрика расстояния
            tolerance: Порог подобия
            device: cuda или cpu
        """
        self.detector = FaceDetector(detector_name)
        self.embedder = FaceEmbedder(embedder_name, device)
        self.matcher = FaceMatcher(distance_metric, tolerance)
        
        logger.info("Initialized Recognition Pipeline")
    
    def register_faces(self, name: str, image_paths: List[str]) -> int:
        """
        Зарегистрировать лица человека
        
        Args:
            name: Имя человека
            image_paths: Список путей к изображениям
            
        Returns:
            Количество успешно обработанных изображений
        """
        import cv2
        
        successful = 0
        embeddings_list = []
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                
                faces = self.detector.detect_faces(image)
                if not faces:
                    logger.warning(f"No face detected in {image_path}")
                    continue
                
                # Использовать первое (обычно главное) лицо
                face = faces[0]
                embedding = self.embedder.get_embedding(face.image)
                
                if embedding is not None:
                    embeddings_list.append(embedding)
                    successful += 1
            
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        if embeddings_list:
            self.matcher.register_face(name, np.array(embeddings_list))
        
        logger.info(f"Registered {successful}/{len(image_paths)} faces for {name}")
        return successful
    
    def recognize(self, image: np.ndarray, top_k: int = 1) -> Dict:
        """
        Распознать лица в изображении
        
        Args:
            image: Входное изображение (BGR)
            top_k: Количество топ результатов для каждого лица
            
        Returns:
            Словарь с результатами:
            {
                "faces_detected": int,
                "recognized_people": [
                    {
                        "name": str,
                        "confidence": float,
                        "location": [top, right, bottom, left],
                        "all_matches": [Match]
                    }
                ]
            }
        """
        # Детектировать лица
        faces = self.detector.detect_faces(image)
        
        recognized_people = []
        
        for face in faces:
            # Извлечь эмбеддинг
            embedding = self.embedder.get_embedding(face.image)
            
            if embedding is None:
                logger.warning("Could not extract embedding")
                continue
            
            # Найти совпадение
            matches = self.matcher.match_face(embedding, top_k=top_k)
            
            if matches:
                best_match = matches[0]
                recognized_people.append({
                    "name": best_match.name,
                    "confidence": float(best_match.confidence),
                    "distance": float(best_match.distance),
                    "location": list(face.box),
                    "all_matches": [
                        {
                            "name": m.name,
                            "confidence": float(m.confidence),
                            "distance": float(m.distance)
                        }
                        for m in matches
                    ]
                })
            else:
                recognized_people.append({
                    "name": "Unknown",
                    "confidence": 0.0,
                    "location": list(face.box),
                    "all_matches": []
                })
        
        return {
            "faces_detected": len(faces),
            "recognized_people": recognized_people
        }
    
    def verify(self, image: np.ndarray, name: str) -> Dict:
        """
        Проверить, есть ли конкретный человек в изображении
        
        Args:
            image: Входное изображение
            name: Имя для проверки
            
        Returns:
            Словарь с результатом верификации
        """
        faces = self.detector.detect_faces(image)
        
        if not faces:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "No faces detected"
            }
        
        # Проверить первое лицо (основное)
        face = faces[0]
        embedding = self.embedder.get_embedding(face.image)
        
        if embedding is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "Could not extract embedding"
            }
        
        is_match, confidence = self.matcher.verify_face(embedding, name)
        
        return {
            "verified": is_match,
            "confidence": float(confidence),
            "message": "Face verified" if is_match else "Face not verified"
        }
    
    def get_stats(self) -> Dict:
        """Получить статистику системы"""
        matcher_stats = self.matcher.get_stats()
        
        return {
            "detector": self.detector.model_name,
            "embedder": self.embedder.model_name,
            **matcher_stats
        }