"""
Face Embedding Module

Extracts face embeddings using pretrained models:
- FaceNet
- VGGFace2
- OpenFace
"""

import numpy as np
import cv2
from typing import List, Union
import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Извлечение эмбеддингов лиц
    """
    
    def __init__(self, model_name: str = "facenet", device: str = "cuda"):
        """
        Инициализация эмбедера
        
        Args:
            model_name: Название модели (facenet, vggface2, openface)
            device: cuda или cpu
        """
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()
        logger.info(f"Loaded {model_name} embedding model on {device}")
    
    def _load_model(self):
        """Загрузить модель"""
        if self.model_name == "facenet":
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2')
            model = model.to(self.device)
            model.eval()
            return model
        
        elif self.model_name == "vggface2":
            import tensorflow as tf
            import tensorflow_hub as hub
            module_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
            model = hub.load(module_url)
            return model
        
        elif self.model_name == "openface":
            import dlib
            path = "models/pretrained_weights/dlib_face_recognition_resnet_model_v1.dat"
            model = dlib.face_recognition_model_v1(path)
            return model
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Получить эмбеддинг одного лица
        
        Args:
            face_image: Изображение лица (BGR)
            
        Returns:
            Вектор эмбеддинга (1D array)
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided")
            return None
        
        try:
            if self.model_name == "facenet":
                return self._get_embedding_facenet(face_image)
            elif self.model_name == "openface":
                return self._get_embedding_openface(face_image)
            else:
                return self._get_embedding_vggface2(face_image)
        
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def _get_embedding_facenet(self, face_image: np.ndarray) -> np.ndarray:
        """Эмбеддинг с FaceNet"""
        # Конвертировать BGR в RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Изменить размер до 160x160
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        # Конвертировать в tensor
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float()
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Нормализация (ImageNet)
        face_tensor = (face_tensor - 127.5) / 128.0
        
        # Получить эмбеддинг
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def _get_embedding_openface(self, face_image: np.ndarray) -> np.ndarray:
        """Эмбеддинг с OpenFace"""
        import dlib
        
        # Конвертировать BGR в RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_rgb = dlib.matrix(face_rgb)
        
        # Получить эмбеддинг
        embedding = self.model.compute_face_descriptor(face_rgb)
        
        return np.array(embedding)
    
    def _get_embedding_vggface2(self, face_image: np.ndarray) -> np.ndarray:
        """Эмбеддинг с VGGFace2"""
        # Конвертировать BGR в RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Изменить размер до 299x299 (для Inception v3)
        face_resized = cv2.resize(face_rgb, (299, 299))
        
        # Нормализация
        face_resized = face_resized / 255.0
        face_resized = (face_resized - 0.5) * 2
        
        # Получить эмбеддинг
        embedding = self.model(np.expand_dims(face_resized, 0))
        
        return embedding.numpy().flatten()
    
    def get_embeddings_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Получить эмбеддинги для списка лиц
        
        Args:
            face_images: Список изображений лиц
            
        Returns:
            Матрица эмбеддингов (N, embedding_size)
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.get_embedding(face_image)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            logger.warning("No valid embeddings extracted")
            return None
        
        return np.array(embeddings)
    
    def align_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Выравнять лицо (опционально для улучшения качества)
        
        Args:
            face_image: Изображение лица
            
        Returns:
            Выравненное изображение
        """
        # Можно добавить выравнивание по глазам
        # Здесь просто возвращаем как есть
        return face_image