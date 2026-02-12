"""
Face Detection Module

Supports multiple detection backends:
- MTCNN (Multi-task Cascaded Convolutional Networks)
- RetinaFace
- dlib
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Face:
    """Класс для хранения информации о лице"""
    def __init__(self, image: np.ndarray, box: Tuple[int, int, int, int], 
                 confidence: float = 1.0):
        self.image = image
        self.box = box  # (top, right, bottom, left)
        self.confidence = confidence
        self.embedding = None
    
    def __repr__(self):
        return f"Face(confidence={self.confidence:.3f}, box={self.box})"


class FaceDetector:
    """
    Детектор лиц с поддержкой множественных бэкендов
    """
    
    def __init__(self, model_name: str = "mtcnn", confidence_threshold: float = 0.95):
        """
        Инициализация детектора лиц
        
        Args:
            model_name: Название модели (mtcnn, retinaface, dlib)
            confidence_threshold: Порог уверенности
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.detector = self._load_detector()
        logger.info(f"Loaded {model_name} detector")
    
    def _load_detector(self):
        """Загрузить детектор"""
        if self.model_name == "mtcnn":
            from mtcnn import MTCNN
            return MTCNN()
        elif self.model_name == "retinaface":
            from retinaface import RetinaFace
            return RetinaFace
        elif self.model_name == "dlib":
            import dlib
            return dlib.get_frontal_face_detector()
        else:
            raise ValueError(f"Unknown detector: {self.model_name}")
    
    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """
        Обнаружить лица в изображении
        
        Args:
            image: Входное изображение (BGR формат OpenCV)
            
        Returns:
            Список объектов Face с обнаруженными лицами
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return []
        
        if self.model_name == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.model_name == "dlib":
            return self._detect_dlib(image)
        else:
            return self._detect_retinaface(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Face]:
        """Детектирование с MTCNN"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb_image)
            
            faces = []
            for result in results:
                confidence = result['confidence']
                
                if confidence < self.confidence_threshold:
                    continue
                
                bbox = result['box']
                x, y, w, h = bbox
                
                # Конвертировать в формат (top, right, bottom, left)
                top, right, bottom, left = y, x + w, y + h, x
                
                # Обрезать лицо
                face_image = image[top:bottom, left:right]
                
                face = Face(
                    image=face_image,
                    box=(top, right, bottom, left),
                    confidence=confidence
                )
                faces.append(face)
            
            logger.info(f"Detected {len(faces)} faces with MTCNN")
            return faces
            
        except Exception as e:
            logger.error(f"Error in MTCNN detection: {e}")
            return []
    
    def _detect_dlib(self, image: np.ndarray) -> List[Face]:
        """Детектирование с dlib"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dlib_rects = self.detector(gray, 1)
            
            faces = []
            for rect in dlib_rects:
                top = rect.top()
                left = rect.left()
                bottom = rect.bottom()
                right = rect.right()
                
                face_image = image[top:bottom, left:right]
                face = Face(
                    image=face_image,
                    box=(top, right, bottom, left),
                    confidence=1.0
                )
                faces.append(face)
            
            logger.info(f"Detected {len(faces)} faces with dlib")
            return faces
            
        except Exception as e:
            logger.error(f"Error in dlib detection: {e}")
            return []
    
    def _detect_retinaface(self, image: np.ndarray) -> List[Face]:
        """Детектирование с RetinaFace"""
        try:
            results = self.detector.detect_faces(image)
            
            faces = []
            for key in results:
                resp = results[key]
                confidence = resp['score']
                
                if confidence < self.confidence_threshold:
                    continue
                
                facial_area = resp['facial_area']
                top, right, bottom, left = facial_area
                
                face_image = image[top:bottom, left:right]
                face = Face(
                    image=face_image,
                    box=(top, right, bottom, left),
                    confidence=confidence
                )
                faces.append(face)
            
            logger.info(f"Detected {len(faces)} faces with RetinaFace")
            return faces
            
        except Exception as e:
            logger.error(f"Error in RetinaFace detection: {e}")
            return []
    
    def draw_faces(self, image: np.ndarray, faces: List[Face], 
                   show_confidence: bool = True) -> np.ndarray:
        """
        Нарисовать прямоугольники вокруг лиц
        
        Args:
            image: Исходное изображение
            faces: Список обнаруженных лиц
            show_confidence: Показывать ли уверенность
            
        Returns:
            Изображение с нарисованными лицами
        """
        output = image.copy()
        
        for face in faces:
            top, right, bottom, left = face.box
            
            # Рисовать прямоугольник
            cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Рисовать уверенность
            if show_confidence:
                text = f"{face.confidence:.3f}"
                cv2.putText(output, text, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output