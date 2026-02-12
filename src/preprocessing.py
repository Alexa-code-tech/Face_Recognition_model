"""
Image Preprocessing Module
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Предобработка изображений для распознавания лиц"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Инициализация препроцессора
        
        Args:
            target_size: Целевой размер изображения
        """
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Полная предобработка изображения
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Предобработанное изображение
        """
        # Изменить размер
        image = self.resize(image, self.target_size)
        
        # Нормализировать
        image = self.normalize(image)
        
        return image
    
    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Изменить размер изображения"""
        return cv2.resize(image, size)
    
    @staticmethod
    def normalize(image: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        Нормализировать пиксели изображения
        
        Args:
            image: Входное изображение
            method: "standard" или "minmax"
            
        Returns:
            Нормализированное изображение
        """
        if method == "standard":
            # Стандартная нормализация (ImageNet)
            mean = np.array([103.939, 116.779, 123.68])
            image = image.astype(np.float32)
            image = image - mean
            return image
        
        elif method == "minmax":
            # Min-Max нормализация
            return image.astype(np.float32) / 255.0
        
        else:
            return image
    
    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """Выравнять гистограмму"""
        if len(image.shape) == 3:
            # BGR изображение
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_hsv[:, :, 2] = cv2.equalizeHist(image_hsv[:, :, 2])
            return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        else:
            # Серое изображение
            return cv2.equalizeHist(image)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, alpha: float = 1.0, 
                         beta: int = 0) -> np.ndarray:
        """
        Отрегулировать яркость и контраст
        
        Args:
            image: Входное изображение
            alpha: Коэффициент контраста (1.0 - без изменений)
            beta: Смещение яркости (0 - без изменений)
            
        Returns:
            Отрегулированное изображение
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Удалить шум с изображения"""
        return cv2.fastNlMeansDenoisingColored(image, None, h=10, 
                                               hForColorComponents=10,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Применить гауссовское размытие"""
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Заострить изображение"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """Отразить изображение горизонтально"""
        return cv2.flip(image, 1)
    
    @staticmethod
    def flip_vertical(image: np.ndarray) -> np.ndarray:
        """Отразить изображение вертикально"""
        return cv2.flip(image, 0)
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Повернуть изображение
        
        Args:
            image: Входное изображение
            angle: Угол поворота в градусах
            
        Returns:
            Повернутое изображение
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
