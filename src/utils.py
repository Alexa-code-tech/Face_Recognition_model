"""
Utility Functions for Face Recognition System
"""

import os
import pickle
import json
import logging
from typing import Any, Dict, List
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """
    Настроить логирование
    
    Args:
        log_dir: Директория для логов
        level: Уровень логирования
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"face_recognition_{timestamp}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def save_embeddings(embeddings_dict: Dict[str, np.ndarray], 
                   output_path: str) -> None:
    """
    Сохранить эмбеддинги в pickle файл
    
    Args:
        embeddings_dict: Словарь {name: embeddings}
        output_path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    logger.info(f"Saved embeddings to {output_path}")


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    Загрузить эмбеддинги из pickle файла
    
    Args:
        embeddings_path: Путь к файлу
        
    Returns:
        Словарь эмбеддингов
    """
    if not os.path.exists(embeddings_path):
        logger.warning(f"Embeddings file not found: {embeddings_path}")
        return {}
    
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    logger.info(f"Loaded embeddings from {embeddings_path}")
    return embeddings


def get_image_paths(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Получить пути ко всем изображениям в директории
    
    Args:
        directory: Путь к директории
        extensions: Расширения файлов
        
    Returns:
        Список путей к изображениям
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_paths = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return image_paths
    
    for ext in extensions:
        image_paths.extend(
            str(p) for p in Path(directory).rglob(f'*{ext}')
        )
        image_paths.extend(
            str(p) for p in Path(directory).rglob(f'*{ext.upper()}')
        )
    
    logger.info(f"Found {len(image_paths)} images in {directory}")
    return image_paths


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Сохранить результаты в JSON
    
    Args:
        results: Результаты для сохранения
        output_path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def draw_recognition_results(image: np.ndarray, results: Dict) -> np.ndarray:
    """
    Нарисовать результаты распознавания на изображение
    
    Args:
        image: Входное изображение
        results: Результаты распознавания
        
    Returns:
        Изображение с результатами
    """
    output = image.copy()
    
    for person in results["recognized_people"]:
        top, right, bottom, left = person["location"]
        name = person["name"]
        confidence = person["confidence"]
        
        # Цвет зависит от уверенности
        if confidence > 0.8:
            color = (0, 255, 0)  # Зеленый
        elif confidence > 0.5:
            color = (0, 165, 255)  # Оранжевый
        else:
            color = (0, 0, 255)  # Красный
        
        # Рисовать прямоугольник
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        
        # Рисовать текст с именем и уверенностью
        text = f"{name} ({confidence:.2f})"
        cv2.putText(output, text, (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return output


def get_face_directories(root_dir: str) -> Dict[str, List[str]]:
    """
    Получить структуру директорий известных лиц
    
    Args:
        root_dir: Корневая директория
        
    Returns:
        Словарь {name: [image_paths]}
    """
    face_dirs = {}
    
    if not os.path.exists(root_dir):
        logger.warning(f"Directory not found: {root_dir}")
        return face_dirs
    
    for person_name in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_name)
        
        if os.path.isdir(person_path):
            image_paths = get_image_paths(person_path)
            if image_paths:
                face_dirs[person_name] = image_paths
    
    logger.info(f"Found {len(face_dirs)} people in {root_dir}")
    return face_dirs


def time_function(func):
    """Декоратор для измерения времени выполнения функции"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.3f} seconds")
        return result
    
    return wrapper
