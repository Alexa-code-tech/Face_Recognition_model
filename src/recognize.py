"""
Face Recognition in Images

Recognizes faces in a single image and saves results
"""

import os
import argparse
import logging
import cv2
import json

from models.recognition_pipeline import RecognitionPipeline
from models.face_detector import FaceDetector
from src.utils import (
    setup_logging, load_embeddings, 
    save_results, draw_recognition_results, time_function
)

logger = logging.getLogger(__name__)


@time_function
def recognize_image(image_path: str,
                   embeddings_path: str,
                   output_path: str = None,
                   detector_name: str = "mtcnn",
                   embedder_name: str = "facenet",
                   distance_metric: str = "euclidean",
                   tolerance: float = 0.6,
                   device: str = "cuda") -> dict:
    """
    Распознать лица в изображении
    
    Args:
        image_path: Путь к изображению
        embeddings_path: Путь к сохраненным эмбеддингам
        output_path: Путь для сохранения результата
        detector_name: Название детектора
        embedder_name: Название эмбедера
        distance_metric: Метрика расстояния
        tolerance: Порог подобия
        device: cuda или cpu
        
    Returns:
        Результаты распознавания
    """
    
    # Проверить файл изображения
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    
    # Загрузить изображение
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return None
    
    logger.info(f"Processing image: {image_path}")
    
    # Инициализировать pipeline
    pipeline = RecognitionPipeline(
        detector_name=detector_name,
        embedder_name=embedder_name,
        distance_metric=distance_metric,
        tolerance=tolerance,
        device=device
    )
    
    # Загрузить известные эмбеддинги
    embeddings_dict = load_embeddings(embeddings_path)
    for name, embeddings in embeddings_dict.items():
        pipeline.matcher.register_face(name, embeddings)
    
    logger.info(f"Loaded {len(embeddings_dict)} people from {embeddings_path}")
    
    # Распознать лица
    results = pipeline.recognize(image)
    
    logger.info(f"Detected {results['faces_detected']} faces")
    for person in results['recognized_people']:
        logger.info(f"  - {person['name']}: {person['confidence']:.3f}")
    
    # Нарисовать результаты
    output_image = draw_recognition_results(image, results)
    
    # Сохранить результаты
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохранить изображение
        image_output = output_path.replace(".json", "_result.jpg")
        cv2.imwrite(image_output, output_image)
        logger.info(f"Saved result image: {image_output}")
        
        # Сохранить JSON
        save_results(results, output_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Recognize faces in an image")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--embeddings", type=str, default="data/embeddings/face_embeddings.pkl",
                       help="Path to embeddings file")
    parser.add_argument("--output", type=str, default="results/recognition_result.json",
                       help="Output path for results")
    parser.add_argument("--detector", type=str, default="mtcnn",
                       help="Face detector to use")
    parser.add_argument("--embedder", type=str, default="facenet",
                       help="Face embedder to use")
    parser.add_argument("--distance_metric", type=str, default="euclidean",
                       help="Distance metric for matching")
    parser.add_argument("--tolerance", type=float, default=0.6,
                       help="Tolerance threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    recognize_image(
        image_path=args.image,
        embeddings_path=args.embeddings,
        output_path=args.output,
        detector_name=args.detector,
        embedder_name=args.embedder,
        distance_metric=args.distance_metric,
        tolerance=args.tolerance,
        device=args.device
    )


if __name__ == "__main__":
    main()