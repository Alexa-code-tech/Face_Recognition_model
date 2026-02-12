"""
Real-Time Face Recognition

Recognizes faces from webcam in real-time
"""

import argparse
import logging
import cv2
import time
from typing import Optional

from models.recognition_pipeline import RecognitionPipeline
from src.utils import setup_logging, load_embeddings

logger = logging.getLogger(__name__)


class RealtimeFaceRecognition:
    """Real-time face recognition from webcam"""
    
    def __init__(self,
                 embeddings_path: str,
                 detector_name: str = "mtcnn",
                 embedder_name: str = "facenet",
                 distance_metric: str = "euclidean",
                 tolerance: float = 0.6,
                 device: str = "cuda"):
        """
        Инициализация real-time системы
        
        Args:
            embeddings_path: Путь к эмбеддингам
            detector_name: Название детектора
            embedder_name: Название эмбедера
            distance_metric: Метрика расстояния
            tolerance: Порог подобия
            device: cuda или cpu
        """
        self.pipeline = RecognitionPipeline(
            detector_name=detector_name,
            embedder_name=embedder_name,
            distance_metric=distance_metric,
            tolerance=tolerance,
            device=device
        )
        
        # Загрузить эмбеддинги
        embeddings_dict = load_embeddings(embeddings_path)
        for name, embeddings in embeddings_dict.items():
            self.pipeline.matcher.register_face(name, embeddings)
        
        logger.info(f"Loaded {len(embeddings_dict)} people for recognition")
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame):
        """
        Обработать один кадр
        
        Args:
            frame: Видеокадр
            
        Returns:
            Обработанный кадр с результатами
        """
        # Распознать лица
        results = self.pipeline.recognize(frame)
        
        # Нарисовать результаты
        output = frame.copy()
        for person in results['recognized_people']:
            top, right, bottom, left = person['location']
            name = person['name']
            confidence = person['confidence']
            
            # Цвет зависит от уверенности
            if confidence > 0.8:
                color = (0, 255, 0)  # Зеленый
            elif confidence > 0.5:
                color = (0, 165, 255)  # Оранжевый
            else:
                color = (0, 0, 255)  # Красный
            
            # Рисовать прямоугольник
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)
            
            # Рисовать текст
            text = f"{name} ({confidence:.2f})"
            cv2.putText(output, text, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Обновить FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # Рисовать FPS
        cv2.putText(output, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output
    
    def run(self, camera_id: int = 0, save_output: Optional[str] = None):
        """
        Запустить real-time распознавание
        
        Args:
            camera_id: ID камеры
            save_output: Путь для сохранения видео (опционально)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return
        
        # Получить параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Инициализировать writer если нужно сохранять
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        logger.info("Starting real-time recognition. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обработать кадр
                output = self.process_frame(frame)
                
                # Сохранить кадр если нужно
                if writer:
                    writer.write(output)
                
                # Показать результат
                cv2.imshow("Face Recognition", output)
                
                # Выход по 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info("Finished real-time recognition")


def main():
    parser = argparse.ArgumentParser(description="Real-time face recognition")
    parser.add_argument("--embeddings", type=str, default="data/embeddings/face_embeddings.pkl",
                       help="Path to embeddings file")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save output video")
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
    
    recognizer = RealtimeFaceRecognition(
        embeddings_path=args.embeddings,
        detector_name=args.detector,
        embedder_name=args.embedder,
        distance_metric=args.distance_metric,
        tolerance=args.tolerance,
        device=args.device
    )
    
    recognizer.run(camera_id=args.camera, save_output=args.output)


if __name__ == "__main__":
    main()