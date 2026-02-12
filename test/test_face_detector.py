"""
Tests for Face Detector
"""

import pytest
import numpy as np
import cv2
from models.face_detector import FaceDetector, Face


@pytest.fixture
def face_detector():
    """Fixture for face detector"""
    return FaceDetector(model_name="mtcnn")


@pytest.fixture
def sample_image():
    """Create a sample image"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_detector_initialization(face_detector):
    """Test detector initialization"""
    assert face_detector is not None
    assert face_detector.model_name == "mtcnn"


def test_detect_faces_empty_image(face_detector):
    """Test detection with empty image"""
    empty_image = np.array([])
    faces = face_detector.detect_faces(empty_image)
    assert faces == []


def test_detect_faces_none_image(face_detector):
    """Test detection with None image"""
    faces = face_detector.detect_faces(None)
    assert faces == []


def test_face_object_creation():
    """Test Face object creation"""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    box = (10, 110, 110, 10)
    confidence = 0.95
    
    face = Face(image, box, confidence)
    
    assert face.image.shape == (100, 100, 3)
    assert face.box == box
    assert face.confidence == confidence
    assert face.embedding is None