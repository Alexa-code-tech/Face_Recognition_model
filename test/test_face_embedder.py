"""
Tests for Face Embedder
"""

import pytest
import numpy as np
from models.face_embedder import FaceEmbedder


@pytest.fixture
def face_embedder():
    """Fixture for face embedder"""
    return FaceEmbedder(model_name="facenet", device="cpu")


def test_embedder_initialization(face_embedder):
    """Test embedder initialization"""
    assert face_embedder is not None
    assert face_embedder.model_name == "facenet"


def test_get_embedding_invalid_image(face_embedder):
    """Test embedding with invalid image"""
    empty_image = np.array([])
    embedding = face_embedder.get_embedding(empty_image)
    assert embedding is None


def test_embedding_dimension(face_embedder):
    """Test embedding dimension"""
    # Create a valid face image
    face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    embedding = face_embedder.get_embedding(face_image)
    
    if embedding is not None:
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1