"""
Pytest Configuration and Fixtures

Shared fixtures for health screening pipeline tests.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Generate a sample video frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_pose_array() -> np.ndarray:
    """Generate sample pose landmarks."""
    # 33 landmarks, 4 values each (x, y, z, visibility)
    return np.random.rand(33, 4).astype(np.float32)


@pytest.fixture
def sample_ris_data() -> np.ndarray:
    """Generate sample RIS data."""
    return np.random.rand(100, 16).astype(np.float32) * 500 + 400


@pytest.fixture
def data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def temp_session_id() -> str:
    """Generate a temporary session ID."""
    import uuid
    return str(uuid.uuid4())
