"""Tests for Concealed Obfuscation REST API."""

import io
from PIL import Image
import numpy as np
import pytest
from starlette.testclient import TestClient

from concealed.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a synthetic 128x128 RGB test image."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Add gradients and patterns
    arr[:, :, 0] = np.linspace(0, 255, 128, dtype=np.uint8)[:, None]
    arr[:, :, 1] = np.linspace(255, 0, 128, dtype=np.uint8)[None, :]
    arr[32:96, 32:96, 2] = 200
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Concealed AI Obfuscation Engine"
    assert "endpoints" in data


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "engine" in data
    assert "PNG" in data["supported_formats"]


def test_parameters_catalog(client):
    response = client.get("/api/parameters")
    assert response.status_code == 200
    data = response.json()
    param_names = [p["name"] for p in data["parameters"]]
    assert "epsilon" in param_names
    assert "mode" in param_names
    assert "conforming_mask" in param_names
    assert "target_features" in param_names
    assert "feather_radius" in param_names
    assert "texture_masking" in param_names
    assert "chroma_damping" in param_names
    assert "output_format" in param_names
    assert "quality" in param_names


def test_obfuscate_image_default(client, sample_image_bytes):
    """Test default upload returning binary image stream."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/obfuscate", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "X-Processing-Time-Ms" in response.headers
    assert "X-Epsilon" in response.headers
    assert float(response.headers["X-Epsilon"]) == 8.0
    assert "X-PSNR-dB" in response.headers

    # Validate returned image is valid PIL image
    result_img = Image.open(io.BytesIO(response.content))
    assert result_img.size == (128, 128)
    assert result_img.mode == "RGB"


def test_obfuscate_image_custom_parameters(client, sample_image_bytes):
    """Test upload with custom epsilon, JPEG output, chroma damping, and canonical mode."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    data = {
        "epsilon": "16.0",
        "mode": "canonical_residual",
        "output_format": "JPEG",
        "quality": "85",
        "chroma_damping": "0.85",
        "strip_metadata": "true",
        "canonical_size": "256",
    }
    response = client.post("/api/obfuscate", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert float(response.headers["X-Epsilon"]) == 16.0
    assert response.headers["X-Mode"] == "canonical_residual"

    result_img = Image.open(io.BytesIO(response.content))
    assert result_img.size == (128, 128)


def test_obfuscate_conforming_mask(client, sample_image_bytes):
    """Test conforming silhouette mask targeting face/person with feathering."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    data = {
        "epsilon": "12.0",
        "mode": "native",
        "target_features": "face,person",
        "conforming_mask": "true",
        "feather_radius": "10",
        "texture_masking": "true",
    }
    response = client.post("/api/obfuscate", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    result_img = Image.open(io.BytesIO(response.content))
    assert result_img.size == (128, 128)


def test_obfuscate_image_json_response(client, sample_image_bytes):
    """Test multipart upload requesting response_type=json."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    data = {
        "epsilon": "10.0",
        "output_format": "WEBP",
        "quality": "90",
        "response_type": "json",
    }
    response = client.post("/api/obfuscate", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["format"] == "WEBP"
    assert res_data["mime_type"] == "image/webp"
    assert res_data["image_base64"].startswith("data:image/webp;base64,")
    assert "analytics" in res_data
    assert res_data["analytics"]["psnr_db"] > 0
    assert res_data["analytics"]["original_resolution"] == [128, 128]


def test_obfuscate_json_endpoint(client, sample_image_bytes):
    """Test pure JSON base64 endpoint."""
    import base64
    b64_in = base64.b64encode(sample_image_bytes).decode("ascii")
    payload = {
        "image_base64": f"data:image/png;base64,{b64_in}",
        "epsilon": 14.0,
        "mode": "hybrid",
        "output_format": "PNG",
        "texture_masking": True,
        "chroma_damping": 0.75,
    }
    response = client.post("/api/obfuscate/json", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["image_base64"].startswith("data:image/png;base64,")
    assert res_data["parameters"]["epsilon"] == 14.0
    assert "analytics" in res_data


def test_concealed_client():
    """Test ConcealedClient Python SDK."""
    from concealed import ConcealedClient

    client = ConcealedClient("http://127.0.0.1:8001")
    health = client.check_health()
    assert health["status"] == "healthy"

    test_img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    obf_img, analytics = client.obfuscate_with_analytics(
        image=test_img,
        epsilon=10.0,
        mode="native",
        target_features="face",
        conforming_mask=True,
        feather_radius=6,
        output_format="PNG",
    )
    assert obf_img.size == (64, 64)
    assert analytics["epsilon"] == 10.0
    assert analytics["mode"] == "native"

