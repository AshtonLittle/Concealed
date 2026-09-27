import io
from PIL import Image
import httpx

SERVER_URL = "http://127.0.0.1:8001"

def test_live_health():
    resp = httpx.get(f"{SERVER_URL}/api/health")
    assert resp.status_code == 200
    print("Health check OK:", resp.json())

def test_live_parameters():
    resp = httpx.get(f"{SERVER_URL}/api/parameters")
    assert resp.status_code == 200
    data = resp.json()
    print(f"Parameters catalog OK ({len(data['parameters'])} parameters):")
    for p in data["parameters"]:
        print(f"  * {p['name']} ({p['type']}): default={p['default']}")

def test_live_obfuscate_stream():
    # Create test image
    img = Image.new("RGB", (256, 256), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("input.png", buf.getvalue(), "image/png")}
    data = {
        "epsilon": "12.0",
        "mode": "hybrid",
        "texture_masking": "true",
        "chroma_damping": "0.70",
        "output_format": "PNG",
        "strip_metadata": "true",
        "response_type": "image",
    }
    resp = httpx.post(f"{SERVER_URL}/api/obfuscate", files=files, data=data)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    # Verify received image
    out_img = Image.open(io.BytesIO(resp.content))
    assert out_img.size == (256, 256)
    print("Image stream upload & receive OK!")
    print("  Headers:", {k: v for k, v in resp.headers.items() if k.startswith("x-") or k == "content-type"})

def test_live_obfuscate_json():
    img = Image.new("RGB", (128, 128), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("input.png", buf.getvalue(), "image/png")}
    data = {
        "epsilon": "20.0",
        "mode": "native",
        "target_features": "face,person",
        "conforming_mask": "true",
        "feather_radius": "16",
        "output_format": "WEBP",
        "quality": "92",
        "response_type": "json",
    }
    resp = httpx.post(f"{SERVER_URL}/api/obfuscate", files=files, data=data)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["success"] is True
    print("JSON upload response OK!")
    print(f"  MIME: {res_data['mime_type']}")
    print(f"  Format: {res_data['format']}")
    print(f"  PSNR: {res_data['analytics']['psnr_db']} dB")
    print(f"  SSIM: {res_data['analytics']['ssim']}")
    print(f"  L_inf: {res_data['analytics']['linf_255']}/255")
    print(f"  Processing time: {res_data['analytics']['processing_time_ms']} ms")

if __name__ == "__main__":
    test_live_health()
    test_live_parameters()
    test_live_obfuscate_stream()
    test_live_obfuscate_json()
    print("\nALL LIVE TESTS COMPLETED SUCCESSFULLY!")
