"""API contracts tested with a fake predictor, so CI needs no trained weights."""
import io
from PIL import Image
from fastapi.testclient import TestClient
from backend import main
from backend.model import CLASSES


class FakePredictor:
    def predict(self, _image):
        probs = {name: (0.91 if name == 'brown_spot' else 0.01) for name in CLASSES}
        return {'label':'brown_spot', 'confidence':0.91, 'probabilities':probs}


def make_jpeg():
    img = Image.new('RGB', (48,48), '#7ea55a')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def test_api_health_and_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'DB_PATH', tmp_path / 'test.db')
    monkeypatch.setattr(main, 'MODEL_PATH', tmp_path / 'missing_model.pth')
    with TestClient(main.app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['model_file_exists'] is False
        assert client.get('/history').json() == []


def test_reject_bad_image(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'DB_PATH', tmp_path / 'test.db')
    with TestClient(main.app) as client:
        r = client.post('/predict', files={'file': ('x.txt', b'wrong', 'text/plain')})
        assert r.status_code == 415
        r = client.post('/predict', files={'file': ('x.png', b'not a picture', 'image/png')})
        assert r.status_code == 400


def test_prediction_saved_and_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'DB_PATH', tmp_path / 'test.db')
    monkeypatch.setattr(main, 'get_predictor', lambda: FakePredictor())
    with TestClient(main.app) as client:
        r = client.post('/predict',files={'file':('rice.jpg',make_jpeg(),'image/jpeg')})
        assert r.status_code == 200, r.text
        item = r.json()
        assert item['label']=='brown_spot'
        assert item['confidence']==0.91
        assert len(item['probabilities'])==10
        history = client.get('/history').json()
        assert len(history)==1
        assert history[0]['label']=='brown_spot'
        stats = client.get('/stats').json()
        assert stats['total']==1
        assert stats['distribution']['brown_spot']==1
        assert len(stats['distribution'])==10


def test_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'DB_PATH', tmp_path / 'test.db')
    monkeypatch.setenv('PADDYCARE_API_KEY', 'unit-test-key')
    with TestClient(main.app) as client:
        assert client.get('/history').status_code == 401
        assert client.get('/history', headers={'X-API-Key':'unit-test-key'}).status_code == 200


def test_public_mode_hides_individual_history(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'DB_PATH', tmp_path / 'test.db')
    monkeypatch.setenv('PADDYCARE_PUBLIC_MODE', '1')
    with TestClient(main.app) as client:
        assert client.get('/history').status_code == 403
        assert client.get('/stats').status_code == 200
