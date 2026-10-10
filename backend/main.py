"""PaddyCare REST API: inference, private/local SQLite history, analytics."""
import io
import json
import os
import sqlite3
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from .model import CLASSES, get_predictor

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv('DB_PATH', ROOT / 'data' / 'paddycare.db'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', ROOT / 'models' / 'paddycare_model.pth'))
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 24_000_000
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/webp'}


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=15)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities_json TEXT NOT NULL
        )''')
        con.execute('CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)')


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title='PaddyCare AI REST API', description='Rice health image classification, ten classes.',
              version='2.0.0', lifespan=lifespan)
# UI calls the API from the Streamlit Python server; browser CORS is not required.
# Override only with a tightly controlled origin if building a browser JS client.


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv('PADDYCARE_API_KEY', '')
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail='Invalid or missing API key')


@app.get('/health')
def health():
    return {'status': 'ok', 'model_file_exists': MODEL_PATH.is_file(), 'version': app.version}


@app.get('/classes')
def classes(_=Depends(require_api_key)):
    return {'classes': CLASSES}


@app.post('/predict')
async def predict(file: UploadFile = File(...), _=Depends(require_api_key)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail='Only JPEG, PNG or WEBP files are supported')
    data = await file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail='Empty image file')
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail='Image must be 8MB or smaller')
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image_input:
            if image_input.width * image_input.height > MAX_PIXELS:
                raise HTTPException(status_code=413, detail='Image has too many pixels')
            image = ImageOps.exif_transpose(image_input).convert('RGB')
    except (Image.DecompressionBombError, Image.DecompressionBombWarning,
            UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail='Invalid or unsafe image file')

    try:
        output = get_predictor().predict(image)
    except (FileNotFoundError, OSError, KeyError, AssertionError, RuntimeError, ValueError) as exc:
        # Model details should not leak into public error messages.
        raise HTTPException(status_code=503, detail='Model is unavailable or incompatible. Check server setup.') from exc

    identifier = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as con:
        con.execute('''INSERT INTO predictions
                       (id, created_at, label, confidence, probabilities_json)
                       VALUES (?, ?, ?, ?, ?)''',
                    (identifier, created_at, output['label'], output['confidence'],
                     json.dumps(output['probabilities'], separators=(',', ':'))))
    return {'id': identifier, 'created_at': created_at, **output}


@app.get('/history')
def history(limit: int = Query(20, ge=1, le=100), _=Depends(require_api_key)):
    if os.getenv('PADDYCARE_PUBLIC_MODE', '').lower() in {'1','true','yes'}:
        raise HTTPException(status_code=403, detail='Per-record history is disabled in public mode')
    with connect() as con:
        rows = con.execute('''SELECT id, created_at, label, confidence
                              FROM predictions ORDER BY created_at DESC LIMIT ?''', (limit,)).fetchall()
    return [dict(row) for row in rows]


@app.get('/stats')
def stats(_=Depends(require_api_key)):
    with connect() as con:
        total = con.execute('SELECT COUNT(*) AS n FROM predictions').fetchone()['n']
        rows = con.execute('''SELECT label, COUNT(*) AS n
                              FROM predictions GROUP BY label ORDER BY n DESC''').fetchall()
    distribution = {c: 0 for c in CLASSES}
    for row in rows:
        distribution[row['label']] = row['n']
    return {'total': total, 'distribution': distribution}
