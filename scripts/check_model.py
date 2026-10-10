"""Local smoke test for the provided real checkpoint (no retraining)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from backend.model import PaddyCarePredictor

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('model', nargs='?', default='models/paddycare_model.pth')
    args=p.parse_args()
    model_file=Path(args.model)
    if not model_file.is_file():
        raise SystemExit(f'Missing checkpoint: {model_file}')
    predictor=PaddyCarePredictor(str(model_file))
    img=Image.new('RGB',(224,224), (110,145,90))
    result=predictor.predict(img)
    print('Model loaded successfully. Synthetic-input top class:',result['label'])
    print('Probability sum:',round(sum(result['probabilities'].values()),6))
    print('This is a SOFTWARE smoke test; synthetic input has no agronomic significance.')
