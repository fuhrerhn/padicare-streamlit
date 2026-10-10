import os
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

from .constants import CLASSES

class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class PaddyCarePredictor:
    def __init__(self, path: str):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['model_architecture'] == 'efficientnet_b0'
        assert int(checkpoint['feature_dim']) == 1280
        assert int(checkpoint['num_classes']) == len(CLASSES)
        assert checkpoint['class_to_idx'] == {name: i for i, name in enumerate(CLASSES)}
        self.transform = transforms.Compose([
            transforms.Resize((int(checkpoint['image_size']), int(checkpoint['image_size']))),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint['mean'], std=checkpoint['std']),
        ])
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        if 'backbone_state_dict' not in checkpoint:
            raise ValueError('Checkpoint hanya berisi classifier head. Jalankan sel ekspor backbone dari README pada sesi Kaggle yang memiliki bobot pretrained.')
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=True)
        self.head = ClassifierHead(1280, len(CLASSES), float(checkpoint['dropout']))
        self.head.load_state_dict(checkpoint['classifier_state_dict'], strict=True)
        self.backbone.eval()
        self.head.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        x = self.transform(image.convert('RGB')).unsqueeze(0)
        scores = torch.softmax(self.head(self.backbone(x)), dim=1)[0]
        index = int(scores.argmax().item())
        return {'label': CLASSES[index], 'confidence': round(float(scores[index]), 6),
                'probabilities': {label: round(float(value), 6) for label, value in zip(CLASSES, scores.tolist())}}

@lru_cache(maxsize=1)
def get_predictor():
    return PaddyCarePredictor(os.getenv('MODEL_PATH', str(Path(__file__).resolve().parent.parent / 'models' / 'paddycare_model.pth')))
