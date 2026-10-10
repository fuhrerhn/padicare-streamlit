"""Validate serialization and inference with synthetic weights.
This test does NOT validate agricultural accuracy.
"""
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from PIL import Image
from backend.model import CLASSES, ClassifierHead, PaddyCarePredictor


def test_model_roundtrip(tmp_path):
    backbone = efficientnet_b0(weights=None)
    backbone.classifier = nn.Identity()
    head = ClassifierHead(1280,10,0.3)
    ckpt = {
        'model_architecture':'efficientnet_b0',
        'feature_dim':1280,
        'num_classes':10,
        'dropout':0.3,
        'backbone_state_dict':backbone.state_dict(),
        'classifier_state_dict':head.state_dict(),
        'class_to_idx':{name:i for i,name in enumerate(CLASSES)},
        'image_size':224,
        'mean':[0.485,0.456,0.406],
        'std':[0.229,0.224,0.225],
    }
    path = tmp_path/'mock.pth'
    torch.save(ckpt,path)
    predictor = PaddyCarePredictor(str(path))
    p = predictor.predict(Image.new('RGB',(180,160),'green'))
    assert p['label'] in CLASSES
    assert len(p['probabilities'])==10
    assert abs(sum(p['probabilities'].values())-1)<0.0001
