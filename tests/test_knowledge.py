from backend.knowledge import KNOWLEDGE, get_advice
from backend.model import CLASSES


def test_knowledge_coverage():
    assert set(CLASSES) == set(KNOWLEDGE)
    for label in CLASSES:
        for lang in ('id','en'):
            a = get_advice(label,lang)
            for key in ('name','overview','signs','initial','monitor','prevent','sources'):
                assert a[key], (label,lang,key)
            for source in a['sources']:
                assert source['url'].startswith('https://')


def test_pest_vs_disease():
    assert KNOWLEDGE['hispa']['kind']=='pest'
    assert KNOWLEDGE['dead_heart']['kind']=='pest_symptom'
    assert KNOWLEDGE['normal']['kind']=='healthy'
