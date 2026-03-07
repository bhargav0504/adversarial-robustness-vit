"""
realworld/classify.py
Top-k prediction + automatic category labelling (Human / Animal / Plant / Non-Living).
"""
import torch
from torchvision.models import ResNet18_Weights

# Load ImageNet class names from torchvision
IMAGENET_CLASSES = ResNet18_Weights.IMAGENET1K_V1.meta['categories']

# ── Keyword lists ─────────────────────────────────────────────────────────────
_HUMAN = [
    'person', 'man', 'woman', 'boy', 'girl', 'child', 'human', 'face',
    'soldier', 'cowboy', 'groom', 'bride', 'nurse', 'doctor', 'scuba diver',
    'baseball player', 'tennis player', 'rugby player', 'soccer player',
]

_ANIMAL = [
    # Dog breeds
    'retriever', 'spaniel', 'terrier', 'hound', 'poodle', 'shepherd',
    'collie', 'setter', 'pointer', 'bulldog', 'beagle', 'dachshund',
    'husky', 'labrador', 'dalmatian', 'pinscher', 'mastiff', 'bloodhound',
    'basset', 'boxer', 'samoyed', 'chow', 'malamute', 'doberman',
    'rottweiler', 'pomeranian', 'chihuahua', 'corgi', 'shiba', 'akita',
    # Cats
    'cat', 'tabby', 'siamese', 'persian', 'lion', 'tiger', 'cheetah',
    'leopard', 'jaguar', 'cougar', 'lynx', 'puma',
    # Birds
    'bird', 'finch', 'jay', 'robin', 'wren', 'sparrow', 'thrush',
    'bunting', 'warbler', 'pigeon', 'dove', 'woodpecker', 'kingfisher',
    'toucan', 'macaw', 'cockatoo', 'albatross', 'pelican', 'flamingo',
    'parrot', 'ostrich', 'penguin', 'hen', 'cock', 'duck', 'goose',
    'eagle', 'hawk', 'owl', 'crane', 'heron', 'stork', 'ibis', 'kite',
    # Reptiles / Amphibians
    'snake', 'turtle', 'lizard', 'frog', 'toad', 'salamander',
    'iguana', 'gecko', 'chameleon', 'alligator', 'crocodile',
    # Fish / Aquatic
    'fish', 'shark', 'ray', 'eel', 'salmon', 'tuna', 'goldfish',
    'carp', 'trout', 'lobster', 'crab', 'shrimp', 'starfish',
    'jellyfish', 'octopus', 'squid', 'whale', 'dolphin', 'seal',
    'walrus', 'otter',
    # Mammals
    'dog', 'horse', 'bear', 'wolf', 'fox', 'monkey', 'elephant',
    'giraffe', 'bee', 'ant', 'butterfly', 'spider',
    'panda', 'koala', 'kangaroo', 'zebra', 'rhinoceros', 'hippopotamus',
    'hamster', 'rabbit', 'squirrel', 'mouse', 'rat', 'pig', 'cow',
    'sheep', 'goat', 'deer', 'camel', 'llama', 'elk', 'moose',
    'bison', 'ox', 'mink', 'weasel', 'badger', 'skunk', 'raccoon',
    # Insects
    'beetle', 'moth', 'dragonfly', 'cricket', 'grasshopper',
    'cockroach', 'caterpillar',
]

_PLANT = [
    'flower', 'plant', 'tree', 'mushroom', 'fungus',
    'coral', 'seaweed', 'daisy', 'sunflower', 'rose',
    'dandelion', 'lotus', 'bouquet',
]


# Pre-compute category for all 1000 ImageNet classes at startup
def _matches(name: str, keywords: list) -> bool:
    """
    Match keywords against a class name using word boundaries for single-word
    keywords (avoids 'man' matching 'German') and substring for multi-word ones.
    """
    n = name.lower()
    words = set(n.replace(',', ' ').replace('-', ' ').replace("'", ' ').split())
    for kw in keywords:
        if ' ' in kw:          # multi-word: substring match
            if kw in n:
                return True
        else:                  # single-word: exact word match
            if kw in words:
                return True
    return False


def _build_category_map():
    mapping = {}
    for idx, name in enumerate(IMAGENET_CLASSES):
        if _matches(name, _HUMAN):
            mapping[idx] = 'Human'
        elif _matches(name, _ANIMAL):
            mapping[idx] = 'Animal'
        elif _matches(name, _PLANT):
            mapping[idx] = 'Plant'
        else:
            mapping[idx] = 'Non-Living Object'
    return mapping


CATEGORY_MAP = _build_category_map()


def get_category(class_name: str) -> str:
    if _matches(class_name, _HUMAN):
        return 'Human'
    if _matches(class_name, _ANIMAL):
        return 'Animal'
    if _matches(class_name, _PLANT):
        return 'Plant'
    return 'Non-Living Object'


def predict_topk(model, tensor: torch.Tensor, k: int = 5):
    """Run inference and return top-k results as a list of dicts."""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_probs, top_indices = probs.topk(k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        name = IMAGENET_CLASSES[idx]
        results.append({
            'rank':       len(results) + 1,
            'class':      name,
            'confidence': round(prob * 100, 2),
            'category':   CATEGORY_MAP[idx],
            'idx':        idx,
        })
    return results


def format_predictions(predictions: list) -> str:
    lines = []
    for p in predictions:
        bar = '█' * max(1, int(p['confidence'] / 5))
        lines.append(
            f"#{p['rank']}  {p['class']:<35} {p['confidence']:5.1f}%  {bar}  [{p['category']}]"
        )
    return '\n'.join(lines)
