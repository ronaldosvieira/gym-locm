import json
import os
from typing import Dict

import numpy as np

_card_weights = None
_rng = None


def _get_card_weights() -> Dict:
    """
    Read the LOCM 1.5 procedural card generator weights file.
    Available at
    https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/referee1.5-java/src/main/resources/cardWeights.json
    """
    global _card_weights

    if _card_weights is None:
        with open(
            os.path.dirname(__file__) + "/resources/cardWeights.json"
        ) as weights_file:
            _card_weights = json.load(weights_file)

    return _card_weights


def _get_rng() -> np.random.Generator:
    global _rng

    if _rng is None:
        _rng = np.random.default_rng()

    return _rng


def generate_card(rng: np.random.Generator = None):
    if rng is None:
        rng = _get_rng()

    card_weights = _get_card_weights()
