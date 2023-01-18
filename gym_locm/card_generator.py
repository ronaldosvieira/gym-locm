import json
import os
from typing import Dict


def load_card_weights() -> Dict:
    """
    Read the LOCM 1.5 procedural card generator weights file.
    Available at
    https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/referee1.5-java/src/main/resources/cardWeights.json
    """
    with open(os.path.dirname(__file__) + "/resources/cardWeights.json") as weights_file:
        weights_json = json.load(weights_file)

    return weights_json


_card_weights = load_card_weights()
