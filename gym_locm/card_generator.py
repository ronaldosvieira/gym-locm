import json
import os
from collections import namedtuple
from typing import Dict

import numpy as np

from gym_locm.engine import Creature, GreenItem, RedItem, BlueItem, Area

_card_weights = None
_rng = None

PropertyWeights = namedtuple("PropertyWeights", ["weight", "mult_cost", "add_cost"])


def _get_card_weights() -> Dict:
    """
    Read the LOCM 1.5 procedural card generator weights file.
    Available at
    https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/referee1.5-java/src/main/resources/cardWeights.json
    """
    global _card_weights

    if _card_weights is None:
        _card_weights = dict()

        with open(
            os.path.dirname(__file__) + "/resources/cardWeights.json"
        ) as weights_file:
            weights_json = json.load(weights_file)

        _card_weights["type"] = dict(
            zip(
                (Creature, GreenItem, RedItem, BlueItem),
                weights_json["typeProbabilities"].values(),
            )
        )

        _card_weights["cost"] = dict(zip(range(13), weights_json["manaCurve"].values()))

        _card_weights["area"] = dict(
            zip(
                (Area.NONE, Area.TYPE_1, Area.TYPE_2),
                map(
                    lambda p: PropertyWeights(
                        p["weight"],
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["areaProbabilities"],
                ),
            )
        )

        _card_weights["keyword_count"] = dict(
            zip(range(7), weights_json["keywordNumberProbabilities"].values())
        )

        _card_weights["keywords"] = dict(
            zip(
                "BCDGLW",
                map(
                    lambda p: PropertyWeights(
                        p["weight"],
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["keywordProbabilities"],
                ),
            )
        )

        _card_weights["card_draw"] = dict(
            zip(
                range(5),
                map(
                    lambda p: PropertyWeights(
                        p["weight"],
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["drawProbabilities"],
                ),
            )
        )

        _card_weights["player_hp"] = dict(
            zip(
                range(4),
                map(
                    lambda p: PropertyWeights(
                        p["weight"],
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["myHealthProbabilities"],
                ),
            )
        )

        _card_weights["enemy_hp"] = dict(
            zip(
                range(0, -4, -1),
                map(
                    lambda p: PropertyWeights(
                        p["weight"],
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["oppHealthProbabilities"],
                ),
            )
        )

        _card_weights["bonus_attack"] = dict(
            zip(("mean", "std"), weights_json["bonusAttackDistribution"].values())
        )

        _card_weights["bonus_defense"] = dict(
            zip(("mean", "std"), weights_json["bonusDefenseDistribution"].values())
        )

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
