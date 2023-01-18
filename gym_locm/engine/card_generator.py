import json
import os
from collections import namedtuple
from typing import Dict, List, Iterable

import numpy as np

from gym_locm.engine import Creature, GreenItem, RedItem, BlueItem, Area

_card_weights = None
_rng = None

PropertyCosts = namedtuple("PropertyCosts", ["mult_cost", "add_cost"])


def _normalize_weights(weights_array: Iterable) -> List:
    sum_of_weights = sum(weights_array)

    return [weight / sum_of_weights for weight in weights_array]


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

        _card_weights["cost"] = dict(
            zip(range(13), _normalize_weights(weights_json["manaCurve"].values()))
        )

        _card_weights["area"] = dict(
            zip(
                (Area.NONE, Area.TYPE_1, Area.TYPE_2),
                _normalize_weights(
                    list(map(lambda pw: pw["weight"], weights_json["areaProbabilities"]))
                ),
            )
        )

        _card_weights["area_costs"] = dict(
            zip(
                (Area.NONE, Area.TYPE_1, Area.TYPE_2),
                map(
                    lambda p: PropertyCosts(
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["areaProbabilities"],
                ),
            )
        )

        _card_weights["keyword_count"] = dict(
            zip(
                range(7),
                _normalize_weights(weights_json["keywordNumberProbabilities"].values()),
            )
        )

        _card_weights["keywords"] = dict(
            zip(
                "BCDGLW",
                _normalize_weights(
                    list(map(lambda pw: pw["weight"], weights_json["keywordProbabilities"]))
                ),
            )
        )

        _card_weights["keywords_costs"] = dict(
            zip(
                "BCDGLW",
                map(
                    lambda p: PropertyCosts(
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
                _normalize_weights(
                    list(map(lambda pw: pw["weight"], weights_json["drawProbabilities"]))
                ),
            )
        )

        _card_weights["card_draw_costs"] = dict(
            zip(
                range(5),
                map(
                    lambda p: PropertyCosts(
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
                _normalize_weights(
                    list(map(lambda pw: pw["weight"], weights_json["myHealthProbabilities"]))
                ),
            )
        )

        _card_weights["player_hp_costs"] = dict(
            zip(
                range(4),
                map(
                    lambda p: PropertyCosts(
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
                _normalize_weights(
                    list(map(lambda pw: pw["weight"], weights_json["oppHealthProbabilities"]))
                ),
            )
        )

        _card_weights["enemy_hp_costs"] = dict(
            zip(
                range(0, -4, -1),
                map(
                    lambda p: PropertyCosts(
                        p["multCost"],
                        p["addCost"],
                    ),
                    weights_json["oppHealthProbabilities"],
                ),
            )
        )

        _card_weights["bonus_attack"] = dict(
            zip(
                ("mean", "std"),
                weights_json["bonusAttackDistribution"].values(),
            )
        )

        _card_weights["bonus_defense"] = dict(
            zip(
                ("mean", "std"),
                weights_json["bonusDefenseDistribution"].values(),
            )
        )

    return _card_weights


def _get_rng() -> np.random.Generator:
    global _rng

    if _rng is None:
        _rng = np.random.default_rng()

    return _rng


def generate_card(card_id: int = None, rng: np.random.Generator = None):
    if rng is None:
        rng = _get_rng()

    card_weights = _get_card_weights()

    card_type = rng.choice(
        list(card_weights["type"].keys()), p=list(card_weights["type"].values())
    )
    card_cost = rng.choice(
        list(card_weights["cost"].keys()), p=list(card_weights["cost"].values())
    )

    card_budget = card_cost

    properties = ["area", "card_draw", "player_hp", "enemy_hp", "keywords"]
    rng.shuffle(properties)

    chosen_properties = {
        "area": Area.NONE,
        "card_draw": 0,
        "player_hp": 0,
        "enemy_hp": 0,
        "keywords": "",
    }

    for p in properties:
        if p == "keywords":
            if card_type == BlueItem:
                continue

            number_of_keywords = rng.choice(
                list(card_weights["keyword_count"].keys()),
                p=list(card_weights["keyword_count"].values()),
            )

            chosen_keywords = rng.choice(
                list(card_weights["keywords"].keys()),
                p=list(card_weights["keywords"].values()),
                size=number_of_keywords,
                replace=False,
            )

            chosen_properties["keywords"] = []

            for keyword in chosen_keywords:
                property_weights = card_weights["keywords_costs"][keyword]
                new_card_budget = (
                    card_budget * property_weights.mult_cost - property_weights.add_cost
                )

                if new_card_budget >= 0:
                    chosen_properties["keywords"].append(keyword)
                    card_budget = new_card_budget
        else:
            chosen_value = rng.choice(
                list(card_weights[p].keys()),
                p=list(card_weights[p].values()),
            )

            property_weights = card_weights[p + "_costs"][chosen_value]
            new_card_budget = (
                card_budget * property_weights.mult_cost - property_weights.add_cost
            )

            if new_card_budget >= 0:
                chosen_properties[p] = chosen_value
                card_budget = new_card_budget

    card_attack = int(
        card_budget
        + rng.normal(
            card_weights["bonus_attack"]["mean"], card_weights["bonus_attack"]["std"]
        )
    )

    card_defense = int(
        card_budget
        + rng.normal(
            card_weights["bonus_defense"]["mean"], card_weights["bonus_defense"]["std"]
        )
    )

    if card_type == Creature:
        card_attack = max(card_attack, 0)
        card_defense = max(card_defense, 1)
    elif card_type == GreenItem:
        card_attack = max(card_attack, 0)
        card_defense = max(card_defense, 0)
    elif card_type == RedItem:
        card_attack = -max(card_attack, 0)
        card_defense = -max(card_defense, 0)
    elif card_type == BlueItem:
        card_attack = 0
        card_defense = -max(card_defense, 0)

    card = card_type(
        card_id,
        f"Card #{card_id}",
        (Creature, GreenItem, RedItem, BlueItem).index(card_type),
        card_cost,
        card_attack,
        card_defense,
        ''.join(chosen_properties["keywords"]),
        chosen_properties["player_hp"],
        chosen_properties["enemy_hp"],
        chosen_properties["card_draw"],
        chosen_properties["area"],
        "No text",  # todo: generate card text,
        instance_id=None,
    )

    return card
