import numpy as np


def is_it(card_type):
    return lambda card: isinstance(card, card_type)


def has_enough_mana(available_mana):
    return lambda card: card.cost <= available_mana


def encode_card(card):
    """Encodes a card object into a numerical array."""

    card_type = [1.0 if card.type == i else 0.0 for i in range(4)]
    cost = card.cost / 12
    attack = card.attack / 12
    defense = max(-12, card.defense) / 12
    keywords = list(map(int, map(card.keywords.__contains__, 'BCDGLW')))
    player_hp = card.player_hp / 12
    enemy_hp = card.enemy_hp / 12
    card_draw = card.card_draw / 2

    return card_type + [cost, attack, defense, player_hp,
                        enemy_hp, card_draw] + keywords
