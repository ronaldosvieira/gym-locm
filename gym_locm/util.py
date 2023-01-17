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
    keywords = list(map(int, map(card.keywords.__contains__, "BCDGLW")))
    player_hp = card.player_hp / 12
    enemy_hp = card.enemy_hp / 12
    card_draw = card.card_draw / 2

    return (
        card_type + [cost, attack, defense, player_hp, enemy_hp, card_draw] + keywords
    )


def encode_state_draft(
    state, use_history=False, use_mana_curve=False, past_choices=None
):
    card_features = 16
    current_card_choices = state.k
    state_size = card_features * current_card_choices

    if use_history:
        state_size += card_features * state.n
        assert (
            past_choices is not None
        ), "If encoding the draft history, past_choices should not be None."

    if use_mana_curve:
        state_size += 13
        assert (
            past_choices is not None
        ), "If encoding the mana curve, past_choices should not be None."

    encoded_state = np.full((state_size,), 0, dtype=np.float32)

    # if draft is not over, fill current choices
    if state.is_draft():
        card_choices = state.current_player.hand[0 : state.k]

        for i in range(len(card_choices)):
            lo = -(state.k - i) * card_features
            hi = lo + card_features
            hi = hi if hi < 0 else None

            encoded_state[lo:hi] = encode_card(card_choices[i])

    # if using history, fill past choices
    if use_history:
        for j, card in enumerate(past_choices):
            lo = -(state.n + state.k - j) * card_features
            hi = lo + card_features
            hi = hi if hi < 0 else None

            encoded_state[lo:hi] = encode_card(card)

    # if using mana curve, fill mana curve slots
    if use_mana_curve:
        for chosen_card in past_choices:
            encoded_state[chosen_card.cost] += 1

    return encoded_state
