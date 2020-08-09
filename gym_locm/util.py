from gym_locm import engine


def is_it(card_type):
    return lambda card: isinstance(card, card_type)


def has_enough_mana(available_mana):
    return lambda card: card.cost <= available_mana


def mockup_card():
    return engine.Card(0, "", 0, 0, 0, 0, "------", 0, 0, 0, "",
                       instance_id=None)


def state_from_native_input(game_input):
    state = engine.State()

    game_input = iter(game_input)

    for i, player in enumerate(state.players):
        health, mana, deck, rune, draw = map(int, next(game_input).split())

        player.health = health
        player.mana = mana
        player.base_mana = mana
        player.next_rune = rune
        player.bonus_draw = 0 if i == 0 else draw - 1
        player.last_drawn = draw if i == 0 else 1

        player.hand = []
        player.deck = [mockup_card() for _ in range(deck)]

    state.phase = engine.Phase.DRAFT if mana == 0 else engine.Phase.BATTLE

    opp_hand, opp_actions = map(int, next(game_input).split())

    state.opposing_player.hand = [mockup_card() for _ in range(opp_hand)]

    for _ in range(opp_actions):
        next(game_input)

    card_count = int(next(game_input))

    for _ in range(card_count):
        card_id, instance_id, location, card_type, \
            cost, attack, defense, keywords, player_hp, \
            opp_hp, card_draw, lane = next(game_input).split()

        card_type = int(card_type)

        types_dict = {0: engine.Creature, 1: engine.GreenItem,
                      2: engine.RedItem, 3: engine.BlueItem}

        card_class = types_dict[card_type]

        card = card_class(int(card_id), "", card_type, int(cost),
                          int(attack), int(defense), keywords,
                          int(player_hp), int(opp_hp), int(card_draw),
                          "", instance_id=int(instance_id))

        location = int(location)
        lane = int(lane)

        if location == 0:
            state.players[0].hand.append(card)
        elif location == 1:
            state.players[0].lanes[lane].append(card)
        elif location == -1:
            state.players[1].lanes[lane].append(card)

    return state


def encode_card(card):
    """Encodes a card object into a numerical array."""
    card_types = (engine.Creature, engine.GreenItem,
                  engine.RedItem, engine.BlueItem)

    card_type = [1.0 if isinstance(card, card_type) else 0.0
                 for card_type in card_types]
    cost = card.cost / 12
    attack = card.attack / 12
    defense = max(-12, card.defense) / 12
    keywords = list(map(int, map(card.keywords.__contains__, 'BCDGLW')))
    player_hp = card.player_hp / 12
    enemy_hp = card.enemy_hp / 12
    card_draw = card.card_draw / 12

    return card_type + [cost, attack, defense, player_hp,
                        enemy_hp, card_draw] + keywords
