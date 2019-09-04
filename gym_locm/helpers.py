from gym_locm import engine


def is_it(card_type):
    return lambda card: isinstance(card, card_type)


def has_enough_mana(available_mana):
    return lambda card: card.cost <= available_mana


def state_from_native_input(game_input):
    state = engine.State()

    game_input = iter(game_input)

    for player in state.players:
        health, mana, deck, rune, draw = map(int, next(game_input).split())

        player.health = health
        player.mana = mana
        player.next_rune = rune
        player.bonus_draw = 1 - draw

        player.hand = []

    state.phase = engine.Phase.DRAFT if mana == 0 else engine.Phase.BATTLE

    opp_hand, opp_actions = map(int, next(game_input).split())

    for _ in range(opp_actions):
        next(game_input)

    card_count = int(next(game_input))

    for _ in range(card_count):
        card_id, instance_id, location, card_type, \
            cost, attack, defense, keywords, player_hp, \
            opp_hp, card_draw, lane = next(game_input).split()

        card = engine.Card(int(card_id), "", int(card_type), int(cost),
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
