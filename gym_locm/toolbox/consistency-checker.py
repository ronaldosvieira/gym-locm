import json
from gym_locm.engine import State
from gym_locm.agents import NativeBattleAgent

def extract_match_json(match_line):
    json_ending_index = match_line.rfind('}')
    match_json = json.loads(match_line[:json_ending_index + 1])

    return match_json


def extract_match_transitions(match_json):
    player1_states = list(filter(bool, match_json['errors']['0']))
    player2_states = list(filter(bool, match_json['errors']['1']))
    player1_actions = list(map(str.rstrip, filter(bool, match_json['outputs']['0'])))
    player2_actions = list(map(str.rstrip, filter(bool, match_json['outputs']['1'])))

    return dict(player1=dict(states=player1_states, actions=player1_actions),
                player2=dict(states=player2_states, actions=player2_actions))


def _find_deck_order(match_transitions, player):
    known_instance_ids = set()
    deck_order = []

    for battle_state in match_transitions[player]['states'][30:]:
        state = State.from_native_input(battle_state)

        for card in state.current_player.hand:
            if card.instance_id not in known_instance_ids:
                known_instance_ids.add(card.instance_id)
                deck_order.append((card.id, card.instance_id))

    return deck_order


def find_deck_orders(match_transitions):
    return _find_deck_order(match_transitions, 'player1'), \
           _find_deck_order(match_transitions, 'player2')


def recreate_initial_state(match_transitions, deck_orders):
    state = State.from_native_input(match_transitions['player1']['states'][30], deck_orders)

    state.opposing_player.base_mana -= 1
    state.opposing_player.bonus_mana += 1

    return state


def validate_match(match_transitions, state, p1_agent, p2_agent):
    p1_states_iter = iter(match_transitions['player1']['states'][30:])
    p1_actions_iter = iter(match_transitions['player1']['actions'][30:])
    p2_states_iter = iter(match_transitions['player2']['states'][30:])
    p2_actions_iter = iter(match_transitions['player2']['actions'][30:])

    states_iter = p1_states_iter, p2_states_iter
    actions_iter = p1_actions_iter, p2_actions_iter
    agents = p1_agent, p2_agent

    while not state.is_ended():
        player = state.current_player.id

        original_state = next(states_iter[player]).strip()
        recreated_state = str(state).strip()

        print(f"Turn {state.turn}, player {player}")
        print(state)

        try:
            assert original_state == recreated_state, f"{original_state}\n{recreated_state}"
        except AssertionError:
            return False

        original_actions = list(map(str.strip, next(actions_iter[player]).split(';')))
        recreated_actions = []

        while state.current_player.id == player and not state.is_ended():
            action = agents[player].act(state)
            recreated_actions.append(str(action))

            state.act(action)

        if state.is_ended():
            break

        # remove pass action
        recreated_actions.pop()

        print(recreated_actions)

        try:
            assert all([oa == ra for oa, ra in zip(original_actions, recreated_actions)]), \
            f"{original_actions}\n{recreated_actions}"
        except AssertionError:
            return False

    return True


def run():
    dataset_path = '/home/ronaldo/Projects/Strategy-Card-Game-AI-Competition/contest-2021-08-COG/' \
                   'baseline2-battle-dataset.txt'

    with open(dataset_path, 'r') as dataset:
        matches = dataset.readlines()

    print(len(matches))
    match_json = extract_match_json(matches[2])

    print(match_json)

    transitions = extract_match_transitions(match_json)

    p1 = transitions['player1']['states']
    p2 = transitions['player2']['states']
    p1a = transitions['player1']['actions']
    p2a = transitions['player2']['actions']

    print(len(p1), p1[30:])
    print(len(p2), p2[30:])
    print(len(p1a), p1a[30:])
    print(len(p2a), p2a[30:])

    deck_orders = find_deck_orders(transitions)
    print(deck_orders)

    state = recreate_initial_state(transitions, deck_orders)

    p1_agent = NativeBattleAgent('python /home/ronaldo/Projects/Strategy-Card-Game-AI-Competition/contest-2020-08-COG'
                                 '/Baseline2/main.py', verbose=False)
    p2_agent = NativeBattleAgent('python /home/ronaldo/Projects/Strategy-Card-Game-AI-Competition/contest-2020-08-COG'
                                 '/Baseline2/main.py', verbose=False)

    result = validate_match(transitions, state, p1_agent, p2_agent)

    print("Matches are consistent?", result)

    p1_agent.close()
    p2_agent.close()


if __name__ == '__main__':
    run()
