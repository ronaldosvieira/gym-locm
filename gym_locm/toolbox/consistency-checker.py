import json
from typing import List

from gym_locm.engine import State, Phase
from gym_locm.agents import NativeAgent
from gym_locm.exceptions import ActionError


def extract_match_json(match_line):
    json_ending_index = match_line.rfind("}")
    match_json = json.loads(match_line[: json_ending_index + 1])

    return match_json


def extract_match_transitions(match_json):
    player1_states = list(filter(bool, match_json["errors"]["0"]))
    player2_states = list(filter(bool, match_json["errors"]["1"]))
    player1_actions = list(map(str.rstrip, filter(bool, match_json["outputs"]["0"])))
    player2_actions = list(map(str.rstrip, filter(bool, match_json["outputs"]["1"])))

    return dict(
        player1=dict(states=player1_states, actions=player1_actions),
        player2=dict(states=player2_states, actions=player2_actions),
    )


def _find_deck_order(match_transitions, player):
    known_instance_ids = set()
    deck_order = []

    for battle_state in match_transitions[player]["states"][30:]:
        state = State.from_native_input(battle_state)

        for card in state.current_player.hand:
            if card.instance_id not in known_instance_ids:
                known_instance_ids.add(card.instance_id)
                deck_order.append((card.id, card.instance_id))

    return deck_order


def find_deck_orders(match_transitions):
    return _find_deck_order(match_transitions, "player1"), _find_deck_order(
        match_transitions, "player2"
    )


def find_draft_options(match_transitions):
    draft = []

    for draft_state in match_transitions["player1"]["states"][:30]:
        state = State.from_native_input(draft_state)

        draft.append(state.current_player.hand)

    return draft


def recreate_initial_state(match_transitions, draft_options, deck_orders, phase=Phase.DECK_BUILDING):
    if phase == Phase.DECK_BUILDING:
        start_index = 0
    else:
        start_index = 30

    state = State.from_native_input(
        match_transitions["player1"]["states"][start_index],
        deck_orders,
    )

    if phase == Phase.DECK_BUILDING:
        state._phase._draft_cards = draft_options
    else:
        state.opposing_player.base_mana -= 1
        state.opposing_player.bonus_mana += 1


    return state


def validate_match(match_transitions, state, p1_agent, p2_agent, phase=Phase.DECK_BUILDING):
    if phase == Phase.DECK_BUILDING:
        start_index = 0
    else:
        start_index = 30

    p1_states_iter = iter(match_transitions["player1"]["states"][start_index:])
    p1_actions_iter = iter(match_transitions["player1"]["actions"][start_index:])
    p2_states_iter = iter(match_transitions["player2"]["states"][start_index:])
    p2_actions_iter = iter(match_transitions["player2"]["actions"][start_index:])

    states_iter = p1_states_iter, p2_states_iter
    actions_iter = p1_actions_iter, p2_actions_iter
    agents = p1_agent, p2_agent

    while state.phase == phase:
        player = state.current_player.id

        original_state = next(states_iter[player]).strip()
        recreated_state = str(state).strip()

        print(f"Turn {state.turn}, player {player}")
        print(state)

        remove_trailing_pass_action = state.is_battle()

        assert (
            original_state == recreated_state
        ), f"{original_state}\n{recreated_state}"

        print("States match!")

        original_actions: List[str] = list(
            map(str.strip, next(actions_iter[player]).split(";"))
        )
        recreated_actions = []

        print("Original actions:", original_actions)

        while state.current_player.id == player and state.phase == phase:
            action = agents[player].act(state)
            recreated_actions.append(str(action))

            try:
                state.act(action)

                print(action, "✅")
            except ActionError as e:
                print(action, "❌", e)

        if state.phase != phase:
            break

        if remove_trailing_pass_action and len(recreated_actions) > 1:
            recreated_actions.pop()

        print("Recreated actions:", recreated_actions)

        assert len(original_actions) == len(
            recreated_actions
        ), f"{len(original_actions)} != {len(recreated_actions)}"

        assert all(
            [oa == ra for oa, ra in zip(original_actions, recreated_actions)]
        ), f"{original_actions}\n{recreated_actions}"

        print("Actions match!")


def run():
    dataset_path = "gym_locm/engine/resources/consistency-dataset-1.txt"

    with open(dataset_path, "r") as dataset:
        matches = dataset.readlines()

    print(len(matches))

    p1_agent = NativeAgent(
        "python3 gym_locm/engine/resources/baseline2.py",
        verbose=False,
    )
    p2_agent = NativeAgent(
        "python3 gym_locm/engine/resources/baseline2.py",
        verbose=False,
    )

    for i, match in enumerate(matches):

        p1_agent.reset()
        p2_agent.reset()

        print(f"Match #{i}")

        match_json = extract_match_json(match)

        print(match_json)

        transitions = extract_match_transitions(match_json)

        p1 = transitions["player1"]["states"]
        p2 = transitions["player2"]["states"]
        p1a = transitions["player1"]["actions"]
        p2a = transitions["player2"]["actions"]

        print(len(p1), p1[30:])
        print(len(p2), p2[30:])
        print(len(p1a), p1a[30:])
        print(len(p2a), p2a[30:])

        deck_orders = find_deck_orders(transitions)
        print(deck_orders)

        draft_options = find_draft_options(transitions)

        state = recreate_initial_state(transitions, draft_options, deck_orders, Phase.DRAFT)

        validate_match(transitions, state, p1_agent, p2_agent, Phase.DRAFT)

        state = recreate_initial_state(transitions, draft_options, deck_orders, Phase.BATTLE)

        validate_match(transitions, state, p1_agent, p2_agent, Phase.BATTLE)

    p1_agent.close()
    p2_agent.close()


if __name__ == "__main__":
    run()
