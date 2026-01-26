import argparse
import json
from typing import List

from gym_locm.engine import State, Phase
from gym_locm.agents import NativeAgent
from gym_locm.exceptions import ActionError


def get_arg_parser():
    p = argparse.ArgumentParser(
        description="This is runner script for agent experimentation on gym-locm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--agent",
        help="agent to be used to play the matches (should be Baseline2 for LOCM 1.2, and ByteRL for LOCM 1.5)",
    )
    p.add_argument(
        "--version", help="version to check", choices=["1.2", "1.5"], default="1.5"
    )

    return p


def extract_match_json(match_line):
    json_ending_index = match_line.rfind("}")
    match_json = json.loads(match_line[: json_ending_index + 1])

    return match_json


def extract_match_transitions(match_json, version="1.5"):
    player1_states = list(filter(bool, match_json["errors"]["0"]))
    player2_states = list(filter(bool, match_json["errors"]["1"]))
    player1_actions = list(map(str.rstrip, filter(bool, match_json["outputs"]["0"])))
    player2_actions = list(map(str.rstrip, filter(bool, match_json["outputs"]["1"])))

    if version == "1.5":
        player1_states[0] += player1_states[1]
        player1_states.remove(player1_states[1])

        player2_states[0] += player2_states[1]
        player2_states.remove(player2_states[1])

    return dict(
        player1=dict(states=player1_states, actions=player1_actions),
        player2=dict(states=player2_states, actions=player2_actions),
    )


def _find_deck_order(match_transitions, player, version="1.5"):
    known_instance_ids = set()
    deck_order = []

    if version == "1.5":
        start_index = 1
    else:
        start_index = 30

    for battle_state in match_transitions[player]["states"][start_index:]:
        state = State.from_native_input(battle_state)

        for card in state.current_player.hand:
            if card.instance_id not in known_instance_ids:
                known_instance_ids.add(card.instance_id)
                deck_order.append((card.id, card))

    return deck_order


def find_deck_orders(match_transitions, version="1.5"):
    return _find_deck_order(match_transitions, "player1", version), _find_deck_order(
        match_transitions, "player2", version
    )


def find_draft_options(match_transitions):
    draft = []

    for draft_state in match_transitions["player1"]["states"][:30]:
        state = State.from_native_input(draft_state)

        draft.append(state.current_player.hand)

    return draft


def find_constructed_options(match_transitions):
    state = State.from_native_input(match_transitions["player1"]["states"][0])

    return list(state.current_player.hand)


def recreate_initial_state(
    match_transitions,
    deck_building_options,
    deck_orders,
    phase=Phase.DECK_BUILDING,
    version="1.5",
):
    if phase == Phase.DECK_BUILDING:
        start_index = 0
    else:
        if version == "1.5":
            start_index = 1
        else:
            start_index = 30

    state = State.from_native_input(
        match_transitions["player1"]["states"][start_index],
        deck_orders,
    )

    if phase == Phase.DECK_BUILDING:
        if version == "1.5":
            state._phase._constructed_cards = deck_building_options
        else:
            state._phase._draft_cards = deck_building_options
    elif phase == Phase.BATTLE:
        state.opposing_player.base_mana -= 1
        state.opposing_player.bonus_mana += 1

    return state


def validate_match(
    match_transitions,
    state,
    p1_agent,
    p2_agent,
    phase=Phase.DECK_BUILDING,
    version="1.5",
):
    if phase == Phase.DECK_BUILDING:
        start_index = 0
    else:
        if version == "1.5":
            start_index = 1
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

        assert original_state == recreated_state, f"{original_state}\n{recreated_state}"

        print("States match!")

        original_actions: List[str] = list(
            map(str.strip, next(actions_iter[player]).split(";"))
        )
        recreated_actions = []

        # workaround: ByteRL always outputs a PASS action at the end
        if len(original_actions) > 1 and original_actions[-1] == "PASS":
            original_actions.pop()

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


def check_version_12(agent):
    dataset_path = "gym_locm/engine/resources/consistency-dataset-v1.2.txt"

    with open(dataset_path, "r") as dataset:
        matches = dataset.readlines()

    print(len(matches))

    p1_agent = NativeAgent(
        agent,
        verbose=False,
    )
    p2_agent = NativeAgent(
        agent,
        verbose=False,
    )

    for i, match in enumerate(matches):
        p1_agent.reset()
        p2_agent.reset()

        print(f"Match #{i}")

        match_json = extract_match_json(match)

        print(match_json)

        transitions = extract_match_transitions(match_json, version="1.2")

        p1 = transitions["player1"]["states"]
        p2 = transitions["player2"]["states"]
        p1a = transitions["player1"]["actions"]
        p2a = transitions["player2"]["actions"]

        print(len(p1), p1[30:])
        print(len(p2), p2[30:])
        print(len(p1a), p1a[30:])
        print(len(p2a), p2a[30:])

        deck_orders = find_deck_orders(transitions, version="1.2")
        print(deck_orders)

        draft_options = find_draft_options(transitions)

        state1 = recreate_initial_state(
            transitions, draft_options, deck_orders, Phase.DRAFT, version="1.2"
        )

        validate_match(
            transitions, state1, p1_agent, p2_agent, Phase.DRAFT, version="1.2"
        )

        state2 = recreate_initial_state(
            transitions, draft_options, deck_orders, Phase.BATTLE, version="1.2"
        )

        state2._phase.instance_counter = state1._phase.instance_counter

        validate_match(
            transitions, state2, p1_agent, p2_agent, Phase.BATTLE, version="1.2"
        )

    p1_agent.close()
    p2_agent.close()


def check_version_15(agent):
    dataset_path = "gym_locm/engine/resources/consistency-dataset-v1.5.txt"

    with open(dataset_path, "r") as dataset:
        matches = dataset.readlines()

    print(len(matches))

    p1_agent = NativeAgent(
        agent,
        verbose=False,
    )
    p2_agent = NativeAgent(
        agent,
        verbose=False,
    )

    for i, match in enumerate(matches):
        p1_agent.reset()
        p2_agent.reset()

        print(f"Match #{i}")

        match_json = extract_match_json(match)

        print(match_json)

        transitions = extract_match_transitions(match_json, version="1.5")

        p1 = transitions["player1"]["states"]
        p2 = transitions["player2"]["states"]
        p1a = transitions["player1"]["actions"]
        p2a = transitions["player2"]["actions"]

        print(len(p1), p1[1:])
        print(len(p2), p2[1:])
        print(len(p1a), p1a[1:])
        print(len(p2a), p2a[1:])

        deck_orders = find_deck_orders(transitions, version="1.5")
        print(deck_orders)

        constructed_options = find_constructed_options(transitions)

        state1 = recreate_initial_state(
            transitions,
            constructed_options,
            deck_orders,
            Phase.CONSTRUCTED,
            version="1.5",
        )

        validate_match(
            transitions, state1, p1_agent, p2_agent, Phase.CONSTRUCTED, version="1.5"
        )

        state2 = recreate_initial_state(
            transitions, constructed_options, deck_orders, Phase.BATTLE, version="1.5"
        )

        state2._phase.instance_counter = state1._phase.instance_counter

        validate_match(
            transitions, state2, p1_agent, p2_agent, Phase.BATTLE, version="1.5"
        )

    print("All matches validated successfully! gym-locm is consistent with the original engine.")

    p1_agent.close()
    p2_agent.close()


def run():
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if args.version == "1.5":
        check_version_15(args.agent)
    elif args.version == "1.2":
        check_version_12(args.agent)
    else:
        raise Exception("Invalid version:", args.version)


if __name__ == "__main__":
    run()
