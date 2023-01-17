import argparse
import json
import os
import pathlib

import numpy as np
import pexpect
from scipy.special import softmax

base_path = str(pathlib.Path(__file__).parent.absolute())


def get_arg_parser():
    p = argparse.ArgumentParser(
        description="This is a predictor for trained RL drafts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--draft", help="path to draft model", default="draft.json")
    p.add_argument(
        "--draft-1", help="path to first draft model", default="1st-draft.json"
    )
    p.add_argument(
        "--draft-2", help="path to second draft model", default="2nd-draft.json"
    )
    p.add_argument(
        "--battle", help="command line to execute the battle agent", default="./battle"
    )

    return p


def read_game_input():
    # read players info
    game_input = [input(), input()]

    # read cards in hand and actions from opponent
    opp_hand, opp_actions = [int(i) for i in input().split()]
    game_input.append(f"{opp_hand} {opp_actions}")

    # read all opponent actions
    for i in range(opp_actions):
        game_input.append(input())  # opp action #i

    # read card count
    card_count = int(input())
    game_input.append(str(card_count))

    # read cards
    for i in range(card_count):
        game_input.append(input())  # card #i

    return game_input


def encode_state(game_input):
    # initialize empty state
    state = np.zeros((3, 16), dtype=np.float32)

    # get how many opponent action lines to skip
    opp_actions = int(game_input[2].split()[1])

    # put choices from player hand into the state
    for i, card in enumerate(game_input[4 + opp_actions :]):
        card = card.split()

        card_type = [1.0 if int(card[3]) == i else 0.0 for i in range(4)]
        cost = int(card[4]) / 12
        attack = int(card[5]) / 12
        defense = max(-12, int(card[6])) / 12
        keywords = list(map(int, map(card[7].__contains__, "BCDGLW")))
        player_hp = int(card[8]) / 12
        enemy_hp = int(card[9]) / 12
        card_draw = int(card[10]) / 2

        state[i] = (
            card_type
            + [cost, attack, defense, player_hp, enemy_hp, card_draw]
            + keywords
        )

    return state.flatten()


def act(network, state, past_choices):
    i = 0

    use_history = network[list(network.keys())[0]].shape[0] == 33 * 16

    if use_history:
        state = np.concatenate([past_choices, state])

    # do a forward pass through all fully connected layers
    while f"model/shared_fc{i}/w:0" in network:
        weights = network[f"model/shared_fc{i}/w:0"]
        biases = network[f"model/shared_fc{i}/b:0"]

        state = np.dot(state, weights) + biases
        state = np.tanh(state)

        i += 1

    # calculate the policy
    pi = np.dot(state, network["model/pi/w:0"]) + network["model/pi/b:0"]
    pi = softmax(pi)

    # extract the deterministic action
    action = np.argmax(pi)

    return action


def is_valid_action(action):
    return (
        action.startswith("PASS")
        or action.startswith("PICK")
        or action.startswith("SUMMON")
        or action.startswith("USE")
        or action.startswith("ATTACK")
    )


def load_model(path: str):
    # read the parameters
    with open(base_path + "/" + path, "r") as json_file:
        params = json.load(json_file)

    # initialize the network dict
    network = {}

    # load activation function for hidden layers
    if "version" not in params or params["version"] < 2:
        network["act_fun"] = np.tanh
    else:
        network["act_fun"] = dict(
            tanh=np.tanh,
            relu=lambda x: np.maximum(x, 0),
            elu=lambda x: np.where(x > 0, x, np.exp(x) - 1),
        )[params["act_fun"]]

    del params["version"]
    del params["act_fun"]

    # load weights as numpy arrays
    for label, weights in params.items():
        network[label] = np.array(weights)

    return network


def predict(paths: list, battle_cmd: str):
    network = None

    # spawn the battle agent
    battle_agent = pexpect.spawn(battle_cmd, echo=False, encoding="utf-8")

    # count the draft turns
    turn = 0

    # initialize past choices
    past_choices = np.zeros((30, 16))

    while True:
        game_input = read_game_input()

        # write game input to the agent regardless of the phase
        battle_agent.write("\n".join(game_input) + "\n")

        action = ""

        # find action line between all of the agent output
        while not is_valid_action(action):
            action = battle_agent.readline()

        # if mana is zero then it is draft phase
        is_draft_phase = int(game_input[0].split()[1]) == 0

        if network is None:
            playing_first = game_input[0].split()[2] == game_input[1].split()[2]

            path = paths[0] if playing_first or len(paths) == 1 else paths[1]

            network = load_model(path)

        if is_draft_phase:
            state = encode_state(game_input)
            action = act(network, state, past_choices)

            # update past choices with current pick
            past_choices[turn] = state[action * 16 : (action + 1) * 16]

            turn += 1

            print("PICK", action)
        else:
            # print action from battle agent
            print(action.strip())


def run():
    # get arguments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    # use json as draft agent
    if os.path.isfile(args.draft):
        paths = [args.draft]
    else:
        paths = [args.draft_1, args.draft_2]

    predict(paths, args.battle)


if __name__ == "__main__":
    run()
