import argparse
import json

import numpy as np
from scipy.special import softmax


def get_arg_parser():
    p = argparse.ArgumentParser(
        description="This is a predictor for trained RL drafts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("path", help="path to model file")
    p.add_argument("--convert", action="store_true",
                   help="convert mode - turn a given zip model "
                        "into usable pkl model.")

    return p


def convert(path: str):
    from stable_baselines import PPO2

    model = PPO2.load(path)

    new_path = path.rstrip(r"\.zip") + ".json"

    with open(new_path, 'w') as json_file:
        params = {}

        for label, weights in model.get_parameters().items():
            params[label] = weights.tolist()

        json.dump(params, json_file)

        print("Converted model written to", new_path)


def read_game_input():
    game_input = [input(), input()]

    opp_hand, opp_actions = [int(i) for i in input().split()]
    game_input.append(f"{opp_hand} {opp_actions}")

    for i in range(opp_actions):
        game_input.append(input())  # opp action #i

    card_count = int(input())
    game_input.append(str(card_count))

    for i in range(card_count):
        game_input.append(input())  # card #i

    return game_input


def encode_state(game_input):
    state = np.zeros((3, 16), dtype=np.float32)

    opp_actions = int(game_input[2].split()[1])

    for i, card in enumerate(game_input[4 + opp_actions:]):
        card = card.split()

        card_type = [1.0 if int(card[3]) == i else 0.0 for i in range(4)]
        cost = int(card[4]) / 12
        attack = int(card[5]) / 12
        defense = max(-12, int(card[6])) / 12
        keywords = list(map(int, map(card[7].__contains__, 'BCDGLW')))
        player_hp = int(card[8]) / 12
        enemy_hp = int(card[9]) / 12
        card_draw = int(card[10]) / 2

        state[i] = card_type + [cost, attack, defense, player_hp,
                                enemy_hp, card_draw] + keywords

    return state.flatten()


def act(network, state):
    i = 0

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


def predict(path: str):
    with open(path, 'r') as json_file:
        params = json.load(json_file)

    network = dict((label, np.array(weights)) for label, weights in params.items())

    while True:
        game_input = read_game_input()

        if int(game_input[0].split()[1]) == 0:
            state = encode_state(game_input)
            action = act(network, state)

            print("PICK", action)
        else:
            print("PASS")


def run():
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if args.convert:
        convert(args.path)
    else:
        predict(args.path)


if __name__ == '__main__':
    run()
