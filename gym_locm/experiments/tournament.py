import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from statistics import mean
from typing import Tuple, List

import numpy as np
import pandas as pd

# suppress tensorflow deprecated warnings
from gym_locm.engine import PlayerOrder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

import tensorflow as tf

tf.get_logger().setLevel("INFO")
tf.get_logger().setLevel(logging.ERROR)

# continue importing
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from gym_locm import agents
from gym_locm.envs import LOCMDraftEnv
from gym_locm.util import encode_state_draft


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser.
    :return: a ready-to-use argument parser object
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(
        "--drafters",
        "-d",
        nargs="+",
        required=True,
        help="draft agents in the tournament (at least one, separated by space)",
    )
    p.add_argument(
        "--battler",
        "-b",
        default="random",
        choices=agents.battle_agents.keys(),
        help="battle agent to use (just one)",
    )
    p.add_argument(
        "--games",
        "-g",
        type=int,
        default=100,
        help="amount of games to run in every match-up",
    )
    p.add_argument(
        "--seeds",
        "-s",
        type=int,
        nargs="+",
        default=[1],
        help="seeds to use (at least one - match-ups will be "
        "repeated with each seed",
    )
    p.add_argument(
        "--concurrency", "-c", type=int, default=1, help="amount of concurrent games"
    )
    p.add_argument("--path", "-p", "-o", default=".", help="path to save result files")

    # todo: implement time limit for search-based battlers
    # p.add_argument('--time', '-t', default=200,
    #                help='max thinking time for search-based battlers')

    return p


def run_matchup(
    drafter1: str, drafter2: str, battler: str, games: int, seed: int, concurrency: int
) -> Tuple[
    Tuple[float, float],
    Tuple[list, list],
    Tuple[list, list],
    List[List[Tuple]],
    Tuple[list, list],
    List[float],
]:
    """
    Run the match-up between `drafter1` and `drafter2` using `battler` battler
    :param drafter1: drafter to play as first player
    :param drafter2: drafter to play as second player
    :param battler: battler to simulate the matches
    :param games: amount of matches to simulate
    :param seed: seed used to generate the matches
    :param concurrency: amount of matches executed at the same time
    :return: a tuple containing (i) a tuple containing the win rate of the
    first and second players, (ii) a tuple containing the average mana curves
    of the first and second players, (iii) a tuple containing the
    `30 * games` individual draft choices of the first and second players;
    (iv) a tuple of 3-uples containing the card alternatives presented to the
    players at each of the `games` episodes; and (v) a tuple containing the
    `games` decks built by the first and second players.
    """
    # parse the battle agent
    battler = agents.parse_battle_agent(battler)

    # initialize envs
    env = [
        lambda: LOCMDraftEnv(battle_agents=(battler(), battler()))
        for _ in range(concurrency)
    ]

    # wrap envs in a vectorized env
    env = DummyVecEnv(env)

    for i in range(concurrency):
        # no overlap between episodes at each process
        current_seed = seed + (games // concurrency) * i
        current_seed -= 1  # resetting the env increases the seed by 1

        # set seed to env
        env.env_method("seed", current_seed, indices=[i])

    # reset the env
    env.reset()

    # initialize first player
    if drafter1.endswith("zip"):
        current_drafter = agents.RLDraftAgent(PPO2.load(drafter1))
        current_drafter.use_history = "history" in drafter1
    else:
        current_drafter = agents.parse_draft_agent(drafter1)()

    current_drafter.seed(seed)
    current_drafter.name = drafter1
    drafter1 = current_drafter

    # initialize second player
    if drafter2.endswith("zip"):
        other_drafter = agents.RLDraftAgent(PPO2.load(drafter2))
        other_drafter.use_history = "history" in drafter2
    else:
        other_drafter = agents.parse_draft_agent(drafter2)()

    other_drafter.seed(seed)
    other_drafter.name = drafter2
    drafter2 = other_drafter

    # initialize metrics
    episodes_so_far = 0
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    drafter1.mana_curve = [0 for _ in range(13)]
    drafter2.mana_curve = [0 for _ in range(13)]
    drafter1.choices = [[] for _ in range(env.num_envs)]
    drafter2.choices = [[] for _ in range(env.num_envs)]
    drafter1.decks = [[[]] for _ in range(env.num_envs)]
    drafter2.decks = [[[]] for _ in range(env.num_envs)]
    alternatives = [[] for _ in range(env.num_envs)]

    # run the episodes
    while True:
        observations = env.get_attr("state")

        # get the current agent's action for all concurrent envs
        if isinstance(current_drafter, agents.RLDraftAgent):
            all_past_choices = env.get_attr("choices")
            new_observations = []

            for i, observation in enumerate(observations):
                new_observation = encode_state_draft(
                    observation,
                    use_history=current_drafter.use_history,
                    past_choices=all_past_choices[i][observation.current_player.id],
                )

                new_observations.append(new_observation)

            actions = current_drafter.act(new_observations)
        else:
            actions = [current_drafter.act(observation) for observation in observations]

        # log chosen cards into current agent's mana curve
        for i, (action, observation) in enumerate(zip(actions, observations)):
            # get chosen index
            try:
                chosen_index = action.origin
            except AttributeError:
                chosen_index = action

            # save choice
            current_drafter.choices[i].append(chosen_index)

            # get chosen card
            chosen_card = observation.current_player.hand[chosen_index]

            # increase amount of cards chosen with the chosen card's cost
            current_drafter.mana_curve[chosen_card.cost] += 1

            # add chosen card to this episode's deck
            current_drafter.decks[i][-1].append(chosen_card.id)

            # save card alternatives
            if observation.current_player.id == PlayerOrder.FIRST:
                alternatives[i].append(
                    tuple(map(lambda c: c.id, observation.current_player.hand))
                )

        # perform the action and get the outcome
        _, rewards, terminateds, truncateds, _ = env.step(actions)

        if isinstance(current_drafter, agents.RLDraftAgent):
            current_drafter.dones = np.logical_or(terminateds, truncateds)

        # update metrics
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]

            if terminateds[i] or truncateds[i]:
                episode_rewards[i].append(0.0)
                current_drafter.decks[i].append([])
                other_drafter.decks[i].append([])

                episodes_so_far += 1

        # check exiting condition
        if episodes_so_far >= games:
            break

        # swap drafters
        current_drafter, other_drafter = other_drafter, current_drafter

    # normalize mana curves
    total_choices = sum(drafter1.mana_curve)
    drafter1.mana_curve = [freq / total_choices for freq in drafter1.mana_curve]
    drafter2.mana_curve = [freq / total_choices for freq in drafter2.mana_curve]

    # join all parallel rewards
    all_rewards = [reward for rewards in episode_rewards for reward in rewards[:-1]]

    # join all parallel choices
    drafter1.choices = [c for choices in drafter1.choices for c in choices]
    drafter2.choices = [c for choices in drafter2.choices for c in choices]

    # join all parallel decks
    drafter1.decks = [deck for decks in drafter1.decks for deck in decks if deck]
    drafter2.decks = [deck for decks in drafter2.decks for deck in decks if deck]

    # join all parallel alternatives
    alternatives = [turn for env in alternatives for turn in env]

    # cap any unsolicited data from additional episodes
    all_rewards = all_rewards[:games]
    drafter1.choices = drafter1.choices[: 30 * games]
    drafter2.choices = drafter2.choices[: 30 * games]
    drafter1.decks = drafter1.decks[:games]
    drafter2.decks = drafter2.decks[:games]
    alternatives = alternatives[: 30 * games]

    # convert the list of rewards to the first player's win rate
    win_rate = (mean(all_rewards) + 1) * 50

    return (
        (win_rate, 100 - win_rate),
        (drafter1.mana_curve, drafter2.mana_curve),
        (drafter1.choices, drafter2.choices),
        alternatives,
        (drafter1.decks, drafter2.decks),
        all_rewards,
    )


def run():
    """
    Execute a tournament with the given arguments
    """
    # check python version
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    # read command line arguments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    # create output folder if it doesn't exist
    os.makedirs(args.path, exist_ok=True)

    # initialize data frames
    agg_results = pd.DataFrame(index=args.drafters, columns=args.drafters)
    ind_results = []

    drafter_role_index = pd.MultiIndex.from_product(
        [args.drafters, ["1st", "2nd"]], names=["drafter", "role"]
    )
    mana_curves = pd.DataFrame(index=drafter_role_index, columns=range(13))
    choices = pd.DataFrame(
        index=drafter_role_index, columns=range(30 * args.games * len(args.seeds))
    )

    alternatives_index = pd.MultiIndex.from_product(
        [args.seeds, range(1, args.games + 1), range(1, 31)],
        names=["seed", "episode", "turn"],
    )
    alternatives = pd.DataFrame(
        index=alternatives_index, columns=["card 1", "card 2", "card 3"]
    )

    episodes_index = pd.MultiIndex.from_product(
        [args.seeds, range(1, args.games + 1), args.drafters, args.drafters],
        names=["seed", "episode", "1st_player", "2nd_player"],
    )
    episodes = pd.DataFrame(
        index=episodes_index,
        columns=["timestamp", "reward"] + list(range(30)) + list(range(30)),
    )

    # for each combination of two drafters
    for drafter1 in args.drafters:
        for drafter2 in args.drafters:
            mean_win_rate = 0
            mean_mana_curves_1p, mean_mana_curves_2p = [], []
            choices_1p, choices_2p = [], []

            # for each seed
            for i, seed in enumerate(args.seeds):
                # if any drafter is a path to a folder, then select the
                # appropriate model inside the folder
                d1 = (
                    drafter1 + f"1st/{i + 1}.zip"
                    if drafter1.endswith("/")
                    else drafter1
                )
                d2 = (
                    drafter2 + f"2nd/{i + 1}.zip"
                    if drafter2.endswith("/")
                    else drafter2
                )

                # run the match-up and get the statistics
                wrs, mcs, chs, alts, dks, rwds = run_matchup(
                    d1, d2, args.battler, args.games, seed, args.concurrency
                )

                mean_win_rate += wrs[0]
                mean_mana_curves_1p.append(mcs[0])
                mean_mana_curves_2p.append(mcs[1])
                choices_1p.extend(chs[0])
                choices_2p.extend(chs[1])

                # save the card alternatives
                alternatives.loc[seed, :, :] = alts

                # save the episodes info
                episodes.loc[seed, :, drafter1, drafter2] = [
                    [datetime.now(), rwds[i]] + dks[0][i] + dks[1][i]
                    for i in range(len(rwds))
                ]

                # save individual result
                ind_results.append([drafter1, drafter2, seed, wrs[0], datetime.now()])

            # get the mean win rate of the first player
            mean_win_rate /= len(args.seeds)

            # round the mean win rate up to three decimal places
            mean_win_rate = round(mean_win_rate, 3)

            # get the current time
            current_time = datetime.now()

            # print the match-up and its result
            print(current_time, drafter1, drafter2, mean_win_rate)

            # save aggregate result
            agg_results.loc[drafter1][drafter2] = mean_win_rate

            # save mana curves and choices if they have not been saved yet
            if np.isnan(mana_curves.loc[drafter1, "1st"][0]):
                # get the mean mana curve for the drafter
                mean_mana_curves_1p = np.array(mean_mana_curves_1p).mean(axis=0)

                # change unit from percentage to amount of cards
                mean_mana_curves_1p *= 30

                # update appropriate mana curves data frame row
                mana_curves.loc[drafter1, "1st"] = mean_mana_curves_1p

                # update appropriate choices data frame row
                choices.loc[drafter1, "1st"] = choices_1p

            if np.isnan(mana_curves.loc[drafter2, "2nd"][0]):
                # get the mean mana curve for the drafter
                mean_mana_curves_2p = np.array(mean_mana_curves_2p).mean(axis=0)

                # change unit from percentage to amount of cards
                mean_mana_curves_2p *= 30

                # update appropriate mana curves data frame row
                mana_curves.loc[drafter2, "2nd"] = mean_mana_curves_2p

                # update appropriate choices data frame row
                choices.loc[drafter2, "2nd"] = choices_2p

    # add average win rate to aggregate results
    avg_wr_as_1st_player = agg_results.mean(axis=1)
    avg_wr_as_2nd_player = 100 - agg_results.mean(axis=0)
    agg_results["average"] = (avg_wr_as_1st_player + avg_wr_as_2nd_player) / 2

    # transform individual results matrix into a data frame
    ind_results = np.array(ind_results)
    ind_results_index = pd.MultiIndex.from_product(
        [args.drafters, args.drafters, args.seeds],
        names=["drafter1", "drafter2", "seed"],
    )
    ind_results = pd.DataFrame(
        data=ind_results[:, 3:],
        index=ind_results_index,
        columns=["win_rate", "datetime"],
    )

    # save all tournament data to csv files
    agg_results.to_csv(args.path + "/aggregate_win_rates.csv", index_label="1p \\ 2p")
    ind_results.to_csv(args.path + "/individual_win_rates.csv")
    mana_curves.to_csv(args.path + "/mana_curves.csv")
    episodes.to_csv(args.path + "/episodes.csv")
    alternatives.to_csv(args.path + "/alternatives.csv")
    choices.T.to_csv(args.path + "/choices.csv")

    # and also pickle files for easy reading
    agg_results.to_pickle(args.path + "/aggregate_win_rates.pkl")
    ind_results.to_pickle(args.path + "/individual_win_rates.pkl")
    mana_curves.to_pickle(args.path + "/mana_curves.pkl")
    alternatives.to_pickle(args.path + "/alternatives.pkl")
    choices.to_pickle(args.path + "/choices.pkl")
    episodes.to_pickle(args.path + "/episodes.pkl")


if __name__ == "__main__":
    run()
