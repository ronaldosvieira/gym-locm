import argparse
import logging
import os
import sys
import warnings
import pandas as pd
from datetime import datetime
from statistics import mean

# suppress tensorflow deprecated warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

tf.get_logger().setLevel('INFO')
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
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--drafters', '-d', nargs='+', required=True,
                   help='draft agents in the tournament '
                        '(at least one, separated by space)')
    p.add_argument('--battler', '-b', default='random',
                   choices=agents.battle_agents.keys(),
                   help='battle agent to use (just one)')
    p.add_argument("--games", '-g', type=int, default=100,
                   help='amount of games to run in every match-up')
    p.add_argument('--seeds', '-s', type=int, nargs='+', default=[1],
                   help='seeds to use (at least one - match-ups will be '
                        'repeated with each seed')
    p.add_argument('--concurrency', '-c', type=int, default=1,
                   help='amount of concurrent games')

    # todo: implement time limit for search-based battlers
    # p.add_argument('--time', '-t', default=200,
    #                help='max thinking time for search-based battlers')

    return p


def run_matchup(drafter1: str, drafter2: str, battler: str, games: int,
                seed: int, concurrency: int) -> float:
    """
    Run the match-up between `drafter1` and `drafter2` using `battler` battler
    :param drafter1: drafter to play as first player
    :param drafter2: drafter to play as second player
    :param battler: battler to simulate the matches
    :param games: amount of matches to simulate
    :param seed: seed used to generate the matches
    :param concurrency: amount of matches executed at the same time
    :return: the win rate of the first player
    """
    # parse the battle agent
    battler = agents.parse_battle_agent(battler)

    # initialize envs
    env = []
    for i in range(concurrency):
        # no overlap between episodes at each process
        current_seed = seed + (games // concurrency) * i
        current_seed -= 1  # resetting the env increases the seed by 1

        # create the env
        env.append(lambda: LOCMDraftEnv(seed=current_seed,
                                        battle_agents=(battler(), battler())))

    # wrap envs in a vectorized env
    env = DummyVecEnv(env)

    # initialize first player
    if drafter1.endswith('zip'):
        current_drafter = agents.RLDraftAgent(PPO2.load(drafter1))
        current_drafter.use_history = "history" in drafter1
    else:
        current_drafter = agents.parse_draft_agent(drafter1)()

    # initialize second player
    if drafter2.endswith('zip'):
        other_drafter = agents.RLDraftAgent(PPO2.load(drafter2))
        other_drafter.use_history = "history" in drafter2
    else:
        other_drafter = agents.parse_draft_agent(drafter2)()

    # reset the env
    env.reset()

    # initialize metrics
    episodes_so_far = 0
    episode_rewards = [[0.0] for _ in range(env.num_envs)]

    # run the episodes
    while True:
        observations = env.get_attr('state')

        # get the current agent's action for all concurrent envs
        if isinstance(current_drafter, agents.RLDraftAgent):
            all_past_choices = env.get_attr('choices')
            new_observations = []

            for i, observation in enumerate(observations):
                new_observation = encode_state_draft(
                    observation,
                    use_history=current_drafter.use_history,
                    past_choices=all_past_choices[i][observation.current_player.id]
                )

                new_observations.append(new_observation)

            actions = current_drafter.act(new_observations)
        else:
            actions = [current_drafter.act(observation)
                       for observation in observations]

        # perform the action and get the outcome
        _, rewards, dones, _ = env.step(actions)

        if isinstance(current_drafter, agents.RLDraftAgent):
            current_drafter.dones = dones

        # update metrics
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]

            if dones[i]:
                episode_rewards[i].append(0.0)

                episodes_so_far += 1

        # check exiting condition
        if episodes_so_far >= games:
            break

        # swap drafters
        current_drafter, other_drafter = other_drafter, current_drafter

    # join all parallel rewards
    all_rewards = [reward for rewards in episode_rewards
                   for reward in rewards[:-1]]

    # cap any unsolicited additional episodes
    all_rewards = all_rewards[:games]

    # convert the list of rewards to the first player's win rate
    win_rate = (mean(all_rewards) + 1) * 50

    return win_rate


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

    # initialize data frames
    agg_results = pd.DataFrame(index=args.drafters, columns=args.drafters)

    # for each combination of two drafters
    for drafter1 in args.drafters:
        for drafter2 in args.drafters:
            win_rates = []

            # for each seed
            for i, seed in enumerate(args.seeds):
                # if any drafter is a path to a folder, then select the
                # appropriate model inside the folder
                d1 = drafter1 + f'1st/{i + 1}.zip' if drafter1.endswith('/') else drafter1
                d2 = drafter2 + f'2nd/{i + 1}.zip' if drafter2.endswith('/') else drafter2

                # run the match-up and get the win rate of the first player
                win_rates.append(run_matchup(d1, d2, args.battler, args.games,
                                             seed, args.concurrency))

            # get the mean win rate of the first player
            mean_win_rate = mean(win_rates)

            # save mean win rate
            agg_results[drafter1][drafter2] = mean_win_rate

            # round the mean win rate up to three decimal places
            mean_win_rate = round(mean_win_rate, 3)

            # get the current time
            current_time = datetime.now()

            # print the match-up and its result
            print(current_time, drafter1, drafter2, mean_win_rate)

    # save tournament data to csv file
    agg_results.to_csv('tournament.csv', index_label="1p \\ 2p")


if __name__ == '__main__':
    run()
