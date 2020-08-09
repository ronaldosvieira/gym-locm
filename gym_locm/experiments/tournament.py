import logging
import os
import warnings
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

draft_choices = {
    "pass": agents.PassDraftAgent,
    "random": agents.RandomDraftAgent,
    "rule-based": agents.RuleBasedDraftAgent,
    "max-attack": agents.MaxAttackDraftAgent,
    "icebox": agents.IceboxDraftAgent,
    "closet-ai": agents.ClosetAIDraftAgent,
    "uji1": agents.UJI1DraftAgent,
    "uji2": agents.UJI2DraftAgent,
    "coac": agents.CoacDraftAgent
}

battle_choices = {
    "max-attack": agents.MaxAttackBattleAgent,
    "greedy": agents.GreedyBattleAgent
}

drafter1 = 'models/best/max-attack/lstm/1st/1.zip'
drafter2 = 'random'

battler = 'max-attack'

seed = 19279988
eval_episodes = 1000
num_envs = 4

if __name__ == '__main__':
    env = []

    history = 'history' in drafter1 or 'history' in drafter2

    battler = battle_choices[battler]

    # initialize envs
    for i in range(num_envs):
        # no overlap between episodes at each process
        current_seed = seed + (eval_episodes // num_envs) * i
        current_seed -= 1  # resetting the env increases the seed by 1

        # create the env
        env.append(lambda: LOCMDraftEnv(seed=current_seed,
                                        battle_agents=(battler(), battler()),
                                        use_draft_history=history))

    # wrap envs in a vectorized env
    env = DummyVecEnv(env)

    # initialize first player
    if drafter1.endswith('zip'):
        drafter1 = agents.RLDraftAgent(PPO2.load(drafter1, env=env))
    else:
        drafter1 = draft_choices[drafter1]()

    # initialize second player
    if drafter2.endswith('zip'):
        drafter2 = agents.RLDraftAgent(PPO2.load(drafter2, env=env))
    else:
        drafter2 = draft_choices[drafter2]()

    # reset the env
    observations = env.reset()

    # initialize metrics
    episodes_so_far = 0
    episode_rewards = [[0.0] for _ in range(env.num_envs)]

    current_drafter, other_drafter = drafter1, drafter2

    # run the episodes
    while True:
        # get the current agent's action for all parallel envs
        if isinstance(current_drafter, agents.RLDraftAgent):
            actions = current_drafter.act(observations)
        else:
            observations = env.get_attr('state')
            actions = [current_drafter.act(observation)
                       for observation in observations]

        # perform the action and get the outcome
        observations, rewards, dones, _ = env.step(actions)

        # update metrics
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]

            if dones[i]:
                episode_rewards[i].append(0.0)

                episodes_so_far += 1

        # check exiting condition
        if episodes_so_far >= eval_episodes:
            break

        # swap drafters
        current_drafter, other_drafter = other_drafter, current_drafter

    # join all parallel rewards
    all_rewards = [reward for rewards in episode_rewards
                   for reward in rewards[:-1]]

    # cap any unsolicited additional episodes
    all_rewards = all_rewards[:eval_episodes]

    win_rate = (mean(all_rewards) + 1) * 50

    print(win_rate)
