import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import warnings
from datetime import datetime
from functools import partial
from random import choice

from gym.wrappers import TimeLimit

from gym_locm.envs.battle import LOCMBattleSelfPlayEnv, LOCMBattleSingleEnv, LOCMBattleEnv

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll import scope
from stable_baselines import PPO2, DQN
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from statistics import mean, stdev

from gym_locm.engine import PlayerOrder, Phase
from gym_locm.agents import MaxAttackBattleAgent, MaxAttackDraftAgent, IceboxDraftAgent, RandomBattleAgent
from gym_locm.envs.draft import LOCMDraftSelfPlayEnv, LOCMDraftSingleEnv, LOCMDraftEnv

# which phase to train
phase = Phase.BATTLE

# draft strategy ('basic', 'history', 'lstm' or 'curve')
draft_strat = 'basic'

# battle strategy ('punish', 'map', 'clip')
battle_strat = 'clip'

# training mode ('normal', 'interleaved-self-play', 'self-play')
training_mode = 'normal'

# algorithm ('dqn', 'ppo2')
algorithm = 'dqn'

# training parameters
seed = 96737
num_processes = 1
train_episodes = 30000
eval_episodes = 3000
num_evals = 10

# bayesian optimization parameters
num_trials = 50
num_warmup_trials = 10
optimize_for = PlayerOrder.FIRST

# where to save the model
path = 'models/hyp-search/clip-battle-dqn'

# hyperparameter space
if algorithm == 'ppo2':
    param_dict = {
        'n_switches': hp.choice('n_switches', [10, 100, 1000]),
        'layers': hp.uniformint('layers', 1, 3),
        'neurons': hp.uniformint('neurons', 24, 256),
        'n_steps': scope.int(hp.quniform('n_steps', 30, 300, 30)),
        'nminibatches': scope.int(hp.quniform('nminibatches', 1, 300, 1)),
        'noptepochs': scope.int(hp.quniform('noptepochs', 3, 20, 1)),
        'cliprange': hp.quniform('cliprange', 0.1, 0.3, 0.1),
        'vf_coef': hp.quniform('vf_coef', 0.5, 1.0, 0.5),
        'ent_coef': hp.uniform('ent_coef', 0, 0.01),
        'learning_rate': hp.loguniform('learning_rate',
                                       np.log(0.00005),
                                       np.log(0.01)),
    }

    if phase == phase.DRAFT and draft_strat == 'lstm':
        param_dict['layers'] = hp.uniformint('layers', 0, 2)
        param_dict['nminibatches'] = hp.choice('nminibatches', [1])

    if phase == Phase.BATTLE:
        param_dict['noptepochs'] = scope.int(hp.quniform('noptepochs', 3, 100, 1))
        param_dict['n_steps'] = scope.int(hp.quniform('n_steps', 30, 3000, 1))
        param_dict['nminibatches'] = scope.int(hp.quniform('nminibatches', 1, 3000, 1))
        param_dict['learning_rate'] = hp.loguniform('learning_rate',
                                   np.log(0.000005),
                                   np.log(0.01))
elif algorithm == 'dqn':
    assert draft_strat != 'lstm', 'DQN not currently supported for lstm-draft'

    param_dict = {
        'n_switches': hp.choice('n_switches', [10, 100, 1000]),
        'layers': hp.uniformint('layers', 1, 7),
        'neurons': hp.uniformint('neurons', 24, 512),
        'buffer_size': hp.choice('buffer_size', [5000, 25000, 50000]),
        'batch_size': hp.uniformint('batch_size', 4, 256),
        'learning_rate': hp.loguniform('learning_rate', np.log(.00005), np.log(.01)),
        'exploration_fraction': hp.quniform('exploration_fraction', .2, .6, .005),
        'prioritized_replay_alpha': hp.quniform('prioritized_replay_alpha',
                                                0., 1., .1),
        'prioritized_replay_beta0': hp.quniform('prioritized_replay_beta0',
                                                0.2, 0.8, .2),
        'target_network_update_freq': hp.choice('target_network_update_freq',
                                                [5000, 25000, 50000])
    }

    num_processes = 1
else:
    raise ValueError("Invalid algorithm. Should be 'ppo2' or 'dqn'.")

# initializations
counter = 0
make_draft_agents = lambda: (IceboxDraftAgent(), IceboxDraftAgent())
make_battle_agents = lambda: (MaxAttackBattleAgent(), MaxAttackBattleAgent())


def env_builder_draft(seed, play_first=True, **params):
    env = LOCMDraftSelfPlayEnv2(seed=seed, battle_agents=make_battle_agents(),
                                use_draft_history=draft_strat == 'history',
                                use_mana_curve=draft_strat == 'curve')
    env.play_first = play_first

    return lambda: env


def env_builder_battle(seed, play_first=True, **params):
    env = LOCMBattleSelfPlayEnv2(seed=seed, draft_agents=make_draft_agents(),
                                 return_action_mask=battle_strat == 'clip')
    env.play_first = play_first
    env = TimeLimit(env, max_episode_steps=200)

    return lambda: env


def eval_env_builder_draft(seed, play_first=True, **params):
    env = LOCMDraftSingleEnv(seed=seed, draft_agent=MaxAttackDraftAgent(),
                             battle_agents=make_battle_agents(),
                             use_draft_history=draft_strat == 'history',
                             use_mana_curve=draft_strat == 'curve')
    env.play_first = play_first

    return lambda: env


def eval_env_builder_battle(seed, play_first=True, **params):
    env = LOCMBattleSingleEnv2(seed=seed, battle_agent=MaxAttackBattleAgent(),
                               draft_agents=make_draft_agents(),
                               return_action_mask=battle_strat == 'clip')
    env.play_first = play_first
    env = TimeLimit(env, max_episode_steps=200)

    return lambda: env


def eval_env_builder_battle2(seed, play_first=True, **params):
    env = LOCMBattleSingleEnv2(seed=seed, battle_agent=RandomBattleAgent(),
                               draft_agents=make_draft_agents(),
                               return_action_mask=battle_strat == 'clip')
    env.play_first = play_first
    env = TimeLimit(env, max_episode_steps=200)

    return lambda: env


def model_builder_mlp(env, **params):
    net_arch = [params['neurons']] * params['layers']

    if algorithm == 'ppo2':
        return PPO2(MlpPolicy, env, verbose=0, gamma=1, seed=seed,
                    policy_kwargs=dict(net_arch=net_arch),
                    n_steps=params['n_steps'],
                    nminibatches=params['nminibatches'],
                    noptepochs=params['noptepochs'],
                    cliprange=params['cliprange'],
                    vf_coef=params['vf_coef'],
                    ent_coef=params['ent_coef'],
                    learning_rate=params['learning_rate'],
                    tensorboard_log=None)
    elif algorithm == 'dqn':
        return DQN(LnMlpPolicy, env, verbose=1, gamma=1, seed=seed,
                   policy_kwargs=dict(layers=net_arch),
                   double_q=True, prioritized_replay=True,
                   prioritized_replay_beta_iters=1000000,
                   learning_rate=params['learning_rate'],
                   buffer_size=params['buffer_size'],
                   batch_size=params['batch_size'],
                   exploration_fraction=params['exploration_fraction'],
                   prioritized_replay_alpha=params['prioritized_replay_alpha'],
                   prioritized_replay_beta0=params['prioritized_replay_beta0'],
                   target_network_update_freq=params['target_network_update_freq'])


def model_builder_lstm(env, **params):
    net_arch = ['lstm'] + [params['neurons']] * params['layers']

    return PPO2(MlpLstmPolicy, env, verbose=0, gamma=1, seed=seed,
                policy_kwargs=dict(net_arch=net_arch, n_lstm=params['neurons']),
                n_steps=params['n_steps'],
                nminibatches=params['nminibatches'],
                noptepochs=params['noptepochs'],
                cliprange=params['cliprange'],
                vf_coef=params['vf_coef'],
                ent_coef=params['ent_coef'],
                learning_rate=params['learning_rate'],
                tensorboard_log=None)


if phase == Phase.DRAFT:
    env_builder = env_builder_draft
    eval_env_builder = eval_env_builder_draft

    if draft_strat == 'lstm':
        model_builder = model_builder_lstm
    else:
        model_builder = model_builder_mlp
elif phase == Phase.BATTLE:
    env_builder = env_builder_battle
    eval_env_builder = eval_env_builder_battle
    model_builder = model_builder_mlp


def map_invalid_action(action_mask, action):
    if action < 0 or action > 144:
        return action  # out of bounds, action decoder will handle it

    try:
        return action + action_mask[action:].index(1)
    except ValueError:
        return 0


class LOCMBattleSelfPlayEnv2(LOCMBattleSelfPlayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.play_first = not self.play_first

        return super().reset()

    def create_model(self, **params):
        self.model = model_builder(DummyVecEnv([lambda: self]), **params)

    if battle_strat == 'punish':
        def step(self, action: int):
            state, reward, done, info = super().step(action)

            if info['invalid'] and not done:
                state, reward, done, _ = super().step(0)

                reward -= 0.025

            return state, reward, done, info
    elif battle_strat == 'map':
        def step(self, action: int):
            """Makes an action in the game."""
            player = self.state.current_player.id

            # do the action
            new_action = map_invalid_action(self.state.action_mask, action)
            state, reward, done, info = LOCMBattleEnv.step(self, new_action)

            # have opponent play until its player's turn or there's a winner
            while self.state.current_player.id != player and self.state.winner is None:
                state = self._encode_state()
                action = self.model.predict(state)[0]

                action = map_invalid_action(self.state.action_mask, action)
                state, reward, done, info2 = LOCMBattleEnv.step(self, action)

                if info2['invalid'] and not done:
                    state, reward, done, info2 = LOCMBattleEnv.step(self, 0)
                    break

            if not self.play_first:
                reward = -reward

            return state, reward, done, info
    elif battle_strat == 'clip':
        def step(self, action: int):
            state, reward, done, info = super().step(action)

            while sum(self.action_mask) == 1:
                state, reward2, done, info2 = super().step(0)

                reward += reward2
                info2['invalid'] = info['invalid']
                info = info2

            return state, reward, done, info


class LOCMBattleSingleEnv2(LOCMBattleSingleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.play_first = not self.play_first

        return super().reset()

    if battle_strat == 'punish':
        def step(self, action: int):
            state, reward, done, info = super().step(action)

            if info['invalid'] and not done:
                state, reward, done, _ = super().step(0)

                reward -= 0.025

            return state, reward, done, info
    elif battle_strat == 'map':
        def step(self, action: int):
            action_mask = self.state.action_mask

            return super().step(map_invalid_action(action_mask, action))
    elif battle_strat == 'clip':
        def step(self, action: int):
            state, reward, done, info = super().step(action)

            while sum(self.action_mask) == 1:
                state, reward2, done, info2 = super().step(0)

                reward += reward2
                info2['invalid'] = info['invalid']
                info = info2

            return state, reward, done, info


class LOCMDraftSelfPlayEnv2(LOCMDraftSelfPlayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rstate = None
        self.done = [False] + [True] * (num_processes - 1)

    def create_model(self, **params):
        if draft_strat == 'lstm':
            self.rstate = np.zeros(shape=(num_processes, params['neurons'] * 2))

        env = [env_builder(0, **params) for _ in range(num_processes)]
        env = DummyVecEnv(env)

        self.model = model_builder(env, **params)

    if draft_strat == 'lstm':
        def step(self, action):
            """Makes an action in the game."""
            obs = self._encode_state()

            zero_completed_obs = np.zeros((num_processes, *self.observation_space.shape))
            zero_completed_obs[0, :] = obs

            prediction, self.rstate = self.model.predict(zero_completed_obs,
                                                         state=self.rstate,
                                                         mask=self.done)

            # act according to first and second players
            if self.play_first:
                LOCMDraftEnv.step(self, action)
                state, reward, self.done[0], info = \
                    LOCMDraftEnv.step(self, prediction[0])
            else:
                LOCMDraftEnv.step(self, prediction[0])
                state, reward, self.done[0], info = \
                    LOCMDraftEnv.step(self, action)
                reward = -reward

            return state, reward, self.done[0], info


def normal_training(params):
    global counter

    counter += 1

    # get and print start time
    start_time = str(datetime.now())
    print('Start time:', start_time)

    if algorithm == 'ppo2':
        # ensure integer hyperparams
        params['n_steps'] = int(params['n_steps'])
        params['nminibatches'] = int(params['nminibatches'])
        params['noptepochs'] = int(params['noptepochs'])

        # ensure nminibatches <= n_steps
        params['nminibatches'] = min(params['nminibatches'],
                                     params['n_steps'])

        # ensure n_steps % nminibatches == 0
        while params['n_steps'] % params['nminibatches'] != 0:
            params['nminibatches'] -= 1

    # build the env
    env = []

    for i in range(num_processes):
        current_seed = seed + (train_episodes // num_processes) * i

        env.append(eval_env_builder(current_seed, True, **params))

    env = SubprocVecEnv(env, start_method='spawn')

    # build the evaluation env
    eval_seed = seed + train_episodes

    eval_env = []

    for i in range(num_processes):
        current_seed = eval_seed + (eval_episodes // num_processes) * i

        eval_env.append(eval_env_builder(current_seed, True, **params))

    eval_env = SubprocVecEnv(eval_env, start_method='spawn')

    # build the model
    model = model_builder(env, **params)

    # create the model name
    model_name = f'{counter}'

    # build the model path
    model_path = path + '/' + model_name

    # set tensorflow log dir
    model.tensorboard_log = model_path

    # create necessary folders
    os.makedirs(model_path, exist_ok=True)

    # save starting model
    model.save(model_path + '/0-episodes')

    results = []

    # calculate utilities
    eval_every_ep = train_episodes / num_evals
    model.last_eval, model.next_eval = 0, eval_every_ep

    # print hyperparameters
    print(f"{battle_strat}-battle" if phase == phase.BATTLE else f"{draft_strat}-draft")
    print(f"algorithm={algorithm}, training_mode={training_mode}")
    print(f"seed={seed}, num_processes={num_processes}, train_episodes={train_episodes}, "
          f"eval_episodes={eval_episodes}, num_evals={num_evals}")
    print(f"num_trials={num_trials}, num_warmup_trials={num_warmup_trials}, "
          f"optimize_for={optimize_for}")
    print(f"path={path}")
    print(params)

    def make_evaluate(eval_env):
        def evaluate(model):
            """
            Evaluates a model.
            :param model: (stable_baselines model) Model to be evaluated.
            :return: The mean (win rate) and standard deviation.
            """
            # initialize structures
            episode_rewards = [[0.0] for _ in range(eval_env.num_envs)]
            episode_lengths = []
            episode_wins = []

            # set seeds
            for i in range(num_processes):
                current_seed = eval_seed + (eval_episodes // num_processes) * i

                eval_env.env_method('seed', current_seed, indices=[i])

            # reset the env
            obs = eval_env.reset()
            episodes = 0

            model.set_env(eval_env)

            action_hist = [0] * eval_env.action_space.n

            while True:
                # get current turns
                turns = eval_env.get_attr('turn')

                # get a deterministic prediction from model
                actions, _ = model.predict(obs, deterministic=True)

                for action in actions:
                    action_hist[action] += 1

                # do the predicted action and save the outcome
                obs, rewards, dones, info = eval_env.step(actions)

                # save current reward into episode rewards
                for i in range(eval_env.num_envs):
                    episode_rewards[i][-1] += rewards[i]

                    if dones[i]:
                        episode_wins.append(1 if rewards[i] > 0 else 0)
                        episode_lengths.append(turns[i])
                        episode_rewards[i].append(0.0)

                        episodes += 1

                if any(dones):
                    if episodes >= eval_episodes:
                        for i in range(eval_env.num_envs):
                            if episode_rewards[i][-1] == 0:
                                episode_rewards[i].pop()

                        print("action histogram:", action_hist)
                        print(sum(action_hist) / episodes, "actions per episode")

                        break

            all_rewards = []

            # flatten episode rewards lists
            for part in episode_rewards:
                all_rewards.extend(part)

            model.set_env(env)

            # return the mean reward of all episodes
            return mean(all_rewards), mean(episode_wins), mean(episode_lengths)

        return evaluate

    def callback(_locals, _globals):
        episodes_so_far = sum(env.get_attr('episodes'))

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model.save(model_path + f'/{episodes_so_far}-episodes')

            # evaluate the models and get the metrics
            print(f"Evaluating... ({episodes_so_far})")
            mean_reward, win_rate, mean_length = make_evaluate(eval_env)(model)
            print(f"Done. {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
            print()

            results.append(mean_reward)

            model.last_eval = episodes_so_far
            model.next_eval += eval_every_ep

        return episodes_so_far < train_episodes

    # evaluate the initial model
    print("Evaluating... (0)")
    mean_reward, win_rate, mean_length = make_evaluate(eval_env)(model)
    print(f"Done. {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
    print()

    results.append(mean_reward)

    try:
        # train the model
        model.learn(total_timesteps=25 * train_episodes, callback=callback)
    except KeyboardInterrupt:
        print(f'Training stopped at {sum(env.get_attr("episodes"))}')

    # evaluate the final model
    print("Evaluating... (final)")
    mean_reward, win_rate, mean_length = make_evaluate(eval_env)(model)
    print(f"Done. {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
    print()

    results.append(mean_reward)

    # save the final model
    model.save(model_path + '/final')

    # close the envs
    for e in (env, eval_env):
        e.close()

    # get and print end time
    end_time = str(datetime.now())
    print('End time:', end_time)

    # save model info to results file
    with open(path + '/' + 'results.txt', 'a') as file:
        file.write(json.dumps(dict(id=counter, **params, results=results,
                                   start_time=start_time,
                                   end_time=end_time), indent=2))

    return {'loss': -max(results), 'status': STATUS_OK}


def interleaved_self_play(params):
    global counter

    counter += 1

    # get and print start time
    start_time = str(datetime.now())
    print('Start time:', start_time)

    if algorithm == 'ppo2':
        # ensure integer hyperparams
        params['n_steps'] = int(params['n_steps'])
        params['nminibatches'] = int(params['nminibatches'])
        params['noptepochs'] = int(params['noptepochs'])

        # ensure nminibatches <= n_steps
        params['nminibatches'] = min(params['nminibatches'],
                                     params['n_steps'])

        # ensure n_steps % nminibatches == 0
        while params['n_steps'] % params['nminibatches'] != 0:
            params['nminibatches'] -= 1

    # build the envs
    env1, env2 = [], []

    for i in range(num_processes):
        current_seed = seed + (train_episodes // num_processes) * i

        env1.append(env_builder(current_seed, True, **params))
        env2.append(env_builder(current_seed, False, **params))

    env1 = SubprocVecEnv(env1, start_method='spawn')
    env2 = SubprocVecEnv(env2, start_method='spawn')

    env1.env_method('create_model', **params)
    env2.env_method('create_model', **params)

    # build the evaluation envs
    eval_seed = seed + train_episodes

    eval_env1, eval_env2 = [], []

    for i in range(num_processes):
        current_seed = eval_seed + (eval_episodes // num_processes) * i

        eval_env1.append(eval_env_builder(current_seed, True, **params))
        eval_env2.append(eval_env_builder(current_seed, False, **params))

    eval_env1 = SubprocVecEnv(eval_env1, start_method='spawn')
    eval_env2 = SubprocVecEnv(eval_env2, start_method='spawn')

    # build the models
    model1 = model_builder(env1, **params)
    model2 = model_builder(env2, **params)

    if optimize_for == PlayerOrder.SECOND:
        model1, model2 = model2, model1
        env1, env2 = env2, env1
        eval_env1, eval_env2 = eval_env2, eval_env1

    # update parameters on surrogate models
    env1.env_method('update_parameters', model2.get_parameters())
    env2.env_method('update_parameters', model1.get_parameters())

    # create the model name
    model_name = f'{counter}'

    # build model paths
    model_path1 = path + '/' + model_name + '/1st'
    model_path2 = path + '/' + model_name + '/2nd'

    if optimize_for == PlayerOrder.SECOND:
        model_path1, model_path2 = model_path2, model_path1

    # set tensorflow log dir
    model1.tensorboard_log = model_path1
    model2.tensorboard_log = model_path2

    # create necessary folders
    os.makedirs(model_path1, exist_ok=True)
    os.makedirs(model_path2, exist_ok=True)

    # save starting models
    model1.save(model_path1 + '/0-episodes')
    model2.save(model_path2 + '/0-episodes')

    results = [[[], []], [[], []]]

    # calculate utilities
    eval_every_ep = train_episodes / num_evals
    switch_every_ep = train_episodes / params['n_switches']

    model1.last_eval, model1.next_eval = 0, eval_every_ep
    model2.last_eval, model2.next_eval = 0, eval_every_ep

    model1.last_switch, model1.next_switch = 0, switch_every_ep

    # print hyperparameters
    print(f"{battle_strat}-battle" if phase == phase.BATTLE else f"{draft_strat}-draft")
    print(f"algorithm={algorithm}, training_mode={training_mode}")
    print(f"seed={seed}, num_processes={num_processes}, train_episodes={train_episodes}, "
          f"eval_episodes={eval_episodes}, num_evals={num_evals}")
    print(f"num_trials={num_trials}, num_warmup_trials={num_warmup_trials}, "
          f"optimize_for={optimize_for}")
    print(f"path={path}")
    print(params)

    def make_evaluate(eval_env):
        def evaluate(model):
            """
            Evaluates a model.
            :param model: (stable_baselines model) Model to be evaluated.
            :return: The mean (win rate) and standard deviation.
            """
            # initialize structures
            episode_rewards = [[0.0] for _ in range(eval_env.num_envs)]

            # set seeds
            for i in range(num_processes):
                current_seed = eval_seed + (eval_episodes // num_processes) * i

                eval_env.env_method('seed', current_seed, indices=[i])

            # reset the env
            obs = eval_env.reset()
            states = None
            dones = [False] * num_processes
            episodes = 0

            action_hist = [0] * eval_env.action_space.n

            # runs `num_steps` steps
            while True:
                # get a deterministic prediction from model
                if draft_strat == 'lstm':
                    actions, states = model.predict(obs, deterministic=True,
                                                    state=states, mask=dones)
                else:
                    actions, _ = model.predict(obs, deterministic=True)

                for action in actions:
                    action_hist[action] += 1

                # do the predicted action and save the outcome
                obs, rewards, dones, _ = eval_env.step(actions)

                # save current reward into episode rewards
                for i in range(eval_env.num_envs):
                    episode_rewards[i][-1] += rewards[i]

                    if dones[i]:
                        episode_rewards[i].append(0.0)

                        episodes += 1

                if any(dones):
                    if episodes >= eval_episodes:
                        print("action histogram:", action_hist)

                        break

            all_rewards = []

            # flatten episode rewards lists
            for part in episode_rewards:
                all_rewards.extend(part)

            # return the mean reward and standard deviation from all episodes
            return mean(all_rewards), stdev(all_rewards)

        return evaluate

    def callback2(_locals, _globals):
        episodes_so_far = sum(env2.get_attr('episodes'))

        # if it is time to evaluate, do so
        if episodes_so_far >= model2.next_eval:
            # save models
            model2.save(model_path2 + f'/{episodes_so_far}-episodes')

            # evaluate the models and get the metrics
            print(f"Evaluating player 2... ({episodes_so_far})")
            mean2, std2 = make_evaluate(eval_env2)(model2)
            print(f"Done: {mean2}")
            print()

            if optimize_for == PlayerOrder.SECOND:
                results[0][0].append(mean2)
                results[0][1].append(std2)
            else:
                results[1][0].append(mean2)
                results[1][1].append(std2)

            model2.last_eval = episodes_so_far
            model2.next_eval += eval_every_ep

        return episodes_so_far < sum(env1.get_attr('episodes'))

    def callback(_locals, _globals):
        episodes_so_far = sum(env1.get_attr('episodes'))

        # if it is time to evaluate, do so
        if episodes_so_far >= model1.next_eval:
            # save model
            model1.save(model_path1 + f'/{episodes_so_far}-episodes')

            # evaluate the models and get the metrics
            print(f"Evaluating player 1... ({episodes_so_far})")
            mean1, std1 = make_evaluate(eval_env1)(model1)
            print(f"Done: {mean1}")
            print()

            if optimize_for == PlayerOrder.FIRST:
                results[0][0].append(mean1)
                results[0][1].append(std1)
            else:
                results[1][0].append(mean1)
                results[1][1].append(std1)

            model1.last_eval = episodes_so_far
            model1.next_eval += eval_every_ep

        # if it is time to switch, do so
        if episodes_so_far >= model1.next_switch:
            # train the second player model
            model2.learn(total_timesteps=1000000000,
                         reset_num_timesteps=False, callback=callback2)

            # update parameters on surrogate models
            env1.env_method('update_parameters', model2.get_parameters())
            env2.env_method('update_parameters', model1.get_parameters())

            model1.last_switch = episodes_so_far
            model1.next_switch += switch_every_ep

        return episodes_so_far < train_episodes

    # evaluate the initial models
    print(f"Evaluating player 1... ({sum(env1.get_attr('episodes'))})")
    mean_reward1, std_reward1 = make_evaluate(eval_env1)(model1)
    print(f"Done: {mean_reward1}")
    print()

    print(f"Evaluating player 2... ({sum(env2.get_attr('episodes'))})")
    mean_reward2, std_reward2 = make_evaluate(eval_env2)(model2)
    print(f"Done: {mean_reward2}")
    print()

    # train the first player model
    model1.learn(total_timesteps=1000000000, callback=callback)

    # train the second player model
    model2.learn(total_timesteps=1000000000,
                 reset_num_timesteps=False, callback=callback2)

    # evaluate the final models
    print(f"Evaluating player 1... ({sum(env1.get_attr('episodes'))})")
    mean_reward1, std_reward1 = make_evaluate(eval_env1)(model1)
    print(f"Done: {mean_reward1}")
    print()

    print(f"Evaluating player 2... ({sum(env2.get_attr('episodes'))})")
    mean_reward2, std_reward2 = make_evaluate(eval_env2)(model2)
    print(f"Done: {mean_reward2}")
    print()

    if optimize_for == PlayerOrder.SECOND:
        mean_reward1, mean_reward2 = mean_reward2, mean_reward1
        std_reward1, std_reward2 = std_reward2, std_reward1

    results[0][0].append(mean_reward1)
    results[1][0].append(mean_reward2)
    results[0][1].append(std_reward1)
    results[1][1].append(std_reward2)

    # save the final models
    model1.save(model_path1 + '/final')
    model2.save(model_path2 + '/final')

    # close the envs
    for e in (env1, env2, eval_env1, eval_env2):
        e.close()

    # get and print end time
    end_time = str(datetime.now())
    print('End time:', end_time)

    # save model info to results file
    with open(path + '/' + 'results.txt', 'a') as file:
        file.write(json.dumps(dict(id=counter, **params, results=results,
                                   start_time=start_time,
                                   end_time=end_time), indent=2))

    # calculate and return the metrics
    main_metric, aux_metric = -max(results[0][0]), -max(results[1][0])

    if optimize_for == PlayerOrder.SECOND:
        main_metric, aux_metric = aux_metric, main_metric

    return {'loss': main_metric, 'loss2': aux_metric, 'status': STATUS_OK}


def self_play(params):
    global counter

    counter += 1

    # get and print start time
    start_time = str(datetime.now())
    print('Start time:', start_time)

    if algorithm == 'ppo2':
        # ensure integer hyperparams
        params['n_steps'] = int(params['n_steps'])
        params['nminibatches'] = int(params['nminibatches'])
        params['noptepochs'] = int(params['noptepochs'])

        # ensure nminibatches <= n_steps
        params['nminibatches'] = min(params['nminibatches'],
                                     params['n_steps'])

        # ensure n_steps % nminibatches == 0
        while params['n_steps'] % params['nminibatches'] != 0:
            params['nminibatches'] -= 1

    # build the env
    env = []

    for i in range(num_processes):
        current_seed = seed + (train_episodes // num_processes) * i

        env.append(env_builder(current_seed, True, **params))

    env = SubprocVecEnv(env, start_method='spawn')

    env.env_method('create_model', **params)

    # build the evaluation envs
    eval_seed = seed + train_episodes

    eval_env = []
    eval_env2 = []

    for i in range(num_processes):
        current_seed = eval_seed + (eval_episodes // num_processes) * i

        eval_env.append(eval_env_builder(current_seed, True, **params))
        eval_env2.append(eval_env_builder_battle2(current_seed, True, **params))

    eval_env = SubprocVecEnv(eval_env, start_method='spawn')
    eval_env2 = SubprocVecEnv(eval_env2, start_method='spawn')

    # build the model
    model = model_builder(env, **params)

    # update parameters on surrogate models
    env.env_method('update_parameters', model.get_parameters())

    # create the model name
    model_name = f'{counter}'

    # build the model path
    model_path = path + '/' + model_name

    # set tensorflow log dir
    model.tensorboard_log = model_path

    # create necessary folders
    os.makedirs(model_path, exist_ok=True)

    # save starting model
    model.save(model_path + '/0-episodes')

    results = [[], [], []]

    # calculate utilities
    eval_every_ep = train_episodes / num_evals
    switch_every_ep = train_episodes / params['n_switches']

    model.last_eval, model.next_eval = 0, eval_every_ep
    model.last_switch, model.next_switch = 0, switch_every_ep

    # print hyperparameters
    print(f"{battle_strat}-battle" if phase == phase.BATTLE else f"{draft_strat}-draft")
    print(f"algorithm={algorithm}, training_mode={training_mode}")
    print(f"seed={seed}, num_processes={num_processes}, train_episodes={train_episodes}, "
          f"eval_episodes={eval_episodes}, num_evals={num_evals}")
    print(f"num_trials={num_trials}, num_warmup_trials={num_warmup_trials}, "
          f"optimize_for={optimize_for}")
    print(f"path={path}")
    print(params)

    def make_evaluate(eval_env):
        def evaluate(model):
            """
            Evaluates a model.
            :param model: (stable_baselines model) Model to be evaluated.
            :return: The mean (win rate) and standard deviation.
            """
            # initialize structures
            episode_rewards = [[0.0] for _ in range(eval_env.num_envs)]
            episode_lengths = []
            episode_wins = []

            # set seeds
            for i in range(num_processes):
                current_seed = eval_seed + (eval_episodes // num_processes) * i

                eval_env.env_method('seed', current_seed, indices=[i])

            # reset the env
            obs = eval_env.reset()
            episodes = 0

            model.set_env(eval_env)

            action_hist = [0] * eval_env.action_space.n

            while True:
                # get current turns
                turns = eval_env.get_attr('turn')

                # get a deterministic prediction from model
                actions, _ = model.predict(obs, deterministic=True)

                # print(actions[0], sum(eval_env.get_attr('action_mask')[0]))

                for action in actions:
                    action_hist[action] += 1

                # do the predicted action and save the outcome
                obs, rewards, dones, info = eval_env.step(actions)

                # save current reward into episode rewards
                for i in range(eval_env.num_envs):
                    episode_rewards[i][-1] += rewards[i]

                    if dones[i]:
                        episode_wins.append(1 if rewards[i] > 0 else 0)
                        episode_lengths.append(turns[i])
                        # print(rewards[i])
                        episode_rewards[i].append(0.0)

                        episodes += 1

                if any(dones):
                    if episodes >= eval_episodes:
                        for i in range(eval_env.num_envs):
                            if episode_rewards[i][-1] == 0:
                                episode_rewards[i].pop()

                        print("action histogram:", action_hist)

                        break

            all_rewards = []

            # flatten episode rewards lists
            for part in episode_rewards:
                all_rewards.extend(part)

            model.set_env(env)

            # return the mean reward and standard deviation from all episodes
            return mean(all_rewards), mean(episode_wins), mean(episode_lengths)

        return evaluate

    def callback(_locals, _globals):
        episodes_so_far = sum(env.get_attr('episodes'))

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model.save(model_path + f'/{episodes_so_far}-episodes')

            # evaluate the models and get the metrics
            print(f"Evaluating... ({episodes_so_far})")
            mean_reward, win_rate, mean_length = make_evaluate(eval_env)(model)
            print(f"vs max-attack: {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
            mean_reward, win_rate, mean_length = make_evaluate(eval_env2)(model)
            print(f"vs random: {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
            print()

            results[0].append(mean_reward)
            results[1].append(win_rate)
            results[2].append(mean_length)

            model.last_eval = episodes_so_far
            model.next_eval += eval_every_ep

        # if it is time to switch, do so
        if episodes_so_far >= model.next_switch:
            # update parameters on surrogate models
            env.env_method('update_parameters', model.get_parameters())

            model.last_switch = episodes_so_far
            model.next_switch += switch_every_ep

        return episodes_so_far < train_episodes

    # evaluate the initial model
    print("Evaluating... (0)")
    mean_reward, win_rate, mean_length = make_evaluate(eval_env)(model)
    print(f"vs max-attack: {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
    mean_reward, win_rate, mean_length = make_evaluate(eval_env2)(model)
    print(f"vs random: {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
    print()
    
    try:
        # train the model
        model.learn(total_timesteps=1000000000, callback=callback)
    except KeyboardInterrupt:
        print(f'Training stopped at {sum(env.get_attr("episodes"))}')

    # update opponent
    env.env_method('update_parameters', model.get_parameters())

    # evaluate the final model
    print("Evaluating... (final)")
    mean_reward, win_rate, mean_length = make_evaluate(eval_env)(model)
    print(f"vs max-attack: {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
    mean_reward, win_rate, mean_length = make_evaluate(eval_env2)(model)
    print(f"vs random: {mean_reward} mr / {win_rate * 100}% wr / {mean_length} ml")
    print()

    results[0].append(mean_reward)
    results[1].append(win_rate)
    results[2].append(mean_length)

    # save the final model
    model.save(model_path + '/final')

    # close the envs
    for e in (env, eval_env):
        e.close()

    # get and print end time
    end_time = str(datetime.now())
    print('End time:', end_time)

    # save model info to results file
    with open(path + '/' + 'results.txt', 'a') as file:
        file.write(json.dumps(dict(id=counter, **params, results=results,
                                   start_time=start_time,
                                   end_time=end_time), indent=2))

    return {'loss': -max(results[0]), 'status': STATUS_OK}


if training_mode == 'normal':
    train_and_eval = normal_training
elif training_mode == 'self-play':
    train_and_eval = self_play
elif training_mode == 'interleaved-self-play':
    train_and_eval = interleaved_self_play
else:
    raise ValueError("Invalid training mode.")

if __name__ == '__main__':
    try:
        with open(path + '/trials.p', 'rb') as trials_file:
            trials = pickle.load(trials_file)

        with open(path + '/rstate.p', 'rb') as random_state_file:
            random_state = pickle.load(random_state_file)

        finished_trials = len(trials)
        print(f'Found run state file with {finished_trials} trials.')

        counter = finished_trials
    except FileNotFoundError:
        trials = Trials()
        finished_trials = 0
        random_state = np.random.RandomState(seed)

    # noinspection PyBroadException
    try:
        algo = partial(tpe.suggest,
                       n_startup_jobs=max(0, num_warmup_trials - finished_trials))

        best_param = fmin(train_and_eval, param_dict, algo=algo,
                          max_evals=num_trials, trials=trials,
                          rstate=random_state)

        loss = [x['result']['loss'] for x in trials.trials]

        print("")
        print("##### Results")
        print("Score best parameters: ", min(loss) * -1)
        print("Best parameters: ", best_param)
    finally:
        with open(path + '/trials.p', 'wb') as trials_file:
            pickle.dump(trials, trials_file)

        with open(path + '/rstate.p', 'wb') as random_state_file:
            pickle.dump(random_state, random_state_file)
