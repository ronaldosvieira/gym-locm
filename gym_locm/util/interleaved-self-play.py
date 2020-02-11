import json
import os
import warnings
from datetime import datetime

from gym_locm.envs.battle import LOCMBattleSelfPlayEnv, LOCMBattleSingleEnv

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import numpy as np
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll import scope
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from statistics import mean, stdev

from gym_locm.engine import PlayerOrder, Phase
from gym_locm.agents import MaxAttackBattleAgent, MaxAttackDraftAgent, IceboxDraftAgent
from gym_locm.envs.draft import LOCMDraftSelfPlayEnv, LOCMDraftSingleEnv, LOCMDraftEnv

# which phase to train
phase = Phase.DRAFT

# draft-related parameters
lstm = True
history = False

# battle-related parameters
clip_invalid_actions = False

# training parameters
seed = 96732
num_processes = 4
train_episodes = 30000
eval_episodes = 3000
num_evals = 10

# bayesian optimization parameters
num_trials = 50
num_warmup_trials = 20
optimize_for = PlayerOrder.FIRST

# where to save the model
path = 'models/hyp-search/lstm2-draft-1st-player'

# hyperparameter space
param_dict = {
    'n_switches': hp.choice('n_switches', [10, 100, 1000, 10000]),
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
                                   np.log(0.01))
}

if lstm:
    param_dict['layers'] = hp.uniformint('layers', 0, 2)
    param_dict['nminibatches'] = hp.choice('nminibatches', [1])

# initializations
counter = 0
make_draft_agents = lambda: (IceboxDraftAgent(), IceboxDraftAgent())
make_battle_agents = lambda: (MaxAttackBattleAgent(), MaxAttackBattleAgent())


def env_builder_draft(seed, play_first=True, **params):
    env = LOCMDraftSelfPlayEnv2(seed=seed, battle_agents=make_battle_agents(),
                                use_draft_history=history)
    env.play_first = play_first

    return lambda: env


def env_builder_battle(seed, play_first=True, **params):
    env = LOCMBattleSelfPlayEnv2(seed=seed, draft_agents=make_draft_agents())
    env.play_first = play_first

    return lambda: env


def eval_env_builder_draft(seed, play_first=True, **params):
    env = LOCMDraftSingleEnv(seed=seed, draft_agent=MaxAttackDraftAgent(),
                             battle_agents=make_battle_agents(),
                             use_draft_history=history)
    env.play_first = play_first

    return lambda: env


def eval_env_builder_battle(seed, play_first=True, **params):
    env = LOCMBattleSingleEnv(seed=seed, battle_agent=MaxAttackBattleAgent(),
                              draft_agents=make_draft_agents())
    env.play_first = play_first

    return lambda: env


def model_builder_mlp(env, **params):
    net_arch = [params['neurons']] * params['layers']

    return PPO2(MlpPolicy, env, verbose=0, gamma=1,
                policy_kwargs=dict(net_arch=net_arch),
                n_steps=params['n_steps'],
                nminibatches=params['nminibatches'],
                noptepochs=params['noptepochs'],
                cliprange=params['cliprange'],
                vf_coef=params['vf_coef'],
                ent_coef=params['ent_coef'],
                learning_rate=params['learning_rate'],
                tensorboard_log=None)


def model_builder_lstm(env, **params):
    net_arch = ['lstm'] + [params['neurons']] * params['layers']

    return PPO2(MlpLstmPolicy, env, verbose=0, gamma=1,
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
elif phase == Phase.BATTLE:
    env_builder = env_builder_battle
    eval_env_builder = eval_env_builder_battle

model_builder = model_builder_lstm if lstm else model_builder_mlp


class LOCMBattleSelfPlayEnv2(LOCMBattleSelfPlayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_model(self, **params):
        self.model = model_builder(DummyVecEnv([lambda: self]), **params)


class LOCMDraftSelfPlayEnv2(LOCMDraftSelfPlayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rstate = None
        self.done = [False] + [True] * (num_processes - 1)

    def create_model(self, **params):
        if lstm:
            self.rstate = np.zeros(shape=(num_processes, params['neurons'] * 2))

        env = [env_builder(0, **params) for _ in range(num_processes)]
        env = DummyVecEnv(env)

        self.model = model_builder(env, **params)

    if lstm:
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


def train_and_eval(params):
    global counter

    counter += 1

    # get and print start time
    start_time = str(datetime.now())
    print('Start time:', start_time)

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

            eval_env.set_attr('episodes', 0)

            # runs `num_steps` steps
            while True:
                # get a deterministic prediction from model
                actions, _ = model.predict(obs, deterministic=True)

                # do the predicted action and save the outcome
                obs, rewards, dones, _ = eval_env.step(actions)

                # save current reward into episode rewards
                for i in range(eval_env.num_envs):
                    episode_rewards[i][-1] += rewards[i]

                    if dones[i]:
                        episode_rewards[i].append(0.0)

                if any(dones):
                    if sum(eval_env.get_attr('episodes')) >= eval_episodes:
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

        # if it is time to switch, do so
        if episodes_so_far >= model1.next_switch:
            # train the second player model
            model2.learn(total_timesteps=1000000000,
                         seed=seed + model1.last_switch,
                         reset_num_timesteps=False, callback=callback2)

            # update parameters on surrogate models
            env1.env_method('update_parameters', model2.get_parameters())
            env2.env_method('update_parameters', model1.get_parameters())

            model1.last_switch = episodes_so_far
            model1.next_switch += switch_every_ep

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

        return episodes_so_far < train_episodes

    # train the first player model
    model1.learn(total_timesteps=1000000000, callback=callback, seed=seed)

    # update second player's opponent
    env2.env_method('update_parameters', model1.get_parameters())

    # train the second player model
    model2.learn(total_timesteps=1000000000, seed=seed + model1.last_switch,
                 reset_num_timesteps=False, callback=callback2)

    # update first player's opponent
    env1.env_method('update_parameters', model2.get_parameters())

    # evaluate the final models
    mean_reward1, std_reward1 = make_evaluate(eval_env1)(model1)
    mean_reward2, std_reward2 = make_evaluate(eval_env2)(model2)

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
        loss2 = [x['result']['loss2'] for x in trials.trials]

        print("")
        print("##### Results")
        print("Score best parameters: ", min(loss) * -1)
        print("Best parameters: ", best_param)
    finally:
        with open(path + '/trials.p', 'wb') as trials_file:
            pickle.dump(trials, trials_file)

        with open(path + '/rstate.p', 'wb') as random_state_file:
            pickle.dump(random_state, random_state_file)
