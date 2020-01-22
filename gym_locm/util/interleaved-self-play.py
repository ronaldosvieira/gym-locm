import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import numpy as np
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll import scope
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from statistics import mean, stdev

from gym_locm.agents import MaxAttackBattleAgent, MaxAttackDraftAgent
from gym_locm.envs.draft import LOCMDraftSelfPlayEnv, LOCMDraftSingleEnv

# parameters
seed = 156123055
eval_seed = 402455446
num_processes = 4

train_steps = 30 * 33334
eval_steps = 30 * 33334
num_evals = 10

num_trials = 150

path = 'models/hyp-search/baseline-interleaved'

param_dict = {
    'layers': hp.uniformint('layers', 1, 3),
    'neurons': hp.uniformint('neurons', 24, 128),
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

# initializations
counter = 0
make_battle_agents = lambda: (MaxAttackBattleAgent(), MaxAttackBattleAgent())


def env_builder(seed, play_first=True, **params):
    env = LOCMDraftSelfPlayEnv2(seed=seed, battle_agents=make_battle_agents())
    env.play_first = play_first

    return lambda: env


def eval_env_builder(seed, play_first=True, **params):
    env = LOCMDraftSingleEnv(seed=seed, draft_agent=MaxAttackDraftAgent(),
                             battle_agents=make_battle_agents())
    env.play_first = play_first

    return lambda: env


def model_builder(env, **params):
    return PPO2(MlpPolicy, env, verbose=0, gamma=1,
                policy_kwargs=dict(net_arch=[params['neurons']] * params['layers']),
                n_steps=params['n_steps'],
                nminibatches=params['nminibatches'],
                noptepochs=params['noptepochs'],
                cliprange=params['cliprange'],
                vf_coef=params['vf_coef'],
                ent_coef=params['ent_coef'],
                learning_rate=params['learning_rate'])


class LOCMDraftSelfPlayEnv2(LOCMDraftSelfPlayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_model(self, **params):
        self.model = model_builder(DummyVecEnv([lambda: self]), **params)


def train_and_eval(params):
    global counter

    counter += 1

    # print generated hyperparameters
    print(params)

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
    env1 = [env_builder(seed, True, **params)
            for _ in range(num_processes)]
    env2 = [env_builder(seed, False, **params)
            for _ in range(num_processes)]

    env1 = SubprocVecEnv(env1, start_method='spawn')
    env2 = SubprocVecEnv(env2, start_method='spawn')

    env1.env_method('create_model', **params)
    env2.env_method('create_model', **params)

    # build the evaluation envs
    eval_env1 = [eval_env_builder(seed, True, **params)
                 for _ in range(num_processes)]
    eval_env2 = [eval_env_builder(seed, False, **params)
                 for _ in range(num_processes)]

    eval_env1 = SubprocVecEnv(eval_env1, start_method='spawn')
    eval_env2 = SubprocVecEnv(eval_env2, start_method='spawn')

    # build the models
    model1 = model_builder(env1, **params)
    model2 = model_builder(env2, **params)

    # create the model name
    model_name = f'{counter}'

    # build model paths
    model_path1 = path + '/' + model_name + '/1st'
    model_path2 = path + '/' + model_name + '/2nd'

    # set tensorflow log dir
    model1.tensorboard_log = model_path1
    model2.tensorboard_log = model_path2

    # create necessary folders
    os.makedirs(model_path1, exist_ok=True)
    os.makedirs(model_path2, exist_ok=True)

    # save starting models
    model1.save(model_path1 + '/0-steps')
    model2.save(model_path2 + '/0-steps')

    results = [[], []]

    model1.callback_counter = 0

    eval_every = train_steps // (model1.n_steps * num_processes)
    eval_every //= num_evals

    def make_evaluate(eval_env):
        def evaluate(model):
            """
            Evaluates a model.
            :param model: (stable_baselines model) Model to be evaluated.
            :return: The mean (win rate) and standard deviation.
            """
            # initialize structures
            episode_rewards = [[0.0] for _ in range(eval_env.num_envs)]
            num_steps = int(eval_steps / num_processes)

            # reset the env
            eval_env.env_method('seed', eval_seed)
            obs = eval_env.reset()

            # runs `num_steps` steps
            for j in range(num_steps):
                # get a deterministic prediction from model
                actions, _ = model.predict(obs, deterministic=True)

                # do the predicted action and save the outcome
                obs, rewards, dones, _ = eval_env.step(actions)

                # save current reward into episode rewards
                for i in range(eval_env.num_envs):
                    episode_rewards[i][-1] += rewards[i]

                    if dones[i]:
                        episode_rewards[i].append(0.0)

            all_rewards = []

            # flatten episode rewards lists
            for part in episode_rewards:
                all_rewards.extend(part)

            # return the mean reward and standard deviation from all episodes
            return mean(all_rewards), stdev(all_rewards)

        return evaluate

    def callback(_locals, _globals):
        model = _locals['self']

        model.callback_counter += 1

        # if it is time to evaluate, do so
        if model.callback_counter % eval_every == 0:
            # update second player's opponent
            env2.env_method('update_parameters', model1.get_parameters())

            # train the second player model
            model2.learn(total_timesteps=train_steps // (num_evals + 1),
                         seed=seed, tb_log_name='tf',
                         reset_num_timesteps=False)

            # update first player's opponent
            env1.env_method('update_parameters', model2.get_parameters())

            # evaluate the models and get the metrics
            mean1, std1 = make_evaluate(eval_env1)(model)
            mean2, std2 = make_evaluate(eval_env2)(model2)

            results[0].append((mean1, std1))
            results[1].append((mean2, std2))

            # save models
            model1.save(model_path1 + f'/{model1.num_timesteps}-steps')
            model2.save(model_path2 + f'/{model2.num_timesteps}-steps')

    # train the first player model
    model1.learn(total_timesteps=train_steps,
                 callback=callback,
                 seed=seed,
                 tb_log_name='tf')

    # update second player's opponent
    env2.env_method('update_parameters', model1.get_parameters())

    # train the second player model
    model2.learn(total_timesteps=train_steps // (num_evals + 1),
                 seed=seed, tb_log_name='tf', reset_num_timesteps=False)

    # update first player's opponent
    env1.env_method('update_parameters', model2.get_parameters())

    # evaluate the final models
    mean_reward1, std_reward1 = make_evaluate(eval_env1)(model1)
    mean_reward2, std_reward2 = make_evaluate(eval_env2)(model2)

    results[0].append((mean_reward1, std_reward1))
    results[1].append((mean_reward2, std_reward2))

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

    return {'loss': -max(results[0]),
            'loss2': -max(results[1]),
            'status': STATUS_OK}


if __name__ == '__main__':
    trials = Trials()
    best_param = fmin(train_and_eval, param_dict, algo=tpe.suggest,
                      max_evals=num_trials, trials=trials,
                      rstate=np.random.RandomState(seed))
    loss = [x['result']['loss'] for x in trials.trials]
    loss2 = [x['result']['loss2'] for x in trials.trials]

    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
