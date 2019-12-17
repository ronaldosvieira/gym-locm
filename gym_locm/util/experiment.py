import itertools
import os
import json
from datetime import datetime

import numpy as np
from random import randint
from stable_baselines.common.vec_env import SubprocVecEnv
from scipy.stats import ttest_ind, ttest_rel
from statistics import mean, stdev


class Experiment:
    def __init__(self, config1, config2, sample_size=50, path='.',
                 paired=False, seeds=None):
        """
        Experiment class. Checks whether two configurations yield
        significantly different results.
        :param config1: (Configuration) First configuration to check.
        :param config2: (Configuration) Second configuration to check.
        :param sample_size: (int) Size of the sample to be generated from each
        configuration.
        :param path: Path to save the resulting models.
        """
        self.configs = config1, config2
        self.path = path
        self.paired = paired

        if seeds is None:
            self.seeds = [randint(0, 10e8) for _ in range(sample_size)]
        else:
            self.seeds = seeds

    def run(self, analyze_curve=False):
        """
        Runs each configuration to get a sample of size equal to `sample_size`,
        then apply the Welch's test to them.
        :param analyze_curve: Whether tests should be done for the entire
        learning curve or just for the final evaluation
        :return: The samples, the t-statistic and the p-value. If
        `analyze_curve=True`, multiple t-statistics and p-values are returned.
        """
        # create dirs
        os.makedirs(self.path, exist_ok=True)

        def write_result(info):
            with open(self.path + '/' + 'results.csv', 'a') as file:
                file.write(';'.join(map(str, info)) + '\n')

        write_title = True
        all_results = []

        # for each config,
        for i, config in enumerate(self.configs):
            results = []

            # for each seed, save the win rate
            for seed in self.seeds:
                _, means, _ = config.run(path=self.path + f'/cfg{i}',
                                         seed=seed)

                results.append(means)

                if write_title:
                    write_result(['timestamp', 'cfg_num', 'seed',
                                  *[f'eval{i}' for i in range(len(means))]])
                    write_title = False

                write_result([datetime.now(), i, seed, *means])

            all_results.append(results)

        # rearrange the axes so that the array is first indexed by evaluation
        all_results = np.moveaxis(np.array(all_results), -1, 0)

        # if not analyzing the entire curve, keep only last evaluation
        if not analyze_curve:
            all_results = all_results[-1:]

        statistics, p_values = [], []

        # for each sample from the two configurations, do the appropriate test
        for evaluation in all_results:
            if self.paired:
                statistic, p_value = ttest_rel(*evaluation)
            else:
                statistic, p_value = ttest_ind(*evaluation, equal_var=False)

            statistics.append(statistic)
            p_values.append(p_value)

        # save the samples and the results in a text file
        with open(self.path + '/' + 'stats.txt', 'w') as file:
            file.write(json.dumps({'samples': all_results.tolist(),
                                   'statistics': statistics,
                                   'p-values': p_values},
                                  indent=4))

        return all_results, statistics, p_values


class Configuration:
    def __init__(self, env_builder, model_builder, eval_env_builder=None, *,
                 before=None, after=None, each_eval=None,
                 train_steps=30 * 333334, eval_steps=30 * 33334,
                 num_evals=10, num_processes=1):
        """
        Configuration class. Saves all the relevant hyperparameters.
        :param env_builder: (seed -> env) Function that builds an env.
        :param model_builder: (env -> model) Function that builds a model.
        Model should follow the stable_baselines interface. See more at
        https://github.com/hill-a/stable-baselines.
        :param eval_env_builder: (seed -> env) Function that builds an env for
        evaluation.
        :param before: (model, env -> None) Function to be run before training.
        :param after: (model, env -> None) Function to be run after training.
        :param each_eval: (model, env -> None) Function to be run after each
        evaluation.
        :param train_steps: (int) Amount of timesteps to train for.
        :param eval_steps: (int) Amount of timesteps to evaluate for.
        :param num_evals: (int) Amount of evaluations throughout training.
        :param num_processes: (int) Amount of processes to use.
        """
        self.env_builder = env_builder
        self.model_builder = model_builder
        self.eval_env_builder = \
            env_builder if eval_env_builder is None else eval_env_builder

        self.num_processes = num_processes
        self.train_steps, self.eval_steps = train_steps, eval_steps
        self.num_evals = num_evals
        self.before, self.after, self.each_eval = before, after, each_eval

    def run(self, path='.', seed=None):
        """
        Trains a model with the specified configuration.
        :param path: Path to save the resulting models.
        :param seed: Seed to use on the training and envs.
        :return: the final model, the means and the standard deviations
        obtained from the evaluations.
        """
        # build the env
        env = [lambda: self.env_builder(seed)
               for _ in range(self.num_processes)]
        env = SubprocVecEnv(env, start_method='spawn')

        eval_seed = (seed + self.train_steps) * 2
        eval_env = [lambda: self.eval_env_builder(eval_seed)
                    for _ in range(self.num_processes)]
        eval_env = SubprocVecEnv(eval_env, start_method='spawn')

        # build the model
        model = self.model_builder(env)

        # call `before` callback
        if self.before is not None:
            self.before(model, env)

        # create necessary folders and save the starting model
        os.makedirs(path + '/' + str(seed), exist_ok=True)
        model.save(path + '/' + str(seed) + '/0-steps')

        means, stdevs = [], []

        model.callback_counter = 0
        eval_every = model.train_steps // (model.n_steps * self.num_processes)
        eval_every //= self.num_evals

        def evaluate(model):
            """
            Evaluates a model.
            :param model: (stable_baselines model) Model to be evaluated.
            :return: The mean (win rate) and standard deviation.
            """
            # initialize structures
            episode_rewards = [[0.0] for _ in range(eval_env.num_envs)]
            num_steps = int(self.eval_steps / self.num_processes)

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

        def callback(_locals, _globals):
            model = _locals['self']
            timestep = _locals["timestep"] + 1

            model.callback_counter += 1

            # if it is time to evaluate, do so
            if model.callback_counter % eval_every == 0:
                # evaluate the model and get the metrics
                mean, std = evaluate(model)

                means.append(mean)
                stdevs.append(std)

                # call `each_eval` callback
                if self.each_eval is not None:
                    self.each_eval(model, env, _locals)

                # save current model
                model.save(path + '/' + str(seed) + f'/{timestep}-steps')

        # train the model
        model.learn(total_timesteps=self.train_steps,
                    callback=callback, seed=seed)

        # evaluate the final model
        mean_reward, std_reward = evaluate(model)

        means.append(mean_reward)
        stdevs.append(std_reward)

        # save the final model
        model.save(path + '/' + str(seed) + f'/{model.num_timesteps}-steps')

        # call `after` callback
        if self.after is not None:
            self.after(model, env)

        # close the env
        env.close()
        eval_env.close()

        return model, means, stdevs


class RandomSearch(Configuration):
    def __init__(self, env_builder, model_builder, eval_env_builder,
                 param_dict, **kwargs):
        super(RandomSearch, self).__init__(env_builder, model_builder,
                                           eval_env_builder, **kwargs)

        self.param_dict = param_dict

    def hyparams_product(self):
        keys, values = self.param_dict.keys(), self.param_dict.values()

        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def _train(self, hyparams, path, seed=None):
        # pick a random seed, if none
        if seed is None:
            seed = randint(0, 10e8)

        # build the env
        env = [lambda: self.env_builder(seed, **hyparams)
               for _ in range(self.num_processes)]
        env = SubprocVecEnv(env, start_method='spawn')

        # build the evaluation env
        eval_seed = (seed + self.train_steps) * 2
        eval_env = [lambda: self.eval_env_builder(eval_seed, **hyparams)
                    for _ in range(self.num_processes)]
        eval_env = SubprocVecEnv(eval_env, start_method='spawn')

        # build the model
        model = self.model_builder(env=env, **hyparams)

        # call `before` callback
        if self.before is not None:
            self.before(model, env)

        # create the model name
        model_name = '-'.join(map(str, hyparams.values()))

        # set tensorflow log dir
        model.tensorboard_log = path + '/' + model_name

        # create necessary folders and save the starting model
        os.makedirs(path + '/' + model_name, exist_ok=True)
        model.save(path + '/' + model_name + '/0-steps')

        means, stdevs = [], []

        model.callback_counter = 0

        eval_every = self.train_steps // (model.n_steps * self.num_processes)
        eval_every //= self.num_evals

        def evaluate(model):
            """
            Evaluates a model.
            :param model: (stable_baselines model) Model to be evaluated.
            :return: The mean (win rate) and standard deviation.
            """
            # initialize structures
            episode_rewards = [[0.0] for _ in range(eval_env.num_envs)]
            num_steps = int(self.eval_steps / self.num_processes)

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

        def callback(_locals, _globals):
            model = _locals['self']
            timestep = _locals["timestep"] + 1

            model.callback_counter += 1

            # if it is time to evaluate, do so
            if model.callback_counter % eval_every == 0:
                # evaluate the model and get the metrics
                mean, std = evaluate(model)

                means.append(mean)
                stdevs.append(std)

                # call `each_eval` callback
                if self.each_eval is not None:
                    self.each_eval(model, env)

                # save current model
                model.save(path + '/' + model_name + f'/{timestep}-steps')

        # train the model
        model.learn(total_timesteps=self.train_steps,
                    callback=callback,
                    seed=seed,
                    tb_log_name='tf')

        # evaluate the final model
        mean_reward, std_reward = evaluate(model)

        means.append(mean_reward)
        stdevs.append(std_reward)

        # save the final model
        model.save(path + '/' + model_name + '/final')

        # call `after` callback
        if self.after is not None:
            self.after(model, env)

        # close the env
        env.close()
        eval_env.close()

        return model, means, stdevs

    def run(self, path='.', seed=None, times=50):
        """
        Trains models with the specified configuration and distribution
        of hyperparameters.
        :param path: Path to save the resulting models.
        :param seed: Seed to use on the training and envs.
        :param times: Amount of hyperparameters to test.
        :return: the final version of the best model, the means and the
        standard deviations obtained from it in the evaluations.
        """
        # initialize variables
        best_model = None
        best_mean = float("-inf")

        # for each run in the budget
        for run in range(times):
            # generate a set of hyperparameters
            hyparam_set = dict((k, v()) for k, v in self.param_dict.items())

            # ensure integer hyperparams
            hyparam_set['n_steps'] = int(hyparam_set['n_steps'])
            hyparam_set['nminibatches'] = int(hyparam_set['nminibatches'])
            hyparam_set['noptepochs'] = int(hyparam_set['noptepochs'])

            # ensure nminibatches <= n_steps
            hyparam_set['nminibatches'] = min(hyparam_set['nminibatches'],
                                              hyparam_set['n_steps'])

            # ensure n_steps % nminibatches == 0
            while hyparam_set['n_steps'] % hyparam_set['nminibatches'] != 0:
                hyparam_set['nminibatches'] -= 1

            # print generated hyperparameters
            print(f'### RUN {run} ###')
            print(hyparam_set)

            # get and print start time
            start_time = str(datetime.now())
            print('Start time:', start_time)

            # train and eval a model with the generated hyperparameters
            model, means, stdevs = self._train(hyparam_set, path, seed)

            # get and print end time
            end_time = str(datetime.now())
            print('End time:', end_time)

            # check if it's better than current best model
            if means[-1] > best_mean:
                best_mean, best_model = means[-1], model

            # save model info to results file
            with open(path + '/' + 'results.txt', 'a') as file:
                file.write(json.dumps(dict(**hyparam_set, means=means,
                                           stdevs=stdevs,
                                           start_time=start_time,
                                           end_time=end_time), indent=2))

        return best_model, best_mean
