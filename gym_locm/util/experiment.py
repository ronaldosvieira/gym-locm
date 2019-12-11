import itertools
import os
import json
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
        all_results = []

        # for each config,
        for i, config in enumerate(self.configs):
            results = []

            # for each seed, save the win rate
            for seed in self.seeds:
                _, means, _ = config.run(path=self.path + f'/cfg{i}',
                                         seed=seed)

                results.append(means)

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
        with open(self.path + '/' + 'results.txt', 'w') as file:
            file.write(json.dumps({'samples': all_results.tolist(),
                                   'statistics': statistics,
                                   'p-values': p_values}))

        return all_results, statistics, p_values


class Configuration:
    def __init__(self, env_builder, model_builder, eval_env_builder=None, *,
                 before=None, after=None, each_eval=None,
                 train_steps=30 * 333334, eval_steps=30 * 33334,
                 eval_frequency=30 * 33334, num_processes=1):
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
        :param eval_frequency: (int) Amount of timesteps between each
        evaluation.
        :param num_processes: (int) Amount of processes to use.
        """
        self.env_builder = env_builder
        self.model_builder = model_builder
        self.eval_env_builder = \
            env_builder if eval_env_builder is None else eval_env_builder

        self.num_processes = num_processes
        self.train_steps, self.eval_steps = train_steps, eval_steps
        self.eval_frequency = eval_frequency
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

            # if it is time to evaluate, do so
            if timestep % self.eval_frequency == 0:
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
