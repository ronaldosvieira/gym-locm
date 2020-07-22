import json
import logging
import os
import time
import warnings
from datetime import datetime
from statistics import mean

# suppress tensorflow deprecated warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import SubprocVecEnv, VecEnv

from gym_locm.agents import Agent, MaxAttackDraftAgent, MaxAttackBattleAgent
from gym_locm.envs import LOCMDraftSingleEnv

verbose = True

if verbose:
    logging.basicConfig(level=logging.DEBUG)


class RLDraftAgent(Agent):
    def __init__(self, model):
        self.model = model

        self.hidden_states = None
        self.dones = None

    def seed(self, seed):
        pass

    def reset(self):
        self.hidden_states = None
        self.dones = None

    def act(self, state):
        actions, self.hidden_states = \
            self.model.predict(state, deterministic=True,
                               state=self.hidden_states, mask=self.dones)

        return actions


class TrainingSession:
    def __init__(self, env_builder, eval_env_builder, model_builder,
                 train_episodes, evaluation_episodes, num_evals,
                 play_first, params, path, seed, num_processes=1):
        # log start time
        start_time = time.perf_counter()

        # initialize logger
        self.logger = logging.getLogger('{0}.{1}'.format(__name__, type(self).__name__))

        # initialize parallel environments
        self.logger.debug("Initializing env...")
        env = []

        for i in range(num_processes):
            # no overlap between episodes at each process
            current_seed = seed + (train_episodes // num_processes) * i

            # create one env per process
            env.append(env_builder(seed=current_seed, play_first=play_first))

        # wrap envs in a vectorized env
        self.env: VecEnv = SubprocVecEnv(env, start_method='spawn')

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        self.evaluator: Evaluator = Evaluator(eval_env_builder, evaluation_episodes,
                                              seed + train_episodes, num_processes)

        # build the model
        self.logger.debug("Building the model...")
        self.model = model_builder(self.env, seed, **params)

        # create necessary folders
        os.makedirs(path, exist_ok=True)

        # set tensorflow log dir
        self.model.tensorflow_log = path

        # save parameters
        self.train_episodes = train_episodes
        self.num_evals = num_evals
        self.params = params
        self.path = path
        self.eval_frequency = train_episodes / num_evals

        # initialize control attributes
        self.model.last_eval = None
        self.model.next_eval = 0
        self.model.role_id = 0 if play_first else 1

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _evaluate(self, _locals=None, _globals=None):
        episodes_so_far = sum(self.env.get_attr('episodes'))

        # if it is time to evaluate, do so
        if episodes_so_far >= self.model.next_eval:
            # save model
            model_path = self.path + f'/{episodes_so_far}'
            self.model.save(model_path)
            self.logger.debug(f"Saved model at {model_path}.zip.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            mean_reward, ep_length, act_hist = \
                self.evaluator.run(RLDraftAgent(self.model),
                                   play_first=self.model.role_id == 0)

            end_time = time.perf_counter()
            self.logger.info(f"Finished evaluating "
                             f"({round(end_time - start_time, 3)}s). "
                             f"Avg. reward: {mean_reward}")

            # save the results
            self.checkpoints.append(episodes_so_far)
            self.win_rates.append((mean_reward + 1) / 2)
            self.episode_lengths.append(ep_length)
            self.action_histograms.append(act_hist)

            # update control attributes
            self.model.last_eval = episodes_so_far
            self.model.next_eval += self.eval_frequency

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        if training_is_finished:
            self.logger.debug(f"Training ended at {episodes_so_far} episodes")

        return not training_is_finished

    def run(self):
        # log start time
        start_time = datetime.now()
        self.logger.info("Training...")

        # save and evaluate starting model
        self._evaluate()

        try:
            # train the model
            self.model.learn(total_timesteps=30 * self.train_episodes,
                             callback=self._evaluate)
        except KeyboardInterrupt:
            pass

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._evaluate()

        # log end time
        end_time = datetime.now()
        self.logger.info(f"End of training. Time elapsed: {end_time - start_time}.")

        # save model info to results file
        compiled_results = self.checkpoints, self.win_rates, \
            self.episode_lengths, self.action_histograms

        results_path = self.path + '/results.txt'

        with open(results_path, 'a') as file:
            info = dict(**self.params, results=compiled_results,
                        start_time=str(start_time), end_time=str(end_time))
            info = json.dumps(info, indent=2)

            file.write(info)

        self.logger.debug(f"Results saved at {results_path}.")

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()

        return compiled_results


class Evaluator:
    def __init__(self, env_builder, episodes, seed, num_processes=1):
        # log start time
        start_time = time.perf_counter()

        # initialize logger
        self.logger = logging.getLogger('{0}.{1}'.format(__name__, type(self).__name__))

        # initialize parallel environments
        self.logger.debug("Initializing envs...")
        self.env: VecEnv = SubprocVecEnv([env_builder() for _ in range(num_processes)],
                                         start_method='spawn')

        # save parameters
        self.episodes = episodes
        self.seed = seed

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing evaluator "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def run(self, agent: Agent, play_first=True):
        """
        Evaluates an agent.
        :param agent: (gym_locm.agents.Agent) Agent to be evaluated.
        :return: A tuple containing the `mean_reward`, the `mean_length`
        and the `action_histogram` of the evaluation episodes.
        """
        # set appropriate seeds
        for i in range(self.env.num_envs):
            current_seed = self.seed + (self.episodes // self.env.num_envs) * i
            current_seed -= 1  # resetting the env increases the seed by one

            self.env.env_method('seed', current_seed, indices=[i])

        # set agent role
        self.env.set_attr('play_first', play_first)

        # reset the env
        observations = self.env.reset()

        # initialize metrics
        episodes_so_far = 0
        episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
        episode_lengths = [[0] for _ in range(self.env.num_envs)]
        action_histogram = [0] * self.env.action_space.n

        # run the episodes
        while True:
            # get the agent's action for all parallel envs
            # todo: do this in a more elegant way
            if isinstance(agent, RLDraftAgent):
                actions = agent.act(observations)
            else:
                observations = self.env.get_attr('state')
                actions = [agent.act(observation) for observation in observations]

            # update the action histogram
            for action in actions:
                action_histogram[action] += 1

            # perform the action and get the outcome
            observations, rewards, dones, _ = self.env.step(actions)

            # update metrics
            for i in range(self.env.num_envs):
                episode_rewards[i][-1] += rewards[i]
                episode_lengths[i][-1] += 1

                if dones[i]:
                    episode_rewards[i].append(0.0)
                    episode_lengths[i].append(0)

                    episodes_so_far += 1

            # check exiting condition
            if episodes_so_far >= self.episodes:
                break

        # join all parallel metrics
        all_rewards = [reward for rewards in episode_rewards
                       for reward in rewards]
        all_lengths = [length for lengths in episode_lengths
                       for length in lengths]

        # transform the action histogram in a probability distribution
        action_histogram = [action_freq / sum(action_histogram)
                            for action_freq in action_histogram]

        # cap any unsolicited additional episodes
        all_rewards = all_rewards[:self.episodes]
        all_lengths = all_lengths[:self.episodes]

        return mean(all_rewards), mean(all_lengths), action_histogram

    def close(self):
        self.env.close()


if __name__ == '__main__':
    def build_env(seed=None, play_first=True):
        env = LOCMDraftSingleEnv(seed=seed, draft_agent=MaxAttackDraftAgent(),
                                 battle_agents=(MaxAttackBattleAgent(), MaxAttackBattleAgent()),
                                 use_draft_history=False, use_mana_curve=False,
                                 play_first=play_first)

        return lambda: env


    def build_model(env, seed, neurons, layers, activation, n_steps, nminibatches,
                    noptepochs, cliprange, vf_coef, ent_coef, learning_rate):
        net_arch = [neurons] * layers
        activation = dict(tanh=tf.nn.tanh, relu=tf.nn.relu, elu=tf.nn.elu)[activation]

        return PPO2(MlpPolicy, env, verbose=0, gamma=1, seed=seed,
                    policy_kwargs=dict(net_arch=net_arch, act_fun=activation),
                    n_steps=n_steps, nminibatches=nminibatches,
                    noptepochs=noptepochs, cliprange=cliprange,
                    vf_coef=vf_coef, ent_coef=ent_coef,
                    learning_rate=learning_rate, tensorboard_log=None)


    params = {'layers': 1, 'neurons': 29, 'n_steps': 210, 'nminibatches': 30,
              'noptepochs': 19, 'cliprange': 0.1, 'vf_coef': 1.0,
              'ent_coef': 0.00781891437626065, 'learning_rate': 0.0001488768154153614,
              'activation': 'tanh'}

    ts = TrainingSession(build_env, build_env, build_model, 3000, 300, 12, True, params,
                        'models/trashcan/trash03', 36987, num_processes=2)

    ts.run()
