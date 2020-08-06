import json
import logging
import os
import time
import warnings
import numpy as np
from abc import abstractmethod
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
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import VecEnv, DummyVecEnv

from gym_locm.agents import Agent, MaxAttackDraftAgent, MaxAttackBattleAgent
from gym_locm.envs import LOCMDraftSingleEnv
from gym_locm.envs.draft import LOCMDraftSelfPlayEnv

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
    def __init__(self, params, path, seed):
        # initialize logger
        self.logger = logging.getLogger('{0}.{1}'.format(__name__,
                                                         type(self).__name__))

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []

        # save parameters
        self.params = params
        self.path = os.path.dirname(__file__) + "/../../" + path
        self.seed = seed

    @abstractmethod
    def _train(self):
        pass

    def run(self):
        # log start time
        start_time = datetime.now()
        self.logger.info("Training...")

        # do the training
        self._train()

        # log end time
        end_time = datetime.now()
        self.logger.info(f"End of training. Time elapsed: {end_time - start_time}.")

        # save model info to results file
        results_path = self.path + '/results.txt'

        with open(results_path, 'a') as file:
            info = dict(**self.params, seed=self.seed, checkpoints=self.checkpoints,
                        win_rates=self.win_rates, ep_lengths=self.episode_lengths,
                        action_histograms=self.action_histograms,
                        start_time=str(start_time), end_time=str(end_time))
            info = json.dumps(info, indent=2)

            file.write(info)

        self.logger.debug(f"Results saved at {results_path}.")


class FixedAdversary(TrainingSession):
    def __init__(self, env_builder, eval_env_builder, model_builder,
                 train_episodes, eval_episodes, num_evals,
                 play_first, params, path, seed, num_envs=1):
        super(FixedAdversary, self).__init__(params, path, seed)

        # log start time
        start_time = time.perf_counter()

        # initialize parallel environments
        self.logger.debug("Initializing training env...")
        env = []

        for i in range(num_envs):
            # no overlap between episodes at each concurrent env
            current_seed = seed + (train_episodes // num_envs) * i

            # create the env
            env.append(env_builder(seed=current_seed, play_first=play_first))

        # wrap envs in a vectorized env
        self.env: VecEnv = DummyVecEnv([lambda: e for e in env])

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        self.evaluator: Evaluator = Evaluator(eval_env_builder, eval_episodes,
                                              seed + train_episodes, num_envs)

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
        self.eval_frequency = train_episodes / num_evals

        # initialize control attributes
        self.model.last_eval = None
        self.model.next_eval = 0
        self.model.role_id = 0 if play_first else 1

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _training_callback(self, _locals=None, _globals=None):
        episodes_so_far = self.model.num_timesteps // 30

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

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        try:
            # train the model
            self.model.learn(total_timesteps=30 * self.train_episodes,
                             callback=self._training_callback)
        except KeyboardInterrupt:
            pass

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback()

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()


class SelfPlay(TrainingSession):
    def __init__(self, env_builder, eval_env_builder, model_builder,
                 train_episodes, eval_episodes, num_evals,
                 num_switches, params, path, seed, num_envs=1):
        super(SelfPlay, self).__init__(params, path, seed)

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env = []

        for i in range(num_envs):
            # no overlap between episodes at each process
            current_seed = seed + (train_episodes // num_envs) * i

            # create one env per process
            env.append(env_builder(seed=current_seed, play_first=True))

        # wrap envs in a vectorized env
        self.env = DummyVecEnv([lambda: e for e in env])

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        self.evaluator: Evaluator = Evaluator(eval_env_builder, eval_episodes // 2,
                                               seed + train_episodes, num_envs)

        # build the models
        self.logger.debug("Building the models...")
        self.model = model_builder(self.env, seed, **params)
        self.model.adversary = model_builder(self.env, seed, **params)

        # initialize parameters of adversary models accordingly
        self.model.adversary.load_parameters(self.model.get_parameters(), exact_match=True)

        # set adversary models as adversary policies of the self-play envs
        def make_adversary_policy(model, env):
            def adversary_policy(obs):
                zero_completed_obs = np.zeros((num_envs,) + env.observation_space.shape)
                zero_completed_obs[0, :] = obs

                actions, _ = model.adversary.predict(zero_completed_obs)

                return actions[0]

            return adversary_policy

        self.env.set_attr('adversary_policy',
                           make_adversary_policy(self.model, self.env))

        # create necessary folders
        os.makedirs(path, exist_ok=True)

        # set tensorflow log dirs
        self.model.tensorflow_log = path

        # save parameters
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.num_switches = num_switches
        self.eval_frequency = train_episodes / num_evals

        # initialize control attributes
        self.model.last_eval, self.model.next_eval = None, 0

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _training_callback(self, _locals=None, _globals=None):
        model = _locals['self']
        episodes_so_far = model.num_timesteps // 30

        turns = model.env.get_attr('turn')
        playing_first = model.env.get_attr('play_first')

        for i in range(model.env.num_envs):
            if turns[i] in range(0, model.env.num_envs):
                model.env.set_attr('play_first', not playing_first[i], indices=[i])

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model_path = self.path + f'/{episodes_so_far}'
            model.save(model_path)
            self.logger.debug(f"Saved model at {model_path}.zip.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            self.evaluator.seed = self.seed + self.train_episodes
            mean_reward, ep_length, act_hist = \
                self.evaluator.run(RLDraftAgent(model), play_first=True)

            self.evaluator.seed += self.eval_episodes
            mean_reward2, ep_length2, act_hist2 = \
                self.evaluator.run(RLDraftAgent(model), play_first=False)

            mean_reward = (mean_reward + mean_reward2) / 2
            ep_length = (ep_length + ep_length2) / 2
            act_hist = [(act_hist[i] + act_hist2[i]) / 2 for i in range(3)]

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
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        if training_is_finished:
            self.logger.debug(f"Training ended at {episodes_so_far} episodes")

        return not training_is_finished

    def _train(self):
        # save and evaluate starting models
        self._training_callback({'self': self.model})

        try:
            episodes_per_switch = self.train_episodes // self.num_switches
            self.logger.debug(f"Training will switch models every "
                              f"{episodes_per_switch} episodes")

            for _ in range(self.num_switches):
                # train the model
                self.model.learn(total_timesteps=30 * episodes_per_switch,
                                 reset_num_timesteps=False,
                                 callback=self._training_callback)
                self.logger.debug(f"Model trained for "
                                  f"{self.model.num_timesteps // 30} episodes. ")

                # update parameters of adversary models
                self.model.adversary.load_parameters(self.model.get_parameters(),
                                                     exact_match=True)
                self.logger.debug("Parameters of adversary network updated.")
        except KeyboardInterrupt:
            pass

        # save and evaluate final models, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback({'self': self.model})

        if len(self.win_rates) < self.num_evals:
            self._training_callback({'self': self.model})

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()


class AsymmetricSelfPlay(TrainingSession):
    def __init__(self, env_builder, eval_env_builder, model_builder,
                 train_episodes, eval_episodes, num_evals,
                 num_switches, params, path, seed, num_envs=1):
        super(AsymmetricSelfPlay, self).__init__(params, path, seed)

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env1, env2 = [], []

        for i in range(num_envs):
            # no overlap between episodes at each process
            current_seed = seed + (train_episodes // num_envs) * i

            # create one env per process
            env1.append(env_builder(seed=current_seed, play_first=True))
            env2.append(env_builder(seed=current_seed, play_first=False))

        # wrap envs in a vectorized env
        self.env1 = DummyVecEnv([lambda: e for e in env1])
        self.env2 = DummyVecEnv([lambda: e for e in env2])

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        self.evaluator: Evaluator = Evaluator(eval_env_builder, eval_episodes,
                                              seed + train_episodes, num_envs)

        # build the models
        self.logger.debug("Building the models...")
        self.model1 = model_builder(self.env1, seed, **params)
        self.model1.adversary = model_builder(self.env2, seed, **params)
        self.model2 = model_builder(self.env2, seed, **params)
        self.model2.adversary = model_builder(self.env1, seed, **params)

        # initialize parameters of adversary models accordingly
        self.model1.adversary.load_parameters(self.model2.get_parameters(), exact_match=True)
        self.model2.adversary.load_parameters(self.model1.get_parameters(), exact_match=True)

        # set adversary models as adversary policies of the self-play envs
        def make_adversary_policy(model, env):
            def adversary_policy(obs):
                zero_completed_obs = np.zeros((num_envs,) + env.observation_space.shape)
                zero_completed_obs[0, :] = obs

                actions, _ = model.adversary.predict(zero_completed_obs)

                return actions[0]

            return adversary_policy

        self.env1.set_attr('adversary_policy',
                           make_adversary_policy(self.model1, self.env1))
        self.env2.set_attr('adversary_policy',
                           make_adversary_policy(self.model2, self.env2))

        # create necessary folders
        os.makedirs(path + '/role0', exist_ok=True)
        os.makedirs(path + '/role1', exist_ok=True)

        # set tensorflow log dirs
        self.model1.tensorflow_log = path + '/role0'
        self.model2.tensorflow_log = path + '/role1'

        # save parameters
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.num_switches = num_switches
        self.eval_frequency = train_episodes / num_evals

        # initialize control attributes
        self.model1.last_eval, self.model1.next_eval = None, 0
        self.model2.last_eval, self.model2.next_eval = None, 0
        self.model1.role_id, self.model2.role_id = 0, 1

        # initialize results
        self.checkpoints = [], []
        self.win_rates = [], []
        self.episode_lengths = [], []
        self.action_histograms = [], []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _training_callback(self, _locals=None, _globals=None):
        model = _locals['self']
        episodes_so_far = model.num_timesteps // 30

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model_path = self.path + f'/{episodes_so_far}'
            model.save(model_path)
            self.logger.debug(f"Saved model at {model_path}.zip.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            mean_reward, ep_length, act_hist = \
                self.evaluator.run(RLDraftAgent(model),
                                   play_first=model.role_id == 0)

            end_time = time.perf_counter()
            self.logger.info(f"Finished evaluating "
                             f"({round(end_time - start_time, 3)}s). "
                             f"Avg. reward: {mean_reward}")

            # save the results
            self.checkpoints[model.role_id].append(episodes_so_far)
            self.win_rates[model.role_id].append((mean_reward + 1) / 2)
            self.episode_lengths[model.role_id].append(ep_length)
            self.action_histograms[model.role_id].append(act_hist)

            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        if training_is_finished:
            self.logger.debug(f"Training ended at {episodes_so_far} episodes")

        return not training_is_finished

    def _train(self):
        # save and evaluate starting models
        self._training_callback({'self': self.model1})
        self._training_callback({'self': self.model2})

        try:
            episodes_per_switch = self.train_episodes // self.num_switches
            self.logger.debug(f"Training will switch models every "
                              f"{episodes_per_switch} episodes")

            for _ in range(self.num_switches):
                # train the first player model
                self.model1.learn(total_timesteps=30 * episodes_per_switch,
                                  reset_num_timesteps=False,
                                  callback=self._training_callback)
                self.logger.debug(f"Model {self.model1.role_id} trained for "
                                  f"{self.model1.num_timesteps // 30} episodes. "
                                  f"Switching to model {self.model2.role_id}.")

                # train the second player model
                self.model2.learn(total_timesteps=30 * episodes_per_switch,
                                  reset_num_timesteps=False,
                                  callback=self._training_callback)
                self.logger.debug(f"Model {self.model2.role_id} trained for "
                                  f"{self.model2.num_timesteps // 30} episodes. "
                                  f"Switching to model {self.model1.role_id}.")

                # update parameters of adversary models
                self.model1.adversary.load_parameters(self.model2.get_parameters(),
                                                      exact_match=True)
                self.model2.adversary.load_parameters(self.model1.get_parameters(),
                                                      exact_match=True)
                self.logger.debug("Parameters of adversary networks updated.")
        except KeyboardInterrupt:
            pass

        # save and evaluate final models, if not done yet
        if len(self.win_rates[0]) < self.num_evals:
            self._training_callback({'self': self.model1})

        if len(self.win_rates[1]) < self.num_evals:
            self._training_callback({'self': self.model1})

        # close the envs
        for e in (self.env1, self.env2, self.evaluator):
            e.close()


class Evaluator:
    def __init__(self, env_builder, episodes, seed, num_envs):
        # log start time
        start_time = time.perf_counter()

        # initialize logger
        self.logger = logging.getLogger('{0}.{1}'.format(__name__, type(self).__name__))

        # initialize parallel environments
        self.logger.debug("Initializing envs...")
        self.env = [lambda: env_builder() for _ in range(num_envs)]
        self.env: VecEnv = DummyVecEnv(self.env)

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
        :param play_first: Whether the agent will be playing first.
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
                       for reward in rewards[:-1]]
        all_lengths = [length for lengths in episode_lengths
                       for length in lengths[:-1]]

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
        battle_agents = (MaxAttackBattleAgent(), MaxAttackBattleAgent())

        return LOCMDraftSelfPlayEnv(seed=seed, play_first=play_first,
                                    battle_agents=battle_agents,
                                    use_draft_history=False, use_mana_curve=False)

    def build_eval_env(seed=None, play_first=True):
        adversary_draft_agent = MaxAttackDraftAgent()
        battle_agents = (MaxAttackBattleAgent(), MaxAttackBattleAgent())

        return LOCMDraftSingleEnv(seed=seed, draft_agent=adversary_draft_agent,
                                  battle_agents=battle_agents, play_first=play_first,
                                  use_draft_history=False, use_mana_curve=False)

    def build_model(env, seed, neurons, layers, activation, n_steps, nminibatches,
                    noptepochs, cliprange, vf_coef, ent_coef, learning_rate):
        net_arch = [neurons] * layers
        activation = dict(tanh=tf.nn.tanh, relu=tf.nn.relu, elu=tf.nn.elu)[activation]

        return PPO2(MlpPolicy, env, verbose=0, gamma=1, seed=seed,
                    policy_kwargs=dict(net_arch=net_arch, act_fun=activation),
                    n_steps=n_steps, nminibatches=nminibatches,
                    noptepochs=noptepochs, cliprange=cliprange,
                    vf_coef=vf_coef, ent_coef=ent_coef,
                    learning_rate=learning_rate, tensorboard_log='models/trashcan/trash03/tf_log/',
                    n_cpu_tf_sess=env.num_envs)
        '''net_arch = ['lstm'] + [neurons] * layers
        activation = dict(tanh=tf.nn.tanh, relu=tf.nn.relu, elu=tf.nn.elu)[activation]

        return PPO2(MlpLstmPolicy, env, verbose=0, gamma=1, seed=seed,
                    policy_kwargs=dict(net_arch=net_arch, n_lstm=neurons, act_fun=activation),
                    n_steps=n_steps, nminibatches=nminibatches,
                    noptepochs=noptepochs, cliprange=cliprange,
                    vf_coef=vf_coef, ent_coef=ent_coef,
                    learning_rate=learning_rate, tensorboard_log=None,
                    n_cpu_tf_sess=env.num_envs)'''


    params = {'layers': 1, 'neurons': 29, 'n_steps': 30, 'nminibatches': 1,#30,
              'noptepochs': 19, 'cliprange': 0.1, 'vf_coef': 1.0,
              'ent_coef': 0.00781891437626065, 'learning_rate': 0.0001488768154153614,
              'activation': 'tanh'}

    ts = AsymmetricSelfPlay(build_env, build_eval_env, build_model, 3000, 300,
                            12, 10, params, 'models/trashcan/trash03', 36987,
                            num_envs=4)

    ts.run()
