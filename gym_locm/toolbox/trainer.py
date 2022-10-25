import json
import logging
import math
import os
import time
import warnings
from abc import abstractmethod
from datetime import datetime
from statistics import mean
from typing import Callable

# suppress tensorflow deprecated warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

import numpy as np
import tensorflow as tf
import torch as th
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback as BaseCallback3,
    CallbackList as CallbackList3,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as VecEnv3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv as DummyVecEnv3
from stable_baselines.common.callbacks import BaseCallback, CallbackList
from stable_baselines.common.policies import MlpLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env.base_vec_env import VecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ppo2.ppo2 import PPO2
from wandb.integration.sb3 import WandbCallback

from gym_locm.agents import (
    Agent,
    MaxAttackBattleAgent,
    MaxAttackDraftAgent,
    RLBattleAgent,
    RLDraftAgent,
)
from gym_locm.envs import LOCMBattleSingleEnv, LOCMDraftSingleEnv
from gym_locm.envs.battle import LOCMBattleSelfPlayEnv
from gym_locm.envs.draft import LOCMDraftSelfPlayEnv

verbose = True
REALLY_BIG_INT = 1_000_000_000

if verbose:
    logging.basicConfig(level=logging.DEBUG)

tf.get_logger().setLevel("INFO")
tf.get_logger().setLevel(logging.ERROR)


class TrainingSession:
    def __init__(self, task, params, path, seed, wandb_run=None):
        # initialize logger
        self.logger = logging.getLogger("{0}.{1}".format(__name__, type(self).__name__))

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []
        self.start_time, self.end_time = None, None
        self.wandb_run = wandb_run

        # save parameters
        self.task = task
        self.params = params
        self.path = os.path.dirname(__file__) + "/../../" + path
        self.seed = seed

    @abstractmethod
    def _train(self):
        pass

    def _save_results(self):
        results_path = self.path + "/results.json"

        with open(results_path, "w") as file:
            info = dict(
                task=self.task,
                **self.params,
                seed=self.seed,
                checkpoints=self.checkpoints,
                win_rates=self.win_rates,
                ep_lengths=self.episode_lengths,
                action_histograms=self.action_histograms,
                start_time=str(self.start_time),
                end_time=str(self.end_time),
            )
            info = json.dumps(info, indent=2)

            file.write(info)

        self.logger.debug(f"Results saved at {results_path}.")

    def run(self):
        # log start time
        self.start_time = datetime.now()
        self.logger.info(f"Training a {self.task} agent...")

        # do the training
        self._train()

        # log end time
        self.end_time = datetime.now()
        self.logger.info(
            f"End of training. Time elapsed: {self.end_time - self.start_time}."
        )

        # save model info to results file
        self._save_results()


class FixedAdversary(TrainingSession):
    def __init__(
        self,
        task,
        model_builder,
        model_params,
        env_params,
        eval_env_params,
        train_episodes,
        eval_episodes,
        num_evals,
        play_first,
        path,
        seed,
        num_envs=1,
        wandb_run=None,
    ):
        super(FixedAdversary, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run
        )

        # log start time
        start_time = time.perf_counter()

        # initialize parallel environments
        self.logger.debug("Initializing training env...")
        env = []

        if task == "battle":
            env_class = LOCMBattleSingleEnv
        else:
            env_class = LOCMDraftSingleEnv

        for i in range(num_envs):
            # no overlap between episodes at each concurrent env
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create the env
            env.append(
                lambda: env_class(
                    seed=current_seed, play_first=play_first, **env_params
                )
            )

        # wrap envs in a vectorized env
        if task == "battle":
            self.env: VecEnv3 = DummyVecEnv3(env)
        else:
            self.env: VecEnv = DummyVecEnv(env)

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluator: Evaluator = Evaluator(
            task, eval_env_params, eval_episodes, eval_seed, num_envs
        )

        # build the model
        self.logger.debug("Building the model...")
        self.model = model_builder(self.env, seed, **model_params)

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dir
        self.model.tensorflow_log = self.path

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

        self.logger.debug(
            "Finished initializing training session "
            f"({round(end_time - start_time, ndigits=3)}s)."
        )

    def _training_callback(self, _locals=None, _globals=None):
        episodes_so_far = sum(self.env.get_attr("episodes"))

        # if it is time to evaluate, do so
        if episodes_so_far >= self.model.next_eval:
            # save model
            model_path = self.path + f"/{episodes_so_far}"
            self.model.save(model_path)
            save_model_as_json(self.model, self.params["activation"], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            if self.task == "battle":
                agent_class = RLBattleAgent
            else:
                agent_class = RLDraftAgent

            agent = agent_class(self.model)

            mean_reward, ep_length, act_hist = self.evaluator.run(
                agent, play_first=self.model.role_id == 0
            )

            end_time = time.perf_counter()
            self.logger.info(
                f"Finished evaluating "
                f"({round(end_time - start_time, 3)}s). "
                f"Avg. reward: {mean_reward}"
            )

            # save the results
            self.checkpoints.append(episodes_so_far)
            win_rate = (mean_reward + 1) / 2
            self.win_rates.append(win_rate)
            self.episode_lengths.append(ep_length)
            self.action_histograms.append(act_hist)

            # update control attributes
            self.model.last_eval = episodes_so_far
            self.model.next_eval += self.eval_frequency

            # write partial results to file
            self._save_results()

            # upload stats to wandb, if enabled
            if self.wandb_run:
                info = dict(
                    checkpoint=episodes_so_far,
                    mean_reward=mean_reward,
                    win_rate=win_rate,
                    mean_ep_length=ep_length,
                )

                if self.task == "battle":
                    info["pass_actions"] = act_hist[0]
                    info["summon_actions"] = sum(act_hist[1:17])

                    if self.env.get_attr("items", indices=[0])[0]:
                        info["use_actions"] = sum(act_hist[17:121])
                        info["attack_actions"] = sum(act_hist[121:])
                    else:
                        info["attack_actions"] = sum(act_hist[17:])

                self.wandb_run.log(info)

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        if training_is_finished:
            self.logger.debug(f"Training ended at {episodes_so_far} episodes")

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callback = _build_callback(
            self.task, self._training_callback, bool(self.wandb_run)
        )

        try:
            # train the model
            # note: dynamic learning or clip rates will require accurate # of timesteps
            self.model.learn(
                total_timesteps=REALLY_BIG_INT,  # we'll stop manually
                callback=callback,
            )
        except KeyboardInterrupt:
            pass

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback()

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()


class SelfPlay(TrainingSession):
    def __init__(
        self,
        task,
        model_builder,
        model_params,
        env_params,
        eval_env_params,
        train_episodes,
        eval_episodes,
        num_evals,
        switch_frequency,
        path,
        seed,
        num_envs=1,
        wandb_run=None,
    ):
        super(SelfPlay, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run
        )

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env = []

        if task == "battle":
            env_class = LOCMBattleSelfPlayEnv
        else:
            env_class = LOCMDraftSelfPlayEnv

        for i in range(num_envs):
            # no overlap between episodes at each process
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create one env per process
            env.append(
                lambda: env_class(seed=current_seed, play_first=True, **env_params)
            )

        # wrap envs in a vectorized env
        if task == "battle":
            self.env: VecEnv3 = DummyVecEnv3(env)
        else:
            self.env: VecEnv = DummyVecEnv(env)

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluator: Evaluator = Evaluator(
            task, eval_env_params, eval_episodes // 2, eval_seed, num_envs
        )

        # build the models
        self.logger.debug("Building the models...")
        self.model = model_builder(self.env, seed, **model_params)
        self.model.adversary = model_builder(self.env, seed, **model_params)

        # initialize parameters of adversary models accordingly
        try:
            self.model.adversary.load_parameters(
                self.model.get_parameters(), exact_match=True
            )
        except AttributeError:
            self.model.adversary.set_parameters(
                self.model.get_parameters(), exact_match=True
            )

        # set adversary models as adversary policies of the self-play envs
        if self.task == "battle":

            def make_adversary_policy(model, env):
                def adversary_policy(obs):
                    actions, _ = model.adversary.predict(
                        obs,
                        deterministic=True,
                        action_masks=env.env_method("action_masks")[0],
                    )

                    return actions

                return adversary_policy

        else:

            def make_adversary_policy(model, env):
                def adversary_policy(obs):
                    zero_completed_obs = np.zeros(
                        (num_envs,) + env.observation_space.shape
                    )
                    zero_completed_obs[0, :] = obs

                    actions, _ = model.adversary.predict(zero_completed_obs)

                    return actions[0]

                return adversary_policy

        self.env.set_attr(
            "adversary_policy", make_adversary_policy(self.model, self.env)
        )

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dirs
        self.model.tensorflow_log = self.path

        # save parameters
        self.task = task
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.switch_frequency = switch_frequency
        self.eval_frequency = train_episodes / num_evals
        self.num_switches = math.ceil(train_episodes / switch_frequency)

        # initialize control attributes
        self.model.last_eval, self.model.next_eval = None, 0
        self.model.last_switch, self.model.next_switch = None, self.switch_frequency

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug(
            "Finished initializing training session "
            f"({round(end_time - start_time, ndigits=3)}s)."
        )

    def _training_callback(self, _locals=None, _globals=None):
        model = self.model
        episodes_so_far = sum(self.env.get_attr("episodes"))

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model_path = self.path + f"/{episodes_so_far}"

            if self.task == "battle":
                model.save(model_path, exclude=["adversary"])
            else:
                model.save(model_path)

            save_model_as_json(model, self.params["activation"], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            if self.task == "battle":
                agent_class = RLBattleAgent
            else:
                agent_class = RLDraftAgent

            if self.evaluator.seed is not None:
                self.evaluator.seed = self.seed + self.train_episodes

            mean_reward, ep_length, act_hist = self.evaluator.run(
                agent_class(model), play_first=True
            )

            if self.evaluator.seed is not None:
                self.evaluator.seed += self.eval_episodes

            mean_reward2, ep_length2, act_hist2 = self.evaluator.run(
                agent_class(model), play_first=False
            )

            mean_reward = (mean_reward + mean_reward2) / 2
            ep_length = (ep_length + ep_length2) / 2
            act_hist = [
                (act_hist[i] + act_hist2[i]) / 2
                for i in range(model.env.get_attr("action_space", indices=[0])[0].n)
            ]

            end_time = time.perf_counter()
            self.logger.info(
                f"Finished evaluating "
                f"({round(end_time - start_time, 3)}s). "
                f"Avg. reward: {mean_reward}"
            )

            # save the results
            self.checkpoints.append(episodes_so_far)
            win_rate = (mean_reward + 1) / 2
            self.win_rates.append(win_rate)
            self.episode_lengths.append(ep_length)
            self.action_histograms.append(act_hist)

            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

            # write partial results to file
            self._save_results()

            # upload stats to wandb, if enabled
            if self.wandb_run:
                info = dict(
                    checkpoint=episodes_so_far,
                    mean_reward=mean_reward,
                    win_rate=win_rate,
                    mean_ep_length=ep_length,
                )

                if self.task == "battle":
                    info["pass_actions"] = act_hist[0]
                    info["summon_actions"] = sum(act_hist[1:17])

                if self.env.get_attr("items", indices=[0])[0]:
                    info["use_actions"] = sum(act_hist[17:121])
                    info["attack_actions"] = sum(act_hist[121:])
                else:
                    info["attack_actions"] = sum(act_hist[17:])

                self.wandb_run.log(info)

        # if it is time to update the adversary model, do so
        if episodes_so_far >= model.next_switch:
            model.last_switch = episodes_so_far
            model.next_switch += self.switch_frequency

            # log training win rate at the time of the switch
            train_mean_reward = np.mean(
                [
                    np.mean(rewards)
                    for rewards in model.env.env_method("get_episode_rewards")
                ]
            )
            if self.wandb_run:
                self.wandb_run.log({"train_mean_reward": train_mean_reward})

            self.logger.debug(
                f"Model trained for "
                f"{sum(model.env.get_attr('episodes'))} episodes. "
                f"Train reward: {train_mean_reward}"
            )

            # reset training env rewards
            for i in range(model.env.num_envs):
                model.env.set_attr("rewards", [0.0], indices=[i])

            # update parameters of adversary models
            try:
                model.adversary.load_parameters(
                    model.get_parameters(), exact_match=True
                )
            except AttributeError:
                model.adversary.set_parameters(model.get_parameters(), exact_match=True)

            self.logger.debug("Parameters of adversary network updated.")

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting models
        self._training_callback({"self": self.model})

        callback = _build_callback(
            self.task, self._training_callback, bool(self.wandb_run)
        )

        try:
            self.logger.debug(
                f"Training will switch models every "
                f"{self.switch_frequency} episodes"
            )

            # train the model
            self.model.learn(
                total_timesteps=REALLY_BIG_INT,
                reset_num_timesteps=False,
                callback=callback,
            )

        except KeyboardInterrupt:
            pass

        self.logger.debug(
            f"Training ended at {sum(self.env.get_attr('episodes'))} " f"episodes"
        )

        # save and evaluate final models, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback({"self": self.model})

        if len(self.win_rates) < self.num_evals:
            self._training_callback({"self": self.model})

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()


class AsymmetricSelfPlay(TrainingSession):
    def __init__(
        self,
        task,
        model_builder,
        model_params,
        env_params,
        eval_env_params,
        train_episodes,
        eval_episodes,
        num_evals,
        switch_frequency,
        path,
        seed,
        num_envs=1,
        wandb_run=None,
    ):
        super(AsymmetricSelfPlay, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run
        )

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env1, env2 = [], []

        if task == "battle":
            env_class = LOCMBattleSelfPlayEnv
        else:
            env_class = LOCMDraftSelfPlayEnv

        for i in range(num_envs):
            # no overlap between episodes at each process
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create one env per process
            env1.append(
                lambda: env_class(seed=current_seed, play_first=True, **env_params)
            )
            env2.append(
                lambda: env_class(seed=current_seed, play_first=False, **env_params)
            )

        # wrap envs in a vectorized env
        if task == "battle":
            self.env1: VecEnv3 = DummyVecEnv3(env1)
            self.env2: VecEnv3 = DummyVecEnv3(env2)
        else:
            self.env1: VecEnv = DummyVecEnv(env1)
            self.env2: VecEnv = DummyVecEnv(env2)

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluator: Evaluator = Evaluator(
            task, eval_env_params, eval_episodes, eval_seed, num_envs
        )

        # build the models
        self.logger.debug("Building the models...")
        self.model1 = model_builder(self.env1, seed, **model_params)
        self.model1.adversary = model_builder(self.env2, seed, **model_params)
        self.model2 = model_builder(self.env2, seed, **model_params)
        self.model2.adversary = model_builder(self.env1, seed, **model_params)

        # initialize parameters of adversary models accordingly
        try:
            self.model1.adversary.load_parameters(
                self.model2.get_parameters(), exact_match=True
            )
            self.model2.adversary.load_parameters(
                self.model1.get_parameters(), exact_match=True
            )
        except AttributeError:
            self.model1.adversary.set_parameters(
                self.model2.get_parameters(), exact_match=True
            )
            self.model2.adversary.set_parameters(
                self.model1.get_parameters(), exact_match=True
            )

        # set adversary models as adversary policies of the self-play envs
        if self.task == "battle":

            def make_adversary_policy(model, env):
                def adversary_policy(obs):
                    actions, _ = model.adversary.predict(
                        obs,
                        deterministic=True,
                        action_masks=env.env_method("action_masks")[0],
                    )

                    return actions

                return adversary_policy

        else:

            def make_adversary_policy(model, env):
                def adversary_policy(obs):
                    zero_completed_obs = np.zeros(
                        (num_envs,) + env.observation_space.shape
                    )
                    zero_completed_obs[0, :] = obs

                    actions, _ = model.adversary.predict(zero_completed_obs)

                    return actions[0]

                return adversary_policy

        self.env1.set_attr(
            "adversary_policy", make_adversary_policy(self.model1, self.env1)
        )
        self.env2.set_attr(
            "adversary_policy", make_adversary_policy(self.model2, self.env2)
        )

        # create necessary folders
        os.makedirs(self.path + "/role0", exist_ok=True)
        os.makedirs(self.path + "/role1", exist_ok=True)

        # set tensorflow log dirs
        self.model1.tensorflow_log = self.path + "/role0"
        self.model2.tensorflow_log = self.path + "/role1"

        # save parameters
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.switch_frequency = switch_frequency
        self.eval_frequency = train_episodes / num_evals
        self.num_switches = math.ceil(train_episodes / switch_frequency)

        # initialize control attributes
        self.model1.role_id, self.model2.role_id = 0, 1
        self.model1.last_eval, self.model1.next_eval = None, 0
        self.model2.last_eval, self.model2.next_eval = None, 0
        self.model1.last_switch, self.model1.next_switch = 0, self.switch_frequency
        self.model2.last_switch, self.model2.next_switch = 0, self.switch_frequency

        # initialize results
        self.checkpoints = [], []
        self.win_rates = [], []
        self.episode_lengths = [], []
        self.action_histograms = [], []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug(
            "Finished initializing training session "
            f"({round(end_time - start_time, ndigits=3)}s)."
        )

    def _training_callback(self, _locals=None, _globals=None):
        model = _locals["self"]
        episodes_so_far = sum(model.env.get_attr("episodes"))

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model_path = f"{self.path}/role{model.role_id}/{episodes_so_far}"

            if self.task == "battle":
                model.save(model_path, exclude=["adversary"])
            else:
                model.save(model_path)

            save_model_as_json(model, self.params["activation"], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(
                f"Evaluating model {model.role_id} " f"({episodes_so_far} episodes)..."
            )
            start_time = time.perf_counter()

            if self.task == "battle":
                agent_class = RLBattleAgent
            else:
                agent_class = RLDraftAgent

            mean_reward, ep_length, act_hist = self.evaluator.run(
                agent_class(model), play_first=model.role_id == 0
            )

            end_time = time.perf_counter()
            self.logger.info(
                f"Finished evaluating "
                f"({round(end_time - start_time, 3)}s). "
                f"Avg. reward: {mean_reward}"
            )

            # save the results
            self.checkpoints[model.role_id].append(episodes_so_far)
            win_rate = (mean_reward + 1) / 2
            self.win_rates[model.role_id].append(win_rate)
            self.episode_lengths[model.role_id].append(ep_length)
            self.action_histograms[model.role_id].append(act_hist)

            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

            # write partial results to file
            self._save_results()

            # upload stats to wandb, if enabled
            if self.wandb_run:
                info = {
                    f"checkpoint_{model.role_id}": episodes_so_far,
                    f"mean_reward_{model.role_id}": mean_reward,
                    f"win_rate_{model.role_id}": win_rate,
                    f"mean_ep_length_{model.role_id}": ep_length,
                }

                if self.task == "battle":
                    info[f"pass_actions_{model.role_id}"] = act_hist[0]
                    info[f"summon_actions_{model.role_id}"] = sum(act_hist[1:17])

                    if model.env.get_attr("items", indices=[0])[0]:
                        info[f"use_actions_{model.role_id}"] = sum(act_hist[17:121])
                        info[f"attack_actions_{model.role_id}"] = sum(act_hist[121:])
                    else:
                        info[f"attack_actions_{model.role_id}"] = sum(act_hist[17:])

                self.wandb_run.log(info)

        # if training should end, return False to end training
        training_is_finished = (
            episodes_so_far >= model.next_switch
            or episodes_so_far >= self.train_episodes
        )

        if training_is_finished:
            model.last_switch = episodes_so_far
            model.next_switch += self.switch_frequency

        return not training_is_finished

    def _train(self):
        # save and evaluate starting models
        self._training_callback({"self": self.model1})
        self._training_callback({"self": self.model2})

        callback1 = _build_callback(
            self.task,
            lambda: self._training_callback({"self": self.model1}),
            bool(self.wandb_run),
        )
        callback2 = _build_callback(
            self.task,
            lambda: self._training_callback({"self": self.model2}),
            bool(self.wandb_run),
        )

        try:
            self.logger.debug(
                f"Training will switch models every "
                f"{self.switch_frequency} episodes"
            )

            for _ in range(self.num_switches):
                # train the first player model
                self.model1.learn(
                    total_timesteps=REALLY_BIG_INT,
                    reset_num_timesteps=False,
                    callback=callback1,
                )

                # log training win rate at the time of the switch
                train_mean_reward1 = np.mean(
                    [
                        np.mean(rewards)
                        for rewards in self.env1.env_method("get_episode_rewards")
                    ]
                )
                if self.wandb_run:
                    self.wandb_run.log({"train_mean_reward1": train_mean_reward1})

                # reset training env rewards
                for i in range(self.env1.num_envs):
                    self.env1.set_attr("rewards", [0.0], indices=[i])

                self.logger.debug(
                    f"Model {self.model1.role_id} trained for "
                    f"{sum(self.env1.get_attr('episodes'))} episodes. "
                    f"Train reward: {train_mean_reward1}. "
                    f"Switching to model {self.model2.role_id}."
                )

                # train the second player model
                self.model2.learn(
                    total_timesteps=REALLY_BIG_INT,
                    reset_num_timesteps=False,
                    callback=callback2,
                )

                # log training win rate at the time of the switch
                train_mean_reward2 = np.mean(
                    [
                        np.mean(rewards)
                        for rewards in self.env2.env_method("get_episode_rewards")
                    ]
                )
                if self.wandb_run:
                    self.wandb_run.log({"train_mean_reward2": train_mean_reward2})

                # reset training env rewards
                for i in range(self.env2.num_envs):
                    self.env2.set_attr("rewards", [0.0], indices=[i])

                self.logger.debug(
                    f"Model {self.model2.role_id} trained for "
                    f"{sum(self.env2.get_attr('episodes'))} episodes. "
                    f"Train reward: {train_mean_reward2}. "
                    f"Switching to model {self.model1.role_id}."
                )

                # update parameters of adversary models
                try:
                    self.model1.adversary.load_parameters(
                        self.model2.get_parameters(), exact_match=True
                    )
                    self.model2.adversary.load_parameters(
                        self.model1.get_parameters(), exact_match=True
                    )
                except AttributeError:
                    self.model1.adversary.set_parameters(
                        self.model2.get_parameters(), exact_match=True
                    )
                    self.model2.adversary.set_parameters(
                        self.model1.get_parameters(), exact_match=True
                    )

                self.logger.debug("Parameters of adversary networks updated.")
        except KeyboardInterrupt:
            pass

        self.logger.debug(
            f"Training ended at {sum(self.env1.get_attr('episodes'))} " f"episodes"
        )

        # save and evaluate final models, if not done yet
        if len(self.win_rates[0]) < self.num_evals:
            self._training_callback({"self": self.model1})

        if len(self.win_rates[1]) < self.num_evals:
            self._training_callback({"self": self.model1})

        # close the envs
        for e in (self.env1, self.env2, self.evaluator):
            e.close()


class Evaluator:
    def __init__(self, task, env_params, episodes, seed, num_envs):
        # log start time
        start_time = time.perf_counter()

        # initialize logger
        self.logger = logging.getLogger("{0}.{1}".format(__name__, type(self).__name__))

        # initialize parallel environments
        self.logger.debug("Initializing envs...")

        if task == "battle":
            env_class = LOCMBattleSingleEnv
        else:
            env_class = LOCMDraftSingleEnv

        self.env = [lambda: env_class(**env_params) for _ in range(num_envs)]

        if task == "battle":
            self.env: VecEnv3 = DummyVecEnv3(self.env)
        else:
            self.env: VecEnv = DummyVecEnv(self.env)

        # save parameters
        self.episodes = episodes
        self.seed = seed

        # log end time
        end_time = time.perf_counter()

        self.logger.debug(
            "Finished initializing evaluator "
            f"({round(end_time - start_time, ndigits=3)}s)."
        )

    def run(self, agent: Agent, play_first=True):
        """
        Evaluates an agent.
        :param agent: (gym_locm.agents.Agent) Agent to be evaluated.
        :param play_first: Whether the agent will be playing first.
        :return: A tuple containing the `mean_reward`, the `mean_length`
        and the `action_histogram` of the evaluation episodes.
        """
        # set appropriate seeds
        if self.seed is not None:
            for i in range(self.env.num_envs):
                current_seed = self.seed
                current_seed += (self.episodes // self.env.num_envs) * i
                current_seed -= 1  # resetting the env increases the seed by one

                self.env.env_method("seed", current_seed, indices=[i])

        # set agent role
        self.env.set_attr("play_first", play_first)

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
            elif isinstance(agent, RLBattleAgent):
                action_masks = self.env.env_method("action_masks")
                actions = agent.act(observations, action_masks)
            else:
                observations = self.env.get_attr("state")
                actions = [agent.act(observation) for observation in observations]

            # update the action histogram
            for action in actions:
                action_histogram[action] += 1

            # perform the action and get the outcome
            observations, rewards, dones, infos = self.env.step(actions)

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
        all_rewards = [reward for rewards in episode_rewards for reward in rewards[:-1]]
        all_lengths = [length for lengths in episode_lengths for length in lengths[:-1]]

        # todo: fix -- sometimes we miss self.episodes by one
        # assert len(all_rewards) == self.episodes
        # assert len(all_lengths) == self.episodes

        # transform the action histogram in a probability distribution
        action_histogram = [
            action_freq / sum(action_histogram) for action_freq in action_histogram
        ]

        # cap any unsolicited additional episodes
        all_rewards = all_rewards[: self.episodes]
        all_lengths = all_lengths[: self.episodes]

        return mean(all_rewards), mean(all_lengths), action_histogram

    def close(self):
        self.env.close()


class BattleTrainingCallback(BaseCallback3):
    def __init__(self, callback_func, verbose=0):
        super(BattleTrainingCallback, self).__init__(verbose)
        self.callback_func = callback_func

    def _on_step(self):
        return self.callback_func()


class DraftTrainingCallback(BaseCallback):
    def __init__(self, callback_func, verbose=0):
        super(DraftTrainingCallback, self).__init__(verbose)
        self.callback_func = callback_func

    def _on_step(self):
        return self.callback_func()


def _build_callback(task: str, callback_func: Callable, include_wandb=False):
    callbacks = []
    if include_wandb:
        callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))
    if task == "battle":
        callbacks.insert(0, BattleTrainingCallback(callback_func))
        return CallbackList3(callbacks)
    else:
        callbacks.insert(0, DraftTrainingCallback(callback_func))
        return CallbackList(callbacks)


def save_model_as_json(model, act_fun, path):
    pass  # todo: reimplement this supporting stable-baselines 2 and 3


def model_builder_mlp(
    env,
    seed,
    neurons,
    layers,
    activation,
    n_steps,
    nminibatches,
    noptepochs,
    cliprange,
    vf_coef,
    ent_coef,
    learning_rate,
    gamma=1,
    tensorboard_log=None,
):
    net_arch = [neurons] * layers
    activation = dict(tanh=tf.nn.tanh, relu=tf.nn.relu, elu=tf.nn.elu)[activation]

    return PPO2(
        MlpPolicy,
        env,
        verbose=0,
        gamma=gamma,
        seed=seed,
        policy_kwargs=dict(net_arch=net_arch, act_fun=activation),
        n_steps=n_steps,
        nminibatches=nminibatches,
        noptepochs=noptepochs,
        cliprange=cliprange,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        n_cpu_tf_sess=env.num_envs,
        tensorboard_log=tensorboard_log,
    )


def model_builder_lstm(
    env,
    seed,
    neurons,
    layers,
    activation,
    n_steps,
    nminibatches,
    noptepochs,
    cliprange,
    vf_coef,
    ent_coef,
    learning_rate,
    gamma=1,
    tensorboard_log=None,
):
    net_arch = ["lstm"] + [neurons] * (layers - 1)
    activation = dict(tanh=tf.nn.tanh, relu=tf.nn.relu, elu=tf.nn.elu)[activation]

    return PPO2(
        MlpLstmPolicy,
        env,
        verbose=0,
        gamma=gamma,
        seed=seed,
        policy_kwargs=dict(net_arch=net_arch, n_lstm=neurons, act_fun=activation),
        n_steps=n_steps,
        nminibatches=nminibatches,
        noptepochs=noptepochs,
        cliprange=cliprange,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        n_cpu_tf_sess=env.num_envs,
        tensorboard_log=tensorboard_log,
    )


def model_builder_mlp_masked(
    env,
    seed,
    neurons,
    layers,
    activation,
    n_steps,
    nminibatches,
    noptepochs,
    cliprange,
    vf_coef,
    ent_coef,
    learning_rate,
    gamma=1,
    tensorboard_log=None,
):
    net_arch = [neurons] * layers
    activation = dict(tanh=th.nn.Tanh, relu=th.nn.ReLU, elu=th.nn.ELU)[activation]

    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=nminibatches,
        n_epochs=noptepochs,
        gamma=gamma,
        clip_range=cliprange,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        seed=seed,
        policy_kwargs=dict(net_arch=net_arch, activation_fn=activation),
        tensorboard_log=tensorboard_log,
    )


if __name__ == "__main__":
    env_params = {
        "battle_agents": (MaxAttackBattleAgent(), MaxAttackBattleAgent()),
        "use_draft_history": False,
        "use_mana_curve": False,
    }

    eval_env_params = {
        "draft_agent": MaxAttackDraftAgent(),
        "battle_agents": (MaxAttackBattleAgent(), MaxAttackBattleAgent()),
        "use_draft_history": False,
        "use_mana_curve": False,
    }

    model_params = {
        "layers": 1,
        "neurons": 29,
        "n_steps": 30,
        "nminibatches": 30,
        "noptepochs": 19,
        "cliprange": 0.1,
        "vf_coef": 1.0,
        "ent_coef": 0.00781891437626065,
        "activation": "tanh",
        "learning_rate": 0.0001488768154153614,
    }

    ts = FixedAdversary(
        model_builder_mlp,
        model_params,
        env_params,
        eval_env_params,
        30000,
        1000,
        12,
        True,
        "models/trashcan/trash04",
        36987,
        num_envs=4,
    )

    ts.run()
