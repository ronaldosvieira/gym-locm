from functools import partial
import json
import logging
import math
import os
import time
from typing import Callable, List, Tuple

import numpy as np
from abc import abstractmethod
from datetime import datetime
from statistics import mean

import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import (
    VecEnv as VecEnv3,
    DummyVecEnv as DummyVecEnv3,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sb3_contrib import MaskablePPO
from wandb.integration.sb3 import WandbCallback

from gymnasium.spaces import Space, Box, Dict
from gym_locm import envs
from gym_locm.agents import Agent, RLBattleAgent, RLDraftAgent
from gym_locm.envs import LOCMBattleSingleEnv
from gym_locm.envs.battle import LOCMBattleSelfPlayEnv

verbose = True
REALLY_BIG_INT = 1_000_000_000

if verbose:
    logging.basicConfig(level=logging.DEBUG)


class TrainingSession:
    def __init__(self, task, params, path, seed, wandb_run=None):
        # initialize logger
        self.logger = logging.getLogger("{0}.{1}".format(__name__, type(self).__name__))

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.battle_lengths = []
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
        role,
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

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env = []

        env_class = LOCMBattleSingleEnv

        for i in range(num_envs):
            # no overlap between episodes at each concurrent env
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create the env
            env.append(
                lambda: env_class(
                    seed=current_seed,
                    play_first=role == "first",
                    alternate_roles=role == "alternate",
                    **env_params,
                )
            )

        # wrap envs in a vectorized env
        self.env: VecEnv3 = DummyVecEnv3(env)

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluators: List[Evaluator] = [
            Evaluator(task, e, eval_episodes, eval_seed, num_envs)
            for e in eval_env_params
        ]

        # build the model
        self.logger.debug("Building the model...")
        self.model = model_builder(self.env, seed, **model_params)

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dir
        # todo: check later if this was meant to be 'tensorboard_log' instead
        self.model.tensorflow_log = self.path

        # save parameters
        self.task = task
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.eval_frequency = train_episodes / num_evals
        self.eval_adversaries = [
            repr(e["battle_agent"]) for e in eval_env_params
        ]
        self.role = role

        # initialize control attributes
        self.model.last_eval, self.model.next_eval = None, 0

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

            model.save(model_path, exclude=["adversary"])
            save_model_as_json(model, self.params["activation"], model_path)

            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            agent_class = RLBattleAgent
            agent = agent_class(model, deterministic=True)

            for evaluator, eval_adversary in zip(
                self.evaluators, self.eval_adversaries
            ):
                if evaluator.seed is not None:
                    evaluator.seed = self.seed + self.train_episodes

                (
                    win_rate,
                    mean_reward,
                    ep_length,
                    battle_length,
                    act_hist,
                ) = evaluator.run(
                    agent,
                    play_first=self.role == "first",
                    alternate_roles=self.role == "alternate",
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"Avg. reward: {mean_reward}"
                )

                # save the results
                self.checkpoints.append(episodes_so_far)
                self.win_rates.append(win_rate)
                self.episode_lengths.append(ep_length)
                self.battle_lengths.append(battle_length)
                self.action_histograms.append(act_hist)

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = dict()

                    info["checkpoint"] = episodes_so_far
                    info[panel_name + "/mean_reward"] = mean_reward
                    info[panel_name + "/win_rate"] = win_rate
                    info[panel_name + "/mean_ep_length"] = ep_length
                    info[panel_name + "/mean_battle_length"] = battle_length

                    info[panel_name + "/pass_actions"] = act_hist[0]
                    info[panel_name + "/summon_actions"] = sum(act_hist[1:17])

                    if self.env.get_attr("items", indices=[0])[0]:
                        info[panel_name + "/use_actions"] = sum(act_hist[17:121])
                        info[panel_name + "/attack_actions"] = sum(act_hist[121:])
                    else:
                        info[panel_name + "/attack_actions"] = sum(act_hist[17:])

                    self.wandb_run.log(info)

            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callbacks = [
            TrainingCallback(self._training_callback),
            RolloutEndLogger(self.wandb_run)
        ]

        if self.wandb_run:
            callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

        try:
            # train the model
            self.model.learn(
                total_timesteps=REALLY_BIG_INT,
                reset_num_timesteps=False,
                callback=CallbackList(callbacks),
            )

        except KeyboardInterrupt:
            pass

        self.logger.debug(
            f"Training ended at {sum(self.env.get_attr('episodes'))} " f"episodes"
        )

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback()

        # close all envs
        self.env.close()

        for e in self.evaluators:
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
        role,
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

        env_class = LOCMBattleSelfPlayEnv

        for i in range(num_envs):
            # no overlap between episodes at each concurrent env
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create the env
            env.append(
                lambda: env_class(
                    seed=current_seed,
                    play_first=role == "first",
                    alternate_roles=role == "alternate",
                    **env_params,
                )
            )

        # wrap envs in a vectorized env
        self.env: VecEnv3 = DummyVecEnv3(env)

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluators: List[Evaluator] = [
            Evaluator(task, e, eval_episodes, eval_seed, num_envs)
            for e in eval_env_params
        ]

        # build the model
        self.logger.debug("Building the model...")
        self.model = model_builder(self.env, seed, **model_params)
        self.model.adversary = model_builder(self.env, seed, **model_params)

        # initialize parameters of adversary models accordingly
        self.model.adversary.set_parameters(
            self.model.get_parameters(), exact_match=True
        )

        # set adversary models as adversary policies of the self-play envs
        def make_adversary_policy(model):
            def adversary_policy(obs, action_mask):
                actions, _ = model.adversary.predict(obs, action_masks=action_mask)

                return actions

            return adversary_policy

        self.env.set_attr("adversary_policy", make_adversary_policy(self.model))

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dir
        self.model.tensorflow_log = self.path

        # save parameters
        self.task = task
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.eval_frequency = train_episodes / num_evals
        self.switch_frequency = switch_frequency
        self.num_switches = math.ceil(train_episodes / switch_frequency)
        self.eval_adversaries = [
            repr(e["battle_agent"]) for e in eval_env_params
        ]
        self.role = role

        # initialize control attributes
        self.model.last_eval, self.model.next_eval = None, 0
        self.model.last_switch, self.model.next_switch = None, self.switch_frequency

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

            model.save(model_path, exclude=["adversary"])
            save_model_as_json(model, self.params["activation"], model_path)

            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            agent_class = RLBattleAgent
            agent = agent_class(model, deterministic=True)

            for evaluator, eval_adversary in zip(
                self.evaluators, self.eval_adversaries
            ):
                if evaluator.seed is not None:
                    evaluator.seed = self.seed + self.train_episodes

                (
                    win_rate,
                    mean_reward,
                    ep_length,
                    battle_length,
                    act_hist,
                ) = evaluator.run(
                    agent,
                    play_first=self.role == "first",
                    alternate_roles=self.role == "alternate",
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"Avg. reward: {mean_reward}"
                )

                # save the results
                self.checkpoints.append(episodes_so_far)
                self.win_rates.append(win_rate)
                self.episode_lengths.append(ep_length)
                self.battle_lengths.append(battle_length)
                self.action_histograms.append(act_hist)

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = dict()

                    info["checkpoint"] = episodes_so_far
                    info[panel_name + "/mean_reward"] = mean_reward
                    info[panel_name + "/win_rate"] = win_rate
                    info[panel_name + "/mean_ep_length"] = ep_length
                    info[panel_name + "/mean_battle_length"] = battle_length

                    info[panel_name + "/pass_actions"] = act_hist[0]
                    info[panel_name + "/summon_actions"] = sum(act_hist[1:17])

                    if self.env.get_attr("items", indices=[0])[0]:
                        info[panel_name + "/use_actions"] = sum(act_hist[17:121])
                        info[panel_name + "/attack_actions"] = sum(act_hist[121:])
                    else:
                        info[panel_name + "/attack_actions"] = sum(act_hist[17:])

                    self.wandb_run.log(info)
            
            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

        # if it is time to update the adversary model, do so
        if episodes_so_far >= model.next_switch:
            model.last_switch = episodes_so_far
            model.next_switch += self.switch_frequency

            # update parameters of adversary models
            model.adversary.set_parameters(model.get_parameters(), exact_match=True)

            self.logger.debug("Parameters of adversary network updated.")

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callbacks = [
            TrainingCallback(self._training_callback),
            RolloutEndLogger(self.wandb_run)
        ]

        if self.wandb_run:
            callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

        try:
            # train the model
            self.model.learn(
                total_timesteps=REALLY_BIG_INT,
                reset_num_timesteps=False,
                callback=CallbackList(callbacks),
            )

        except KeyboardInterrupt:
            pass

        self.logger.debug(
            f"Training ended at {sum(self.env.get_attr('episodes'))} " f"episodes"
        )

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback()

        # close all envs
        self.env.close()

        for e in self.evaluators:
            e.close()


class FixedAndSelfPlayHybrid(TrainingSession):
    def __init__(
        self,
        task,
        model_builder,
        model_params,
        self_play_env_params,
        fixed_adversary_env_params,
        eval_env_params,
        train_episodes,
        eval_episodes,
        num_evals,
        role,
        switch_frequency,
        path,
        seed,
        num_self_play_envs=1,
        num_fixed_adversary_envs=1,
        wandb_run=None,
    ):
        super(FixedAndSelfPlayHybrid, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run
        )

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env = []

        num_envs = num_self_play_envs + num_fixed_adversary_envs

        for i in range(num_envs):
            # no overlap between episodes at each concurrent env
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            if i < num_self_play_envs:
                env.append(
                    lambda: LOCMBattleSelfPlayEnv(
                        seed=current_seed,
                        play_first=role == "first",
                        alternate_roles=role == "alternate",
                        **self_play_env_params,
                    )
                )
            else:
                env.append(
                    lambda: LOCMBattleSingleEnv(
                        seed=current_seed,
                        play_first=role == "first",
                        alternate_roles=role == "alternate",
                        **fixed_adversary_env_params,
                    )
                )

        # wrap envs in a vectorized env
        self.env: VecEnv3 = DummyVecEnv3(env)

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluators: List[Evaluator] = [
            Evaluator(task, e, eval_episodes, eval_seed, num_envs)
            for e in eval_env_params
        ]

        # build the model
        self.logger.debug("Building the model...")
        self.model = model_builder(self.env, seed, **model_params)
        self.model.adversary = model_builder(self.env, seed, **model_params)

        # initialize parameters of the adversary model accordingly
        self.model.adversary.set_parameters(
            self.model.get_parameters(), exact_match=True
        )

        # set the adversary model as an adversary policy in the self-play envs
        def make_adversary_policy(model):
            def adversary_policy(obs, action_mask):
                actions, _ = model.adversary.predict(obs, action_masks=action_mask)

                return actions

            return adversary_policy

        self.env.set_attr("adversary_policy", make_adversary_policy(self.model))

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dir
        self.model.tensorflow_log = self.path

        # save parameters
        self.task = task
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.eval_frequency = train_episodes / num_evals
        self.switch_frequency = switch_frequency
        self.num_switches = math.ceil(train_episodes / switch_frequency)
        self.eval_adversaries = [
            repr(e["battle_agent"]) for e in eval_env_params
        ]
        self.role = role

        # initialize control attributes
        self.model.last_eval, self.model.next_eval = None, 0
        self.model.last_switch, self.model.next_switch = None, self.switch_frequency

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

            model.save(model_path, exclude=["adversary"])
            save_model_as_json(model, self.params["activation"], model_path)

            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            agent_class = RLBattleAgent
            agent = agent_class(model, deterministic=True)

            for evaluator, eval_adversary in zip(
                self.evaluators, self.eval_adversaries
            ):
                if evaluator.seed is not None:
                    evaluator.seed = self.seed + self.train_episodes

                (
                    win_rate,
                    mean_reward,
                    ep_length,
                    battle_length,
                    act_hist,
                ) = evaluator.run(
                    agent,
                    play_first=self.role == "first",
                    alternate_roles=self.role == "alternate",
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"Avg. reward: {mean_reward}"
                )

                # save the results
                self.checkpoints.append(episodes_so_far)
                self.win_rates.append(win_rate)
                self.episode_lengths.append(ep_length)
                self.battle_lengths.append(battle_length)
                self.action_histograms.append(act_hist)

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = dict()

                    info["checkpoint"] = episodes_so_far
                    info[panel_name + "/mean_reward"] = mean_reward
                    info[panel_name + "/win_rate"] = win_rate
                    info[panel_name + "/mean_ep_length"] = ep_length
                    info[panel_name + "/mean_battle_length"] = battle_length

                    info[panel_name + "/pass_actions"] = act_hist[0]
                    info[panel_name + "/summon_actions"] = sum(act_hist[1:17])

                    if self.env.get_attr("items", indices=[0])[0]:
                        info[panel_name + "/use_actions"] = sum(act_hist[17:121])
                        info[panel_name + "/attack_actions"] = sum(act_hist[121:])
                    else:
                        info[panel_name + "/attack_actions"] = sum(act_hist[17:])

                    self.wandb_run.log(info)
                    
            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

        # if it is time to update the adversary model, do so
        if episodes_so_far >= model.next_switch:
            model.last_switch = episodes_so_far
            model.next_switch += self.switch_frequency

            # update parameters of adversary models
            model.adversary.set_parameters(model.get_parameters(), exact_match=True)

            self.logger.debug("Parameters of adversary network updated.")

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callbacks = [
            TrainingCallback(self._training_callback),
            RolloutEndLogger(self.wandb_run)
        ]

        if self.wandb_run:
            callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

        try:
            # train the model
            self.model.learn(
                total_timesteps=REALLY_BIG_INT,
                reset_num_timesteps=False,
                callback=CallbackList(callbacks),
            )

        except KeyboardInterrupt:
            pass

        self.logger.debug(
            f"Training ended at {sum(self.env.get_attr('episodes'))} " f"episodes"
        )

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback()

        # close all envs
        self.env.close()

        for e in self.evaluators:
            e.close()


class Evaluator:
    def __init__(self, task, env_params, episodes, seed, num_envs):
        # log start time
        start_time = time.perf_counter()

        # initialize logger
        self.logger = logging.getLogger("{0}.{1}".format(__name__, type(self).__name__))

        # initialize parallel environments
        self.logger.debug("Initializing envs...")

        env_class = LOCMBattleSingleEnv

        self.env = [lambda: env_class(**env_params) for _ in range(num_envs)]

        self.env: VecEnv3 = DummyVecEnv3(self.env)

        # save parameters
        self.episodes = episodes
        self.seed = seed
        agent_name = repr(env_params["battle_agent"])

        # log end time
        end_time = time.perf_counter()

        self.logger.debug(
            f"Finished initializing evaluator ({agent_name}) "
            f"({round(end_time - start_time, ndigits=3)}s)."
        )

    def run(self, agent: Agent, play_first=True, alternate_roles=False):
        """
        Evaluates an agent.
        :param agent: (gym_locm.agents.Agent) Agent to be evaluated.
        :param play_first: Whether the agent will be playing first.
        :param alternate_roles: Whether the agent should be alternating
        between playing first and second
        :return: A tuple containing the `win_rate`, the `mean_reward`,
        the `mean_length` and the `action_histogram` of the evaluation episodes.
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
        self.env.set_attr("alternate_roles", alternate_roles)

        # reset the env
        observations = self.env.reset()

        # initialize metrics
        episodes_so_far = 0
        episode_wins = [[] for _ in range(self.env.num_envs)]
        episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
        episode_lengths = [[0] for _ in range(self.env.num_envs)]
        episode_turns = [[] for _ in range(self.env.num_envs)]
        action_histogram = [0] * self.env.action_space.n

        # run the episodes
        while True:
            # get current role info
            roles = [
                0 if play_first else 1 for play_first in self.env.get_attr("play_first")
            ]

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
            observations, rewards, dones, infos = self.env.step(
                actions
            )

            # update metrics
            for i in range(self.env.num_envs):
                episode_rewards[i][-1] += rewards[i]
                episode_lengths[i][-1] += 1

                if dones[i]:
                    episode_wins[i].append(1 if infos[i]["winner"] == roles[i] else 0)
                    episode_rewards[i].append(0.0)
                    episode_lengths[i].append(0)
                    episode_turns[i].append(infos[i]["turn"])

                    episodes_so_far += 1

            # check exiting condition
            if episodes_so_far >= self.episodes:
                break

        # join all parallel metrics
        all_rewards = [reward for rewards in episode_rewards for reward in rewards[:-1]]
        all_lengths = [length for lengths in episode_lengths for length in lengths[:-1]]
        all_turns = [turn for turns in episode_turns for turn in turns]
        all_wins = [win for wins in episode_wins for win in wins]

        # todo: fix -- sometimes we miss self.episodes by one
        # assert len(all_rewards) == self.episodes
        # assert len(all_lengths) == self.episodes
        # assert len(all_turns) == self.episodes

        # transform the action histogram in a probability distribution
        action_histogram = [
            action_freq / sum(action_histogram) for action_freq in action_histogram
        ]

        # cap any unsolicited additional episodes
        all_wins = all_wins[: self.episodes]
        all_rewards = all_rewards[: self.episodes]
        all_lengths = all_lengths[: self.episodes]
        all_turns = all_turns[: self.episodes]

        return (
            mean(all_wins),
            mean(all_rewards),
            mean(all_lengths),
            mean(all_turns),
            action_histogram,
        )

    def close(self):
        self.env.close()


class TrainingCallback(BaseCallback):
    def __init__(self, callback_func, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

        self.callback_func = callback_func

    def _on_step(self):
        return self.callback_func()


class RolloutEndLogger(BaseCallback):
    def __init__(self, wandb_run = None, verbose=0):
        # initialize logger
        self.rollout_logger = logging.getLogger("{0}.{1}".format(__name__, type(self).__name__))
        
        super(RolloutEndLogger, self).__init__(verbose)

        self.wandb_run = wandb_run

        self.episode_counter = 0

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self):
        self.log_rollout_stats()
        
        self.log_training_rewards()
        
    def log_rollout_stats(self):
        # log the number of episodes and steps per episode in the rollout that just ended
        all_episodes = sum(self.training_env.get_attr("episodes"))
        n_rollout_steps = self.locals["n_rollout_steps"]
        
        rollout_episodes = all_episodes - self.episode_counter
        self.episode_counter = all_episodes

        self.rollout_logger.debug(
            f"Rollout ended; updating policy ({rollout_episodes} episodes, "
            f"{round(n_rollout_steps / rollout_episodes, 2)} steps/episode)."
        )
        
    def log_training_rewards(self):
        # log avg. training rewards at the end of the rollout
        train_mean_reward = np.mean([
            np.mean(rewards[:-1])
            for rewards in self.training_env.env_method("get_episode_rewards")
        ])
        
        if self.wandb_run:
            log_name = "train_mean_reward"

            self.wandb_run.log({log_name: train_mean_reward})

        self.rollout_logger.debug(
            f"Model trained for "
            f"{sum(self.model.env.get_attr('episodes'))} episodes. "
            f"Train reward: {train_mean_reward}"
        )

        # reset training env rewards
        for i in range(self.training_env.num_envs):
            self.training_env.set_attr("rewards_single_player", [], indices=[i])


def save_model_as_json(model, act_fun, path):
    with open(path + ".json", "w") as json_file:
        params = {}

        # create a parameter dictionary
        for label, weights in model.get_parameters()["policy"].items():
            params[label] = weights.tolist()

        # add activation function to it
        params["act_fun"] = act_fun

        # and save into the new file
        json.dump(params, json_file)


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
    gae_lambda=0.95,
    tensorboard_log=None,
):
    if isinstance(layers, int):
        net_arch = [neurons] * layers
    elif isinstance(layers, dict) or isinstance(layers, list):
        net_arch = layers
    else:
        raise ValueError(f"Invalid type for layers: {type(layers)}.")
    
    activation = dict(tanh=th.nn.Tanh, relu=th.nn.ReLU, elu=th.nn.ELU)[activation]

    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=nminibatches,
        n_epochs=noptepochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=cliprange,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        seed=seed,
        policy_kwargs=dict(net_arch=net_arch, activation_fn=activation),
        tensorboard_log=tensorboard_log,
    )


def load_trained_mlp_masked(path):
    def loaded_model_builder(env, seed, *args, **kwargs):
        return MaskablePPO.load(path + ".zip", env=env, force_reset=True, seed=seed)
    
    return loaded_model_builder


class SharedCardEmbeddingNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, card_dim: int = 32):
        features_dim = observation_space.shape[0]
        
        super().__init__(observation_space, features_dim)
            
        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, card_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 6 player features
        # 2 deck length features
        # 2 hand length features
        # 8 * 17 card features (4 type, 1 cost, 1 attack, 1 defense, 3 ETB abilities, 6 abilities, 1 area)
        # 6 * 9 friendly creature features (1 attack, 1 defense, 1 can_attack, 6 abilities)
        # 6 * 8 enemy creature features (1 attack, 1 defense, 6 abilities)
        # 17 average deck features
        # = 265 features
        
        obs_pre_hand = observations[:, :6 + 2 + 2]
        hand = observations[:, 6 + 2 + 2 : 6 + 2 + 2 + 8 * 17]
        obs_post_hand = observations[:, 6 + 2 + 2 + 8 * 17 :]
        
        # pass each card in hand through the card embedding network
        hand = hand.reshape(-1, 17)  # reshape to (batch_size * max_hand_size, card_feature_dim)
        hand_embedding = self.card_embedding(hand)  # (batch_size * max_hand_size, card_dim)
        hand_embedding = hand_embedding.reshape(-1, 8 * 17)  # reshape back to (batch_size, max_hand_size * card_dim)

        obs = th.cat((obs_pre_hand, hand_embedding, obs_post_hand), dim=1)

        return obs


def model_builder_sce_mlp_masked(
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
    gae_lambda=0.95,
    tensorboard_log=None,
):
    if isinstance(layers, int):
        net_arch = [neurons] * layers
    elif isinstance(layers, dict) or isinstance(layers, list):
        net_arch = layers
    else:
        raise ValueError(f"Invalid type for layers: {type(layers)}.")
    
    activation = dict(tanh=th.nn.Tanh, relu=th.nn.ReLU, elu=th.nn.ELU)[activation]

    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=nminibatches,
        n_epochs=noptepochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=cliprange,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        seed=seed,
        policy_kwargs=dict(
            net_arch=net_arch, 
            activation_fn=activation,
            features_extractor_class=SharedCardEmbeddingNetwork,
            features_extractor_kwargs=dict(card_dim=17),
        ),
        tensorboard_log=tensorboard_log,
    )


class PermutationInvariantFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: Dict, 
        card_dim: int = 17, 
        player_dim: int = 5, 
        creature_dim: int = 8,
        card_emb_dim: int = 32,
        zone_emb_dim: int = 32,
        player_emb_dim: int = 16,
        creature_emb_dim: int = 16,
        lane_emb_dim: int = 16,
        state_emb_dim: int = 256,
    ):
        features_dim = (
            2 * player_emb_dim  # players
            + 30 * card_emb_dim  # deck cards
            + zone_emb_dim  # deck
            + 8 * card_emb_dim  # hand cards
            + zone_emb_dim  # hand
            + 4 * 3 * creature_emb_dim  # lane creatures
            + 4 * lane_emb_dim  # lanes
            + state_emb_dim  # whole state
            # = 1824
        )

        super().__init__(observation_space, features_dim=features_dim)

        self.player_embedding = nn.Sequential(
            nn.Linear(player_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        ) # 5 * 16 + 16 * 16 = 336 parameters

        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        ) # 17 * 32 + 32 * 32 = 1,568 parameters
        
        self.card_zone_embedding = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(),
        ) # 32 * 32 = 1,024 parameters

        self.creature_embedding = nn.Sequential(
            nn.Linear(creature_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        ) # 8 * 16 + 16 * 16 = 384 parameters
        
        self.lane_embedding = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(),
        ) # 16 * 16 = 256 parameters
        
        # 2 * 16 player + 32 hand + 32 deck + 4 * 16 lane = 160 features
        self.state_embedding = nn.Sequential(
            nn.Linear(160, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        ) # 160 * 256 + 256 * 256 = 106,496 parameters

    def forward(self, observations) -> th.Tensor:
        # embedding of both players
        p = self.player_embedding(observations["player_stats"])
        op = self.player_embedding(observations["opponent_stats"])
        
        p_deck_cards = observations["player_deck"]
        
        # embedding of individual deck cards
        p_deck_cards = p_deck_cards.reshape(-1, 17)
        p_deck_cards = self.card_embedding(p_deck_cards)
        p_deck_cards = p_deck_cards.reshape(-1, 30, 32)
        
        # embedding of the whole deck
        p_deck = p_deck_cards.sum(dim=1)
        p_deck = self.card_zone_embedding(p_deck)

        p_hand_cards = observations["player_hand"]

        # embedding of individual hand cards
        p_hand_cards = p_hand_cards.reshape(-1, 17)
        p_hand_cards = self.card_embedding(p_hand_cards)
        p_hand_cards = p_hand_cards.reshape(-1, 8, 32)
        
        # embedding of the whole hand
        p_hand = p_hand_cards.sum(dim=1)
        p_hand = self.card_zone_embedding(p_hand)
        
        p_lane0_creatures = observations["player_lane0"]
        
        # embedding of individual player lane 0 creatures
        p_lane0_creatures = p_lane0_creatures.reshape(-1, 8)
        p_lane0_creatures = self.creature_embedding(p_lane0_creatures)
        p_lane0_creatures = p_lane0_creatures.reshape(-1, 3, 16)

        # embedding of the whole player lane 0
        p_lane0 = p_lane0_creatures.sum(dim=1)
        p_lane0 = self.lane_embedding(p_lane0)

        p_lane1_creatures = observations["player_lane1"]
        
        # embedding of individual player lane 1 creatures
        p_lane1_creatures = p_lane1_creatures.reshape(-1, 8)
        p_lane1_creatures = self.creature_embedding(p_lane1_creatures)
        p_lane1_creatures = p_lane1_creatures.reshape(-1, 3, 16)

        # embedding of the whole player lane 1
        p_lane1 = p_lane1_creatures.sum(dim=1)
        p_lane1 = self.lane_embedding(p_lane1)

        op_lane0_creatures = observations["opponent_lane0"]
        
        # embedding of individual opponent lane 0 creatures
        op_lane0_creatures = op_lane0_creatures.reshape(-1, 8)
        op_lane0_creatures = self.creature_embedding(op_lane0_creatures)
        op_lane0_creatures = op_lane0_creatures.reshape(-1, 3, 16)

        # embedding of the whole opponent lane 0
        op_lane0 = op_lane0_creatures.sum(dim=1)
        op_lane0 = self.lane_embedding(op_lane0)

        op_lane1_creatures = observations["opponent_lane1"]
        
        # embedding of individual opponent lane 1 creatures
        op_lane1_creatures = op_lane1_creatures.reshape(-1, 8)
        op_lane1_creatures = self.creature_embedding(op_lane1_creatures)
        op_lane1_creatures = op_lane1_creatures.reshape(-1, 3, 16)

        # embedding of the whole opponent lane 1
        op_lane1 = op_lane1_creatures.sum(dim=1)
        op_lane1 = self.lane_embedding(op_lane1)
        
        # embedding of the whole state
        state_input = th.cat((
            p, op, 
            p_deck, p_hand, 
            p_lane0, p_lane1, 
            op_lane0, op_lane1
        ), dim=1)
        state = self.state_embedding(state_input)

        embeddings = dict(
            player=p,
            opponent=op,
            deck_cards=p_deck_cards,
            deck=p_deck,
            hand_cards=p_hand_cards,
            hand=p_hand,
            p_lane0_creatures=p_lane0_creatures,
            p_lane0=p_lane0,
            p_lane1_creatures=p_lane1_creatures,
            p_lane1=p_lane1,
            op_lane0_creatures=op_lane0_creatures,
            op_lane0=op_lane0,
            op_lane1_creatures=op_lane1_creatures,
            op_lane1=op_lane1,
            state=state,
        )
        
        return embeddings


class PermutationInvariantLOCMNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        player_dim: int = 5,
        card_dim: int = 17,
        creature_dim: int = 8,
        last_layer_dim_pi: int = 145,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        
        # input: state (256)
        self.pass_action = nn.Sequential(
            nn.Linear(256, last_layer_dim_vf)
        )
        
        # input: source card (32) + target lane (16) + state (256) = 304
        self.summon_action = nn.Sequential(
            nn.Linear(304, 1)
        )
        
        # input: source card (32) + target creature (16) + state (256) = 304
        self.use_action = nn.Sequential(
            nn.Linear(304, 1)
        )

        # input: source card (16) + target creature (16) + state (256) = 288
        self.attack_action = nn.Sequential(
            nn.Linear(288, 1)
        )
        
        # value function head
        # input: state (256)
        self.value_net = nn.Sequential(
            nn.Linear(256, last_layer_dim_vf)
        )

    def forward(self, features: dict) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, embeddings: dict) -> th.Tensor:
        # PASS logit
        pass_logit = self.pass_action(embeddings["state"])  # [bs, 1]

        # SUMMON logits
        hand = embeddings["hand_cards"]  # [bs, max_hand_size, card_dim]
        hand = hand.repeat_interleave(2, dim=1)  # [bs, 2 * max_hand_size, card_dim]
        # hand = hand.reshape(-1, 16, 32)  # [bs, 2 * max_hand_size, card_dim]
        
        lane0 = embeddings["p_lane0"].reshape(-1, 1, 16)  # [bs, 1, lane_dim]
        lane1 = embeddings["p_lane1"].reshape(-1, 1, 16)  # [bs, 1, lane_dim]
        lanes = th.cat((lane0, lane1), dim=1)  # [bs, 2, lane_dim]
        lanes = lanes.repeat(1, 8, 1)  # [bs, 2 * max_hand_size, lane_dim]

        state = embeddings["state"].reshape(-1, 1, 256)  # [bs, 1, state_dim]
        state = state.repeat(1, 8 * 2, 1)  # [bs, 2 * max_hand_size, state_dim]

        summon_input = th.cat((hand, lanes, state), dim=2)  # [bs * 2 * max_hand_size, card_dim + lane_dim + state_dim]
        summon_input = summon_input.reshape(-1, 304)  # [bs * 2 * max_hand_size, card_dim + lane_dim + state_dim]
        summon_logits = self.summon_action(summon_input)  # [bs * 2 * max_hand_size, 1]
        summon_logits = summon_logits.reshape(-1, 8 * 2)  # [bs, max_hand_size * 2]
        
        # USE logits
        hand = embeddings["hand_cards"]  # [bs, max_hand_size, card_dim]
        targets = th.cat((
            th.zeros((hand.size(0), 1, 16), device=hand.device),  # for the "no target" option
            embeddings["p_lane0_creatures"], embeddings["p_lane1_creatures"],
            embeddings["op_lane0_creatures"], embeddings["op_lane1_creatures"],
        ), dim=1)  # [bs, 13, creature_dim]
        targets = targets.repeat(1, 8, 1)  # [bs, 13 * max_hand_size, creature_dim]
        state = embeddings["state"].reshape(-1, 1, 256)  # [bs, 1, state_dim]
        state = state.repeat(1, 8 * 13, 1)  # [bs, 13 * max_hand_size, state_dim]
        hand = hand.repeat_interleave(13, dim=1)  # [bs, 13 * max_hand_size, card_dim]
        
        use_input = th.cat((hand, targets, state), dim=2)  # [bs, 13 * max_hand_size, card_dim + creature_dim + state_dim]
        use_input = use_input.reshape(-1, 304)  # [bs * 13 * max_hand_size, card_dim + creature_dim + state_dim]
        use_logits = self.use_action(use_input)  # [bs * 13 * max_hand_size, 1]
        use_logits = use_logits.reshape(-1, 8 * 13)  # [bs, max_hand_size * 13])

        # ATTACK lane 0 logits
        op_lane0_creatures = embeddings["op_lane0_creatures"]  # [bs, 3, creature_dim]
        op_lane0_creatures = th.cat((th.zeros((op_lane0_creatures.size(0), 1, 16), device=op_lane0_creatures.device), op_lane0_creatures), dim=1)  # [bs, 4, creature_dim]
        op_lane0_creatures = op_lane0_creatures.repeat(1, 3, 1)  # [bs, 12, creature_dim]

        p_lane0_creatures = embeddings["p_lane0_creatures"]  # [bs, 3, creature_dim]
        p_lane0_creatures = p_lane0_creatures.repeat_interleave(4, dim=1)  # [bs, 12, creature_dim]

        state = embeddings["state"].reshape(-1, 1, 256)  # [bs, 1, state_dim]
        state = state.repeat(1, 12, 1)  # [bs, 12, state_dim]
        
        attack_lane0_input = th.cat((p_lane0_creatures, op_lane0_creatures, state), dim=2)  # [bs, 12, 2 * creature_dim + state_dim]
        attack_lane0_input = attack_lane0_input.reshape(-1, 288)  # [bs * 12, 2 * creature_dim + state_dim]

        attack_lane0_logits = self.attack_action(attack_lane0_input)  # [bs * 12, 1]
        attack_lane0_logits = attack_lane0_logits.reshape(-1, 12)  # [bs, 12]
        
        # ATTACK lane 1 logits
        op_lane1_creatures = embeddings["op_lane1_creatures"]  # [bs, 3, creature_dim]
        op_lane1_creatures = th.cat((th.zeros((op_lane1_creatures.size(0), 1, 16), device=op_lane1_creatures.device), op_lane1_creatures), dim=1)  # [bs, 4, creature_dim]
        op_lane1_creatures = op_lane1_creatures.repeat(1, 3, 1)  # [bs, 12, creature_dim]

        p_lane1_creatures = embeddings["p_lane1_creatures"]  # [bs, 3, creature_dim]
        p_lane1_creatures = p_lane1_creatures.repeat_interleave(4, dim=1)  # [bs, 12, creature_dim]
        
        state = embeddings["state"].reshape(-1, 1, 256)  # [bs, 1, state_dim]
        state = state.repeat(1, 12, 1)  # [bs, 12, state_dim]

        attack_lane1_input = th.cat((p_lane1_creatures, op_lane1_creatures, state), dim=2)  # [bs, 12, 2 * creature_dim + state_dim]
        attack_lane1_input = attack_lane1_input.reshape(-1, 288)  # [bs * 12, 2 * creature_dim + state_dim]

        attack_lane1_logits = self.attack_action(attack_lane1_input)  # [bs * 12, 1]
        attack_lane1_logits = attack_lane1_logits.reshape(-1, 12)  # [bs, 12]

        # concat all action logits
        logits = th.cat((
            pass_logit,
            summon_logits,
            use_logits,
            attack_lane0_logits,
            attack_lane1_logits,
        ), dim=1)  # [bs, 145]
        
        return logits

    def forward_critic(self, embeddings: dict) -> th.Tensor:
        return self.value_net(embeddings["state"])


class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):  
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            features_extractor_class=PermutationInvariantFeaturesExtractor,
            features_extractor_kwargs=dict(),
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        
        # do not add a nn.Linear layer on top of what we return at PermutationInvariantLOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # initialize weights of the new output layers (which are not initialized by the base class)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor.pass_action: 0.01,
                self.mlp_extractor.summon_action: 0.01,
                self.mlp_extractor.use_action: 0.01,
                self.mlp_extractor.attack_action: 0.01,
                self.mlp_extractor.value_net: 1,
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PermutationInvariantLOCMNetwork(self.features_dim, last_layer_dim_pi=145, last_layer_dim_vf=1)


def model_builder_pinv_masked(
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
    gae_lambda=0.95,
    tensorboard_log=None,
):
    return MaskablePPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=nminibatches,
        n_epochs=noptepochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=cliprange,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
