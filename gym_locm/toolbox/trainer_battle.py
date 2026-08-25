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
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.vec_env import (
    VecEnv as VecEnv3,
    DummyVecEnv as DummyVecEnv3,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO, RecurrentPPO
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
    def __init__(self, task, params, path, seed, wandb_run=None, profile_gpu: bool = False):
        # initialize logger
        self.logger = logging.getLogger("{0}.{1}".format(__name__, type(self).__name__))
        self.profile_gpu = profile_gpu

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.win_rates_1p = []
        self.win_rates_2p = []
        self.episode_lengths = []
        self.battle_lengths = []
        self.health_diffs = []
        self.action_histograms = []
        
        self.turn_manas = []
        self.turn_hand_sizes = []
        self.lane_board_values = []
        self.lane_balances = []
        self.favorable_trades_ratios = []
        self.face_attacks = []
        self.creature_attacks = []
        self.skipped_dominant_actions = []

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
        profile_gpu: bool = False,
    ):
        super(FixedAdversary, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run, profile_gpu=profile_gpu
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

                results = evaluator.run(
                    agent,
                    play_first=self.role == "first",
                    alternate_roles=self.role == "alternate",
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"wr: {round(results['win_rate'] * 100, 3)}%. "
                    f"Avg. rew: {results['mean_reward']:.3f}"
                )

                # save the results
                self.checkpoints.append(episodes_so_far)
                self.win_rates.append(results["win_rate"])
                self.win_rates_1p.append(results["win_rate_1p"])
                self.win_rates_2p.append(results["win_rate_2p"])
                self.episode_lengths.append(results["ep_length"])
                self.battle_lengths.append(results["battle_length"])
                self.health_diffs.append(results["health_diff"])
                self.action_histograms.append(results["act_hist"])
                self.turn_manas.append(results["mean_turn_mana"])
                self.turn_hand_sizes.append(results["mean_turn_hand_size"])
                self.lane_board_values.append(results["mean_lane_board_value"])
                self.lane_balances.append(results["mean_lane_balance"])
                self.face_attacks.append(results["mean_face_attacks"])
                self.creature_attacks.append(results["mean_creature_attacks"])
                self.favorable_trades_ratios.append(results["favorable_trades_ratio"])
                self.skipped_dominant_actions.append(results["mean_skipped_dominant_actions"])

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = dict()

                    info["checkpoint"] = episodes_so_far
                    info[panel_name + "/mean_reward"] = results["mean_reward"]
                    info[panel_name + "/win_rate"] = results["win_rate"]
                    info[panel_name + "/win_rate_1p"] = results["win_rate_1p"]
                    info[panel_name + "/win_rate_2p"] = results["win_rate_2p"]
                    info[panel_name + "/mean_ep_length"] = results["ep_length"]
                    info[panel_name + "/mean_battle_length"] = results["battle_length"]
                    info[panel_name + "/mean_health_diff"] = results["health_diff"]
                    info[panel_name + "/mean_turn_mana"] = results["mean_turn_mana"]
                    info[panel_name + "/mean_turn_hand_size"] = results["mean_turn_hand_size"]
                    info[panel_name + "/mean_lane_board_value"] = results["mean_lane_board_value"]
                    info[panel_name + "/mean_lane_balance"] = results["mean_lane_balance"]
                    info[panel_name + "/mean_face_attacks"] = results["mean_face_attacks"]
                    info[panel_name + "/mean_creature_attacks"] = results["mean_creature_attacks"]
                    info[panel_name + "/favorable_trades_ratio"] = results["favorable_trades_ratio"]
                    info[panel_name + "/mean_skipped_dominant_actions"] = results["mean_skipped_dominant_actions"]

                    act_hist = results["act_hist"]
                    info[panel_name + "/win_rate"] = results["win_rate"]
                    info[panel_name + "/mean_ep_length"] = results["ep_length"]
                    info[panel_name + "/mean_battle_length"] = results["battle_length"]
                    info[panel_name + "/mean_health_diff"] = results["health_diff"]

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

        if self.profile_gpu:
            callbacks.append(GPUMemoryLogger(self.wandb_run))

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
        profile_gpu: bool = False,
    ):
        super(SelfPlay, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run, profile_gpu=profile_gpu
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
            def adversary_policy(obs):
                actions, _ = model.adversary.predict(obs)

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

                results = evaluator.run(
                    agent,
                    play_first=self.role == "first",
                    alternate_roles=self.role == "alternate",
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"wr: {round(results['win_rate'] * 100, 3)}%. "
                    f"Avg. rew: {results['mean_reward']:.3f}"
                )

                # save the results
                self.checkpoints.append(episodes_so_far)
                self.win_rates.append(results["win_rate"])
                self.win_rates_1p.append(results["win_rate_1p"])
                self.win_rates_2p.append(results["win_rate_2p"])
                self.episode_lengths.append(results["ep_length"])
                self.battle_lengths.append(results["battle_length"])
                self.health_diffs.append(results["health_diff"])
                self.action_histograms.append(results["act_hist"])
                self.turn_manas.append(results["mean_turn_mana"])
                self.turn_hand_sizes.append(results["mean_turn_hand_size"])
                self.lane_board_values.append(results["mean_lane_board_value"])
                self.lane_balances.append(results["mean_lane_balance"])
                self.face_attacks.append(results["mean_face_attacks"])
                self.creature_attacks.append(results["mean_creature_attacks"])
                self.favorable_trades_ratios.append(results["favorable_trades_ratio"])
                self.skipped_dominant_actions.append(results["mean_skipped_dominant_actions"])

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = dict()

                    info["checkpoint"] = episodes_so_far
                    info[panel_name + "/mean_reward"] = results["mean_reward"]
                    info[panel_name + "/win_rate"] = results["win_rate"]
                    info[panel_name + "/win_rate_1p"] = results["win_rate_1p"]
                    info[panel_name + "/win_rate_2p"] = results["win_rate_2p"]
                    info[panel_name + "/mean_ep_length"] = results["ep_length"]
                    info[panel_name + "/mean_battle_length"] = results["battle_length"]
                    info[panel_name + "/mean_health_diff"] = results["health_diff"]
                    info[panel_name + "/mean_turn_mana"] = results["mean_turn_mana"]
                    info[panel_name + "/mean_turn_hand_size"] = results["mean_turn_hand_size"]
                    info[panel_name + "/mean_lane_board_value"] = results["mean_lane_board_value"]
                    info[panel_name + "/mean_lane_balance"] = results["mean_lane_balance"]
                    info[panel_name + "/mean_face_attacks"] = results["mean_face_attacks"]
                    info[panel_name + "/mean_creature_attacks"] = results["mean_creature_attacks"]
                    info[panel_name + "/favorable_trades_ratio"] = results["favorable_trades_ratio"]
                    info[panel_name + "/mean_skipped_dominant_actions"] = results["mean_skipped_dominant_actions"]

                    act_hist = results["act_hist"]

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

        if self.profile_gpu:
            callbacks.append(GPUMemoryLogger(self.wandb_run))

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
        profile_gpu: bool = False,
    ):
        super(FixedAndSelfPlayHybrid, self).__init__(
            task, model_params, path, seed, wandb_run=wandb_run, profile_gpu=profile_gpu
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
            def adversary_policy(obs):
                actions, _ = model.adversary.predict(obs)

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

                results = evaluator.run(
                    agent,
                    play_first=self.role == "first",
                    alternate_roles=self.role == "alternate",
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"wr: {round(results['win_rate'] * 100, 3)}%. "
                    f"Avg. rew: {results['mean_reward']:.3f}"
                )

                # save the results
                self.checkpoints.append(episodes_so_far)
                self.win_rates.append(results["win_rate"])
                self.win_rates_1p.append(results["win_rate_1p"])
                self.win_rates_2p.append(results["win_rate_2p"])
                self.episode_lengths.append(results["ep_length"])
                self.battle_lengths.append(results["battle_length"])
                self.health_diffs.append(results["health_diff"])
                self.action_histograms.append(results["act_hist"])
                self.turn_manas.append(results["mean_turn_mana"])
                self.turn_hand_sizes.append(results["mean_turn_hand_size"])
                self.lane_board_values.append(results["mean_lane_board_value"])
                self.lane_balances.append(results["mean_lane_balance"])
                self.face_attacks.append(results["mean_face_attacks"])
                self.creature_attacks.append(results["mean_creature_attacks"])
                self.favorable_trades_ratios.append(results["favorable_trades_ratio"])
                self.skipped_dominant_actions.append(results["mean_skipped_dominant_actions"])

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = dict()

                    info["checkpoint"] = episodes_so_far
                    info[panel_name + "/mean_reward"] = results["mean_reward"]
                    info[panel_name + "/win_rate"] = results["win_rate"]
                    info[panel_name + "/win_rate_1p"] = results["win_rate_1p"]
                    info[panel_name + "/win_rate_2p"] = results["win_rate_2p"]
                    info[panel_name + "/mean_ep_length"] = results["ep_length"]
                    info[panel_name + "/mean_battle_length"] = results["battle_length"]
                    info[panel_name + "/mean_health_diff"] = results["health_diff"]
                    info[panel_name + "/mean_turn_mana"] = results["mean_turn_mana"]
                    info[panel_name + "/mean_turn_hand_size"] = results["mean_turn_hand_size"]
                    info[panel_name + "/mean_lane_board_value"] = results["mean_lane_board_value"]
                    info[panel_name + "/mean_lane_balance"] = results["mean_lane_balance"]
                    info[panel_name + "/mean_face_attacks"] = results["mean_face_attacks"]
                    info[panel_name + "/mean_creature_attacks"] = results["mean_creature_attacks"]
                    info[panel_name + "/favorable_trades_ratio"] = results["favorable_trades_ratio"]
                    info[panel_name + "/mean_skipped_dominant_actions"] = results["mean_skipped_dominant_actions"]

                    act_hist = results["act_hist"]

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

        if self.profile_gpu:
            callbacks.append(GPUMemoryLogger(self.wandb_run))

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
        episode_wins_1p = [[] for _ in range(self.env.num_envs)]
        episode_wins_2p = [[] for _ in range(self.env.num_envs)]
        episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
        episode_health_diff = [[] for _ in range(self.env.num_envs)]
        episode_lengths = [[0] for _ in range(self.env.num_envs)]
        episode_turns = [[] for _ in range(self.env.num_envs)]
        action_histogram = [0] * self.env.action_space.n

        episode_turn_mana = [[] for _ in range(self.env.num_envs)]
        episode_turn_hand_size = [[] for _ in range(self.env.num_envs)]
        episode_lane_board_value = [[] for _ in range(self.env.num_envs)]
        episode_lane_balance = [[] for _ in range(self.env.num_envs)]
        
        episode_face_attacks = [[0] for _ in range(self.env.num_envs)]
        episode_creature_attacks = [[0] for _ in range(self.env.num_envs)]
        episode_favorable_trades = [[0] for _ in range(self.env.num_envs)]
        episode_skipped_dominant_actions = [[0] for _ in range(self.env.num_envs)]

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
                actions = agent.act(observations)
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

                if "turn_mana" in infos[i]:
                    episode_turn_mana[i].append(infos[i]["turn_mana"])
                    episode_turn_hand_size[i].append(infos[i]["turn_hand_size"])
                    episode_lane_board_value[i].append(infos[i]["lane0_value"] + infos[i]["lane1_value"])
                    episode_lane_balance[i].append(abs(infos[i]["lane0_value"] - infos[i]["lane1_value"]))
                
                if infos[i].get("skipped_dominant_action"):
                    episode_skipped_dominant_actions[i][-1] += 1
                
                if infos[i].get("face_attack"):
                    episode_face_attacks[i][-1] += 1
                elif infos[i].get("creature_attack"):
                    episode_creature_attacks[i][-1] += 1
                    if infos[i].get("favorable_trade"):
                        episode_favorable_trades[i][-1] += 1

                if dones[i]:
                    win = 1 if infos[i]["winner"] == roles[i] else 0
                    episode_wins[i].append(win)
                    if roles[i] == 0:
                        episode_wins_1p[i].append(win)
                    else:
                        episode_wins_2p[i].append(win)

                    episode_rewards[i].append(0.0)
                    episode_lengths[i].append(0)
                    episode_turns[i].append(infos[i]["turn"])
                    episode_health_diff[i].append(infos[i]["health_diff"][roles[i]])
                    
                    episode_face_attacks[i].append(0)
                    episode_creature_attacks[i].append(0)
                    episode_favorable_trades[i].append(0)
                    episode_skipped_dominant_actions[i].append(0)

                    episodes_so_far += 1

            # check exiting condition
            if episodes_so_far >= self.episodes:
                break

        # join all parallel metrics
        all_rewards = [reward for rewards in episode_rewards for reward in rewards[:-1]]
        all_lengths = [length for lengths in episode_lengths for length in lengths[:-1]]
        all_turns = [turn for turns in episode_turns for turn in turns]
        all_health_diff = [health for health_diffs in episode_health_diff for health in health_diffs]
        all_wins = [win for wins in episode_wins for win in wins]
        all_wins_1p = [win for wins in episode_wins_1p for win in wins]
        all_wins_2p = [win for wins in episode_wins_2p for win in wins]
        
        all_turn_mana = [m for m_list in episode_turn_mana for m in m_list]
        all_turn_hand_size = [s for s_list in episode_turn_hand_size for s in s_list]
        all_lane_board_value = [v for v_list in episode_lane_board_value for v in v_list]
        all_lane_balance = [b for b_list in episode_lane_balance for b in b_list]
        
        all_face_attacks = [fa for fa_list in episode_face_attacks for fa in fa_list[:-1]]
        all_creature_attacks = [ca for ca_list in episode_creature_attacks for ca in ca_list[:-1]]
        all_favorable_trades = [ft for ft_list in episode_favorable_trades for ft in ft_list[:-1]]
        all_skipped_dominant = [sd for sd_list in episode_skipped_dominant_actions for sd in sd_list[:-1]]

        # transform the action histogram in a probability distribution
        action_histogram = [
            action_freq / sum(action_histogram) for action_freq in action_histogram
        ]

        # cap any unsolicited additional episodes
        all_wins = all_wins[: self.episodes]
        all_rewards = all_rewards[: self.episodes]
        all_lengths = all_lengths[: self.episodes]
        all_turns = all_turns[: self.episodes]
        all_health_diff = all_health_diff[: self.episodes]
        
        # calculate derived means safely
        mean_turn_mana = mean(all_turn_mana) if all_turn_mana else 0.0
        mean_turn_hand_size = mean(all_turn_hand_size) if all_turn_hand_size else 0.0
        mean_lane_board_value = mean(all_lane_board_value) if all_lane_board_value else 0.0
        mean_lane_balance = mean(all_lane_balance) if all_lane_balance else 0.0
        
        mean_face_attacks = mean(all_face_attacks) if all_face_attacks else 0.0
        mean_creature_attacks = mean(all_creature_attacks) if all_creature_attacks else 0.0
        favorable_trades_ratio = sum(all_favorable_trades) / sum(all_creature_attacks) if sum(all_creature_attacks) > 0 else 0.0
        mean_skipped_dominant = mean(all_skipped_dominant) if all_skipped_dominant else 0.0
        
        win_rate_1p = mean(all_wins_1p) if all_wins_1p else 0.0
        win_rate_2p = mean(all_wins_2p) if all_wins_2p else 0.0

        return dict(
            win_rate=mean(all_wins),
            win_rate_1p=win_rate_1p,
            win_rate_2p=win_rate_2p,
            mean_reward=mean(all_rewards),
            ep_length=mean(all_lengths),
            battle_length=mean(all_turns),
            health_diff=mean(all_health_diff),
            act_hist=action_histogram,
            mean_turn_mana=mean_turn_mana,
            mean_turn_hand_size=mean_turn_hand_size,
            mean_lane_board_value=mean_lane_board_value,
            mean_lane_balance=mean_lane_balance,
            mean_face_attacks=mean_face_attacks,
            mean_creature_attacks=mean_creature_attacks,
            favorable_trades_ratio=favorable_trades_ratio,
            mean_skipped_dominant_actions=mean_skipped_dominant
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


class GPUMemoryLogger(BaseCallback):
    """Logs CUDA memory usage at rollout boundaries (and optionally per training step).

    Reports three numbers every rollout:
      - allocated:  memory actively used by live tensors
      - reserved:   memory held in PyTorch's cache (allocated + fragmented free blocks)
      - fragmented: reserved - allocated (free blocks PyTorch can't return to the OS)

    Set ``per_step=True`` to also log after every single environment step; this lets
    you distinguish memory growth that happens during rollout collection vs. during
    the PPO training update.
    """

    def __init__(self, wandb_run=None, per_step: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.mem_logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.wandb_run = wandb_run
        self.per_step = per_step
        self._rollout_count = 0

    def _cuda_stats(self) -> dict:
        if not th.cuda.is_available():
            return {}
        alloc_mb = th.cuda.memory_allocated() / 1024 ** 2
        reserved_mb = th.cuda.memory_reserved() / 1024 ** 2
        return {
            "gpu_mem/allocated_mb": alloc_mb,
            "gpu_mem/reserved_mb": reserved_mb,
            "gpu_mem/fragmented_mb": reserved_mb - alloc_mb,
        }

    def _log(self, tag: str, stats: dict) -> None:
        if not stats:
            return
        alloc = stats["gpu_mem/allocated_mb"]
        reserved = stats["gpu_mem/reserved_mb"]
        frag = stats["gpu_mem/fragmented_mb"]
        self.mem_logger.debug(
            f"[{tag}] alloc={alloc:.1f} MB  reserved={reserved:.1f} MB  fragmented={frag:.1f} MB"
        )
        if self.wandb_run:
            self.wandb_run.log({f"{tag}/{k}": v for k, v in stats.items()})

    def _on_step(self) -> bool:
        if self.per_step:
            self._log("step", self._cuda_stats())
        return True

    def _on_rollout_start(self) -> None:
        self._log("rollout_start", self._cuda_stats())

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        self._log(f"rollout_end/{self._rollout_count}", self._cuda_stats())


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

