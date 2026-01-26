import json
import logging
import math
import os
import time
from typing import List

import numpy as np
from abc import abstractmethod
from datetime import datetime
from statistics import mean

import torch as th

from stable_baselines3.common.vec_env import (
    VecEnv as VecEnv3,
    DummyVecEnv as DummyVecEnv3,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sb3_contrib import MaskablePPO
from wandb.integration.sb3 import WandbCallback

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
            type(e["battle_agent"]).__name__ for e in eval_env_params
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

                # update control attributes
                model.last_eval = episodes_so_far
                model.next_eval += self.eval_frequency

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

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callbacks = [TrainingCallback(self._training_callback)]

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
            type(e["battle_agent"]).__name__ for e in eval_env_params
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

                # update control attributes
                model.last_eval = episodes_so_far
                model.next_eval += self.eval_frequency

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
                model.env.set_attr("rewards_single_player", [], indices=[i])

            # update parameters of adversary models
            model.adversary.set_parameters(model.get_parameters(), exact_match=True)

            self.logger.debug("Parameters of adversary network updated.")

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callbacks = [TrainingCallback(self._training_callback)]

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
            type(e["battle_agent"]).__name__ for e in eval_env_params
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

                # update control attributes
                model.last_eval = episodes_so_far
                model.next_eval += self.eval_frequency

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
                model.env.set_attr("rewards_single_player", [], indices=[i])

            # update parameters of adversary models
            model.adversary.set_parameters(model.get_parameters(), exact_match=True)

            self.logger.debug("Parameters of adversary network updated.")

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        callbacks = [TrainingCallback(self._training_callback)]

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

        env_class = LOCMBattleSelfPlayEnv

        for i in range(num_envs):
            # no overlap between episodes at each process
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create one env per process
            env1.append(
                lambda: env_class(
                    seed=current_seed,
                    play_first=True,
                    alternate_role=False,
                    **env_params,
                )
            )
            env2.append(
                lambda: env_class(
                    seed=current_seed,
                    play_first=False,
                    alternate_role=False,
                    **env_params,
                )
            )

        # wrap envs in a vectorized env
        self.env1: VecEnv3 = DummyVecEnv3(env1)
        self.env2: VecEnv3 = DummyVecEnv3(env2)

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluators: List[Evaluator] = [
            Evaluator(task, e, eval_episodes, eval_seed, num_envs)
            for e in eval_env_params
        ]

        # build the models
        self.logger.debug("Building the models...")
        self.model1 = model_builder(self.env1, seed, **model_params)
        self.model1.adversary = model_builder(self.env2, seed, **model_params)
        self.model2 = model_builder(self.env2, seed, **model_params)
        self.model2.adversary = model_builder(self.env1, seed, **model_params)

        # initialize parameters of adversary models accordingly
        self.model1.adversary.set_parameters(
            self.model2.get_parameters(), exact_match=True
        )
        self.model2.adversary.set_parameters(
            self.model1.get_parameters(), exact_match=True
        )

        # set adversary models as adversary policies of the self-play envs
        def make_adversary_policy(model):
            def adversary_policy(obs, action_mask):
                actions, _ = model.adversary.predict(obs, action_masks=action_mask)

                return actions

            return adversary_policy

        self.env1.set_attr("adversary_policy", make_adversary_policy(self.model1))
        self.env2.set_attr("adversary_policy", make_adversary_policy(self.model2))

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
        self.eval_adversaries = [
            type(e["battle_agent"]).__name__ for e in eval_env_params
        ]

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

            model.save(model_path, exclude=["adversary"])
            save_model_as_json(model, self.params["activation"], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(
                f"Evaluating model {model.role_id} " f"({episodes_so_far} episodes)..."
            )
            start_time = time.perf_counter()

            agent_class = RLBattleAgent

            for evaluator, eval_adversary in zip(
                self.evaluators, self.eval_adversaries
            ):
                (
                    win_rate,
                    mean_reward,
                    ep_length,
                    battle_length,
                    act_hist,
                ) = evaluator.run(
                    agent_class(model, deterministic=True),
                    play_first=model.role_id == 0,
                )

                end_time = time.perf_counter()
                self.logger.info(
                    f"Finished evaluating vs {eval_adversary} "
                    f"({round(end_time - start_time, 3)}s). "
                    f"Avg. reward: {mean_reward}"
                )

                # save the results
                self.checkpoints[model.role_id].append(episodes_so_far)
                self.win_rates[model.role_id].append(win_rate)
                self.episode_lengths[model.role_id].append(ep_length)
                self.battle_lengths[model.role_id].append(battle_length)
                self.action_histograms[model.role_id].append(act_hist)

                # update control attributes
                model.last_eval = episodes_so_far
                model.next_eval += self.eval_frequency

                # upload stats to wandb, if enabled
                if self.wandb_run:
                    panel_name = f"eval_vs_{eval_adversary}"

                    info = {
                        "checkpoint_" + model.role_id: episodes_so_far,
                        panel_name + "/mean_reward_" + model.role_id: mean_reward,
                        panel_name + "/win_rate_" + model.role_id: win_rate,
                        panel_name + "/mean_ep_length_" + model.role_id: ep_length,
                        panel_name
                        + "/mean_battle_length_"
                        + model.role_id: battle_length,
                        panel_name + "/pass_actions_" + model.role_id: act_hist[0],
                        panel_name
                        + "/summon_actions_"
                        + model.role_id: sum(act_hist[1:17]),
                    }

                    if model.env.get_attr("items", indices=[0])[0]:
                        info[panel_name + "/use_actions" + model.role_id] = sum(
                            act_hist[17:121]
                        )
                        info[panel_name + "/attack_actions" + model.role_id] = sum(
                            act_hist[121:]
                        )
                    else:
                        info[panel_name + "/attack_actions" + model.role_id] = sum(
                            act_hist[17:]
                        )

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

        try:
            self.logger.debug(
                f"Training will switch models every "
                f"{self.switch_frequency} episodes"
            )

            callbacks1 = [
                TrainingCallback(lambda: self._training_callback({"self": self.model1}))
            ]
            callbacks2 = [
                TrainingCallback(lambda: self._training_callback({"self": self.model2}))
            ]

            if self.wandb_run:
                callbacks1.append(WandbCallback(gradient_save_freq=0, verbose=0))
                callbacks2.append(WandbCallback(gradient_save_freq=0, verbose=0))

            for _ in range(self.num_switches):
                # train the first player model
                self.model1.learn(
                    total_timesteps=REALLY_BIG_INT,
                    reset_num_timesteps=False,
                    callback=CallbackList(callbacks1),
                )

                # log training win rate at the time of the switch
                train_mean_reward1 = np.mean(
                    [
                        np.mean(rewards)
                        for rewards in self.env1.env_method("get_episode_rewards")
                    ]
                )
                if self.wandb_run:
                    self.wandb_run.log({"train_mean_reward_0": train_mean_reward1})

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
                    callback=CallbackList(callbacks2),
                )

                # log training win rate at the time of the switch
                train_mean_reward2 = np.mean(
                    [
                        np.mean(rewards)
                        for rewards in self.env2.env_method("get_episode_rewards")
                    ]
                )
                if self.wandb_run:
                    self.wandb_run.log({"train_mean_reward_1": train_mean_reward2})

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

        # close all envs
        self.env1.close()
        self.env2.close()

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

        # log end time
        end_time = time.perf_counter()

        self.logger.debug(
            "Finished initializing evaluator "
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
