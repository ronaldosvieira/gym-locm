"""
Unified policy wrappers for LOCM.

These generic policies accept any feature extractor + actor-critic combination,
eliminating the need for 12+ nearly-identical policy classes.

Two policy classes:
- LOCMActorCriticPolicy: for feedforward PPO
- LOCMRecurrentActorCriticPolicy: for RecurrentPPO (LSTM between extraction and actor-critic)
"""

import torch as th
import torch.nn as nn
from functools import partial
from typing import Callable

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from gymnasium.spaces import Space

from gym_locm.toolbox.networks.utils import safely_compile


class LOCMActorCriticPolicy(ActorCriticPolicy):
    """Unified feedforward policy for standard PPO.
    
    Accepts any (feature_extractor_class, actor_critic_class) combination.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Callable[[float], float],
        actor_critic_class=None,
        actor_critic_kwargs=None,
        compile_model: bool = True,
        *args,
        **kwargs,
    ):
        # Store actor-critic config before super().__init__ calls _build_mlp_extractor
        self._actor_critic_class = actor_critic_class
        self._actor_critic_kwargs = actor_critic_kwargs or {}
        self._compile_model = compile_model

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        if self._compile_model:
            self.features_extractor = safely_compile(self.features_extractor)

        # Bypass SB3's default output layers — our mlp_extractor outputs final logits/values
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Orthogonal initialization
        if self.ortho_init:
            for module in self.mlp_extractor.get_policy_modules():
                module.apply(partial(self.init_weights, gain=0.01))
            for module in self.mlp_extractor.get_value_modules():
                module.apply(partial(self.init_weights, gain=1.0))

    def _build_mlp_extractor(self) -> None:
        # Get embedding dimensions from the feature extractor
        feat_ext = self.features_extractor

        if self._actor_critic_kwargs:
            kwargs = dict(self._actor_critic_kwargs)
        else:
            kwargs = {}

        # Auto-populate dimension kwargs from the feature extractor
        if hasattr(feat_ext, 'state_dim'):
            kwargs.setdefault('state_dim', feat_ext.state_dim)
            kwargs.setdefault('card_emb_dim', feat_ext.hand_cards_dim)
            kwargs.setdefault('creature_emb_dim', feat_ext.creature_tokens_dim)
            kwargs.setdefault('lane_emb_dim', feat_ext.lane_dim)
        else:
            # Simple feature extractor: only has features_dim
            kwargs.setdefault('feature_dim', feat_ext.features_dim)

        self.mlp_extractor = self._actor_critic_class(
            last_layer_dim_pi=145,
            last_layer_dim_vf=1,
            **kwargs,
        )
        if self._compile_model:
            self.mlp_extractor = safely_compile(self.mlp_extractor)


class LOCMRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """Unified recurrent policy for RecurrentPPO (LSTM).

    Handles two feature extractor output formats:
    - Simple: {latent: [bs, feat_dim], action_mask} — LSTM processes latent
    - Entity-level: {hand_cards, ..., state, action_mask} — LSTM processes state
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Callable[[float], float],
        actor_critic_class=None,
        actor_critic_kwargs=None,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        lstm_hidden_size: int = 256,
        compile_model: bool = True,
        *args,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = [256, 256]

        if isinstance(net_arch, int):
            net_arch = [net_arch]
        elif isinstance(net_arch, dict):
            raise ValueError("dict net_arch not supported.")

        self.net_arch = net_arch
        self.lstm_hidden_size = lstm_hidden_size
        self._actor_critic_class = actor_critic_class
        self._actor_critic_kwargs = actor_critic_kwargs or {}
        self._compile_model = compile_model

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        if self._compile_model:
            self.features_extractor = safely_compile(self.features_extractor)

        # Bypass SB3's default output layers
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Orthogonal initialization
        if self.ortho_init:
            for module in self.mlp_extractor.get_policy_modules():
                module.apply(partial(self.init_weights, gain=0.01))
            for module in self.mlp_extractor.get_value_modules():
                module.apply(partial(self.init_weights, gain=1.0))

    def _build_mlp_extractor(self) -> None:
        if self._actor_critic_kwargs:
            kwargs = dict(self._actor_critic_kwargs)
        else:
            kwargs = {}

        # For LSTM, the actor-critic receives lstm_hidden_size as state_dim
        # (since the LSTM replaces the state/latent vector)
        feat_ext = self.features_extractor
        if hasattr(feat_ext, 'state_dim'):
            # Entity-level extractor: LSTM processes state, output replaces state
            kwargs.setdefault('state_dim', self.lstm_hidden_size)
            kwargs.setdefault('card_emb_dim', feat_ext.hand_cards_dim)
            kwargs.setdefault('creature_emb_dim', feat_ext.creature_tokens_dim)
            kwargs.setdefault('lane_emb_dim', feat_ext.lane_dim)
        else:
            # Simple extractor: LSTM processes latent, output replaces latent
            kwargs.setdefault('feature_dim', self.lstm_hidden_size)

        self.mlp_extractor = self._actor_critic_class(
            last_layer_dim_pi=145,
            last_layer_dim_vf=1,
            **kwargs,
        )
        if self._compile_model:
            self.mlp_extractor = safely_compile(self.mlp_extractor)

    @property
    def _is_simple_extractor(self):
        """Check if feature extractor uses simple format ({latent, action_mask})."""
        return not hasattr(self.features_extractor, 'state_dim')

    def _extract_lstm_input(self, features):
        """Extract the tensor that goes through the LSTM."""
        if self._is_simple_extractor:
            return features.get("latent"), features.get("action_mask")
        else:
            return features.get("state"), features.get("action_mask")

    def _replace_lstm_output(self, features, lstm_output, action_mask):
        """Replace the LSTM-processed field in the features dict."""
        if self._is_simple_extractor:
            return dict(latent=lstm_output, action_mask=action_mask)
        else:
            features["state"] = lstm_output
            return features

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        lstm_input, action_mask = self._extract_lstm_input(features)

        latent_pi, lstm_states_pi = self._process_sequence(
            lstm_input, lstm_states.pi, episode_starts, self.lstm_actor
        )

        # Re-use LSTM features but do not backpropagate for critic
        latent_vf = latent_pi.detach()
        lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())

        features_pi = self._replace_lstm_output(dict(features), latent_pi, action_mask)
        latent_pi = self.mlp_extractor.forward_actor(features_pi)

        features_vf = self._replace_lstm_output(dict(features), latent_vf, action_mask)
        latent_vf = self.mlp_extractor.forward_critic(features_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, ...]]:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        lstm_input, action_mask = self._extract_lstm_input(features)
        latent_pi, lstm_states = self._process_sequence(
            lstm_input, lstm_states, episode_starts, self.lstm_actor
        )

        features = self._replace_lstm_output(features, latent_pi, action_mask)
        latent_pi = self.mlp_extractor.forward_actor(features)

        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        lstm_input, action_mask = self._extract_lstm_input(features)

        latent_pi, _ = self._process_sequence(
            lstm_input, lstm_states, episode_starts, self.lstm_actor
        )
        latent_vf = latent_pi.detach()

        features = self._replace_lstm_output(features, latent_vf, action_mask)
        latent_vf = self.mlp_extractor.forward_critic(features)

        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        lstm_input, action_mask = self._extract_lstm_input(features)

        latent_pi, _ = self._process_sequence(
            lstm_input, lstm_states.pi, episode_starts, self.lstm_actor
        )
        latent_vf = latent_pi.detach()

        features_pi = self._replace_lstm_output(dict(features), latent_pi, action_mask)
        latent_pi = self.mlp_extractor.forward_actor(features_pi)

        features_vf = self._replace_lstm_output(dict(features), latent_vf, action_mask)
        latent_vf = self.mlp_extractor.forward_critic(features_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
