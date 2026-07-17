import torch as th
import torch.nn as nn
from functools import partial
from typing import Callable, Tuple

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from gymnasium.spaces import Space, Dict


class SimpleeFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: Dict, 
        net_arch: list[int],
        activation_fn: type[nn.Module],
        card_dim: int = 17, 
        player_dim: int = 5, 
        creature_dim: int = 17,
    ):
        input_dim = (
            2 * player_dim  # players
            + card_dim  # deck
            + 8 * card_dim  # hand cards
            + 4 * 3 * creature_dim  # lane creatures
            # = 367
        )

        super().__init__(observation_space, features_dim=net_arch[-1])

        net_arch = [input_dim] + net_arch
        
        layers = []

        for i in range(len(net_arch) - 1):
            layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
            layers.append(activation_fn())

        self.fc = nn.Sequential(*layers)

    def forward(self, features: dict) -> dict[str, th.Tensor]:
        player_stats = features["player_stats"]  # [bs, 5]
        opponent_stats = features["opponent_stats"]  # [bs, 5]
        player_deck = features["player_deck"]  # [bs, 30, 17]
        player_hand = features["player_hand"]  # [bs, 8, 17]
        p_lane0 = features["player_lane0"]  # [bs, 3, 17]
        p_lane1 = features["player_lane1"]  # [bs, 3, 17]
        op_lane0 = features["opponent_lane0"]  # [bs, 3, 17]
        op_lane1 = features["opponent_lane1"]  # [bs, 3, 17]

        bs = player_stats.shape[0]

        deck_mean = player_deck.mean(dim=1)  # [bs, 17]
        hand_flat = player_hand.view(bs, -1)  # [bs, 8*17]
        p_lane0_flat = p_lane0.view(bs, -1)  # [bs, 3*17]
        p_lane1_flat = p_lane1.view(bs, -1)  # [bs, 3*17]
        op_lane0_flat = op_lane0.view(bs, -1)  # [bs, 3*17]
        op_lane1_flat = op_lane1.view(bs, -1)  # [bs, 3*17]

        features_concat = th.cat((
            player_stats,
            opponent_stats,
            deck_mean,
            hand_flat,
            p_lane0_flat,
            p_lane1_flat,
            op_lane0_flat,
            op_lane1_flat,
        ), dim=1)

        return dict(latent=self.fc(features_concat), action_mask=features["action_mask"])


class SimpleeLOCMNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 145,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        self.policy = nn.Linear(feature_dim, last_layer_dim_pi)
        self.value_function = nn.Linear(feature_dim, last_layer_dim_vf)

    def forward(self, features: dict) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: dict) -> th.Tensor:
        logits = self.policy(features.get("latent"))

        action_mask = features.get("action_mask")
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def forward_critic(self, features: dict) -> th.Tensor:
        return self.value_function(features.get("latent"))


class SimpleeRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Callable[[float], float],
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        lstm_hidden_size: int = 256,
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

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            # Pass remaining arguments to base class
            features_extractor_class=SimpleeFeaturesExtractor,
            features_extractor_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
            *args,
            **kwargs,
        )
        
    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        
        # do not add a nn.Linear layer on top of what we return at SimpleeLOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # initialize weights of the new output layers (which are not initialized by the base class)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor.policy: 0.01,
                self.mlp_extractor.value_function: 1,
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SimpleeLOCMNetwork(self.lstm_hidden_size, last_layer_dim_pi=145, last_layer_dim_vf=1)

    def forward(
            self,
            obs: th.Tensor,
            lstm_states: RNNStates,
            episode_starts: th.Tensor,
            deterministic: bool = False,
        ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
            """
            Forward pass in all the networks (actor and critic)

            :param obs: Observation. Observation
            :param lstm_states: The last hidden and memory states for the LSTM.
            :param episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).
            :param deterministic: Whether to sample or use deterministic actions
            :return: action, value and log probability of the action
            """
            # Preprocess the observation if needed
            features = self.extract_features(obs)
            features, action_mask = features.get("latent"), features.get("action_mask")
                
            # latent_pi, latent_vf = self.mlp_extractor(features)
            latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())

            latent_pi = self.mlp_extractor.forward_actor(dict(latent=latent_pi, action_mask=action_mask))
            latent_vf = self.mlp_extractor.forward_critic(dict(latent=latent_vf, action_mask=action_mask))

            # Evaluate the values for the given observations
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
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        features, action_mask = features.get("latent"), features.get("action_mask")
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(dict(latent=latent_pi, action_mask=action_mask))
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        features, action_mask = features.get("latent"), features.get("action_mask")

        # Use LSTM from the actor
        latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_vf = latent_pi.detach()

        latent_vf = self.mlp_extractor.forward_critic(dict(latent=latent_vf, action_mask=action_mask))
        
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        features, action_mask = features.get("latent"), features.get("action_mask")

        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        latent_vf = latent_pi.detach()

        latent_pi = self.mlp_extractor.forward_actor(dict(latent=latent_pi, action_mask=action_mask))
        latent_vf = self.mlp_extractor.forward_critic(dict(latent=latent_vf, action_mask=action_mask))

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
        

class SimpleeActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Callable[[float], float],
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
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

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            # Pass remaining arguments to base class
            features_extractor_class=SimpleeFeaturesExtractor,
            features_extractor_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        
        # do not add a nn.Linear layer on top of what we return at SimpleeLOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # initialize weights of the new output layers (which are not initialized by the base class)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor.policy: 0.01,
                self.mlp_extractor.value_function: 1,
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SimpleeLOCMNetwork(self.features_extractor.features_dim, last_layer_dim_pi=145, last_layer_dim_vf=1)


def build_simple_network(
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
    lstm=False,
):
    if isinstance(layers, int):
        net_arch = [neurons] * layers
    elif isinstance(layers, dict) or isinstance(layers, list):
        net_arch = layers
    else:
        raise ValueError(f"Invalid type for layers: {type(layers)}.")
    
    activation = dict(tanh=th.nn.Tanh, relu=th.nn.ReLU, elu=th.nn.ELU)[activation]
    
    if lstm:
        algo = RecurrentPPO
        policy = SimpleeRecurrentActorCriticPolicy
        kwargs = dict(lstm_hidden_size=lstm)
    else:
        algo = PPO
        policy = SimpleeActorCriticPolicy
        kwargs = dict()

    return algo(
        policy,
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
        policy_kwargs=dict(net_arch=net_arch, activation_fn=activation, **kwargs),
        tensorboard_log=tensorboard_log,
    )


def load_simple_network(path, lstm=False):
    algo = RecurrentPPO if lstm else PPO
    
    def loaded_model_builder(env, seed, *args, **kwargs):
        return algo.load(path + ".zip", env=env, force_reset=True, seed=seed)

    return loaded_model_builder

