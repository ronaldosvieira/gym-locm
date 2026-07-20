import math
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from gym_locm.toolbox.networks.set_transformer import PMA

class HeteroGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_types = ["summon", "use", "attack", "struct"]
        
        self.W = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim, bias=False) 
            for edge_type in self.edge_types
        })
        self.W_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, X, A_dict):
        # X: [bs, 25, hidden_dim]
        # A_dict: dict of edge_type -> [bs, 25, 25] float tensor
        out = self.W_self(X)
        for edge_type in self.edge_types:
            A = A_dict[edge_type]
            # message passing: A @ X
            msg = th.bmm(A, X)
            out = out + self.W[edge_type](msg)
            
        out = self.norm(out)
        out = F.relu(out)
        return out


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = features_dim
        self.num_layers = num_layers

        player_features = observation_space["player_stats"].shape[0]
        card_features = observation_space["player_hand"].shape[1]
        creature_features = observation_space["player_lane0"].shape[1]

        # Initial projections to project everything to hidden_dim
        self.player_proj = nn.Linear(player_features, self.hidden_dim)
        self.card_proj = nn.Linear(card_features, self.hidden_dim)
        self.creature_proj = nn.Linear(creature_features, self.hidden_dim)
        
        # Learnable nodes for the two lanes
        self.lane0_node = nn.Parameter(th.randn(1, 1, self.hidden_dim))
        self.lane1_node = nn.Parameter(th.randn(1, 1, self.hidden_dim))

        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(self.hidden_dim) for _ in range(self.num_layers)
        ])

    def forward(self, observations) -> dict[str, th.Tensor]:
        bs = observations["player_stats"].size(0)
        device = observations["player_stats"].device

        # Project all raw features to hidden_dim
        p = self.player_proj(observations["player_stats"]).unsqueeze(1)  # [bs, 1, hidden]
        op = self.player_proj(observations["opponent_stats"]).unsqueeze(1) # [bs, 1, hidden]
        
        deck = self.card_proj(observations["player_deck"]).mean(dim=1, keepdim=True) # [bs, 1, hidden]
        hand = self.card_proj(observations["player_hand"]) # [bs, 8, hidden]
        
        p_lane0 = self.creature_proj(observations["player_lane0"]) # [bs, 3, hidden]
        p_lane1 = self.creature_proj(observations["player_lane1"]) # [bs, 3, hidden]
        op_lane0 = self.creature_proj(observations["opponent_lane0"]) # [bs, 3, hidden]
        op_lane1 = self.creature_proj(observations["opponent_lane1"]) # [bs, 3, hidden]

        lane0_emb = self.lane0_node.expand(bs, -1, -1) # [bs, 1, hidden]
        lane1_emb = self.lane1_node.expand(bs, -1, -1) # [bs, 1, hidden]

        # Stack into [bs, 25, hidden]
        # Indices:
        # 0: Player
        # 1: Opponent
        # 2: Deck
        # 3..10: Hand (8)
        # 11..13: P_Lane0 (3)
        # 14..16: P_Lane1 (3)
        # 17..19: Op_Lane0 (3)
        # 20..22: Op_Lane1 (3)
        # 23: Lane0
        # 24: Lane1
        X = th.cat([
            p, op, deck, hand, p_lane0, p_lane1, op_lane0, op_lane1, lane0_emb, lane1_emb
        ], dim=1)

        action_mask = observations["action_mask"].float()

        # Build Adjacency Matrices
        A_summon = th.zeros((bs, 25, 25), device=device)
        mask_summon = action_mask[:, 1:17].view(bs, 8, 2)
        A_summon[:, 3:11, 23:25] = mask_summon
        A_summon = A_summon + A_summon.transpose(1, 2)

        A_use = th.zeros((bs, 25, 25), device=device)
        mask_use = action_mask[:, 17:121].view(bs, 8, 13)
        A_use[:, 3:11, 1:2] = mask_use[:, :, 0:1] # Target opponent
        A_use[:, 3:11, 11:23] = mask_use[:, :, 1:13] # Target all creatures
        A_use = A_use + A_use.transpose(1, 2)

        A_attack = th.zeros((bs, 25, 25), device=device)
        mask_attack = action_mask[:, 121:145].view(bs, 6, 4)
        A_attack[:, 11:14, 1:2] = mask_attack[:, 0:3, 0:1]
        A_attack[:, 11:14, 17:20] = mask_attack[:, 0:3, 1:4]
        A_attack[:, 14:17, 1:2] = mask_attack[:, 3:6, 0:1]
        A_attack[:, 14:17, 20:23] = mask_attack[:, 3:6, 1:4]
        A_attack = A_attack + A_attack.transpose(1, 2)

        A_struct = th.zeros((bs, 25, 25), device=device)
        A_struct[:, 2:17, 0] = 1 # Deck, Hand, P_Lanes belong to Player
        A_struct[:, 17:23, 1] = 1 # Op_Lanes belong to Opponent
        A_struct[:, 11:14, 23] = 1 # P_Lane0 in Lane0
        A_struct[:, 17:20, 23] = 1 # Op_Lane0 in Lane0
        A_struct[:, 14:17, 24] = 1 # P_Lane1 in Lane1
        A_struct[:, 20:23, 24] = 1 # Op_Lane1 in Lane1
        A_struct = A_struct + A_struct.transpose(1, 2)

        A_dict = {
            "summon": A_summon,
            "use": A_use,
            "attack": A_attack,
            "struct": A_struct
        }

        # Message Passing
        for layer in self.gnn_layers:
            X = layer(X, A_dict)

        return {"all_entities": X, "action_mask": observations["action_mask"]}


class GNNLOCMNetwork(nn.Module):
    def __init__(self, features_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
        super(GNNLOCMNetwork, self).__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        hidden_dim = features_dim # features_dim is our hidden_dim

        self.pass_action = nn.Sequential(
            PMA(dim=hidden_dim, num_heads=4, num_seeds=1, ln=True),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.summon_source_card = nn.Linear(hidden_dim, hidden_dim)
        self.summon_target_lane = nn.Linear(hidden_dim, hidden_dim)
        self.summon_query = PMA(dim=hidden_dim, num_heads=4, num_seeds=1, ln=True)
        self.summon_context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.summon_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.use_source_card = nn.Linear(hidden_dim, hidden_dim)
        self.use_target_creature = nn.Linear(hidden_dim, hidden_dim)
        self.use_query = PMA(dim=hidden_dim, num_heads=4, num_seeds=1, ln=True)
        self.use_context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.use_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.attack_source_creature = nn.Linear(hidden_dim, hidden_dim)
        self.attack_target_creature = nn.Linear(hidden_dim, hidden_dim)
        self.attack_query = PMA(dim=hidden_dim, num_heads=4, num_seeds=1, ln=True)
        self.attack_context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attack_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.value_net = nn.Sequential(
            PMA(dim=hidden_dim, num_heads=4, num_seeds=1, ln=True),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf)
        )

    def forward(self, features: dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: dict[str, th.Tensor]) -> th.Tensor:
        all_entities = features["all_entities"]
        bs = all_entities.size(0)

        # Unpack nodes from the GNN output
        op = all_entities[:, 1]
        p_hand = all_entities[:, 3:11]
        p_lane0_creatures = all_entities[:, 11:14]
        p_lane1_creatures = all_entities[:, 14:17]
        op_lane0_creatures = all_entities[:, 17:20]
        op_lane1_creatures = all_entities[:, 20:23]
        lane0_node = all_entities[:, 23]
        lane1_node = all_entities[:, 24]

        # PASS logit
        pass_logit = self.pass_action(all_entities).squeeze(1)  # [bs, 1]

        # SUMMON logits
        lanes = th.stack((lane0_node, lane1_node), dim=1)  # [bs, 2, hidden_dim]
        
        summon_card = self.summon_source_card(p_hand)  # [bs, 8, hidden_dim]
        summon_lane = self.summon_target_lane(lanes)  # [bs, 2, hidden_dim]
        summon_ctx = self.summon_context_proj(self.summon_query(all_entities))  # [bs, 1, hidden_dim]
        
        summon_input = (
            summon_card[:, :, None, :] 
            + summon_lane[:, None, :, :] 
            + summon_ctx[:, None, :, :] 
        )
        
        summon_logits = self.summon_action(summon_input).squeeze(-1).view(bs, -1)  # [bs, 16]
        
        # USE logits
        targets = th.cat((
            op[:, None, :],
            p_lane0_creatures, p_lane1_creatures,
            op_lane0_creatures, op_lane1_creatures,
        ), dim=1)  # [bs, 13, hidden_dim]
        
        use_card = self.use_source_card(p_hand)
        use_target = self.use_target_creature(targets)
        use_ctx = self.use_context_proj(self.use_query(all_entities))

        use_input = (
            use_card[:, :, None, :] 
            + use_target[:, None, :, :] 
            + use_ctx[:, None, :, :]
        )
        
        use_logits = self.use_action(use_input).squeeze(-1).view(bs, -1)  # [bs, 104]

        # ATTACK lane 0 logits
        targets_l0 = th.cat((op[:, None, :], op_lane0_creatures), dim=1)  # [bs, 4, hidden_dim]
        
        attack_src_l0 = self.attack_source_creature(p_lane0_creatures)
        attack_tgt_l0 = self.attack_target_creature(targets_l0)
        attack_ctx = self.attack_context_proj(self.attack_query(all_entities))

        attack_lane0_input = (
            attack_src_l0[:, :, None, :] 
            + attack_tgt_l0[:, None, :, :] 
            + attack_ctx[:, None, :, :]
        )

        attack_lane0_logits = self.attack_action(attack_lane0_input).squeeze(-1).view(bs, -1)  # [bs, 12]
        
        # ATTACK lane 1 logits
        targets_l1 = th.cat((op[:, None, :], op_lane1_creatures), dim=1)  # [bs, 4, hidden_dim]

        attack_src_l1 = self.attack_source_creature(p_lane1_creatures)
        attack_tgt_l1 = self.attack_target_creature(targets_l1)
        
        attack_lane1_input = (
            attack_src_l1[:, :, None, :] 
            + attack_tgt_l1[:, None, :, :] 
            + attack_ctx[:, None, :, :]
        )

        attack_lane1_logits = self.attack_action(attack_lane1_input).squeeze(-1).view(bs, -1)  # [bs, 12]
        
        # Concat all logits
        action_logits = th.cat((
            pass_logit, 
            summon_logits, 
            use_logits, 
            attack_lane0_logits, 
            attack_lane1_logits
        ), dim=1)  # [bs, 145]

        action_mask = features.get("action_mask")
        if action_mask is not None:
            # action_mask is 1 for valid actions and 0 for invalid actions.
            # Mask invalid actions with -1e9.
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)

        return action_logits

    def forward_critic(self, features: dict[str, th.Tensor]) -> th.Tensor:
        return self.value_net(features["all_entities"])


class GNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(GNNActorCriticPolicy, self).__init__(*args, **kwargs)

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        self.features_extractor = th.compile(self.features_extractor)
        
        # do not add a nn.Linear layer on top of what we return at LOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNLOCMNetwork(self.features_dim, last_layer_dim_pi=145, last_layer_dim_vf=1)
        self.mlp_extractor = th.compile(self.mlp_extractor)


class GNNRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(GNNRecurrentActorCriticPolicy, self).__init__(*args, **kwargs)
        
    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        self.features_extractor = th.compile(self.features_extractor)
        
        # do not add a nn.Linear layer on top of what we return at LOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNLOCMNetwork(self.features_dim, last_layer_dim_pi=145, last_layer_dim_vf=1)
        self.mlp_extractor = th.compile(self.mlp_extractor)

    def dict_features_to_tensor(self, features: dict[str, th.Tensor]) -> th.Tensor:
        bs = features["all_entities"].size(0)
        # flatten the entities to a single vector to pass through LSTM
        return features["all_entities"].view(bs, -1)

    def tensor_features_to_dict(self, features: th.Tensor) -> dict[str, th.Tensor]:
        bs = features.size(0)
        return {"all_entities": features.view(bs, 25, -1)}

    def forward(self, obs, lstm_states, episodes_starts, deterministic=False):
        features = self.extract_features(obs)
        features_tensor = self.dict_features_to_tensor(features)
        
        if self.lstm.batch_first:
            features_tensor = features_tensor.unsqueeze(1)
        else:
            features_tensor = features_tensor.unsqueeze(0)
            
        lstm_output, lstm_states = self._process_sequence(features_tensor, lstm_states, episodes_starts, self.lstm)
        
        if self.lstm.batch_first:
            lstm_output = lstm_output.squeeze(1)
        else:
            lstm_output = lstm_output.squeeze(0)
            
        features = self.tensor_features_to_dict(lstm_output)
        
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        values = self.value_net(latent_vf)
        return actions, values, log_prob, lstm_states

    def get_distribution(self, obs, lstm_states, episodes_starts):
        features = self.extract_features(obs)
        features_tensor = self.dict_features_to_tensor(features)
        
        if self.lstm.batch_first:
            features_tensor = features_tensor.unsqueeze(1)
        else:
            features_tensor = features_tensor.unsqueeze(0)
            
        lstm_output, lstm_states = self._process_sequence(features_tensor, lstm_states, episodes_starts, self.lstm)
        
        if self.lstm.batch_first:
            lstm_output = lstm_output.squeeze(1)
        else:
            lstm_output = lstm_output.squeeze(0)
            
        features = self.tensor_features_to_dict(lstm_output)
        
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs, lstm_states, episodes_starts):
        features = self.extract_features(obs)
        features_tensor = self.dict_features_to_tensor(features)
        
        if self.lstm.batch_first:
            features_tensor = features_tensor.unsqueeze(1)
        else:
            features_tensor = features_tensor.unsqueeze(0)
            
        lstm_output, lstm_states = self._process_sequence(features_tensor, lstm_states, episodes_starts, self.lstm)
        
        if self.lstm.batch_first:
            lstm_output = lstm_output.squeeze(1)
        else:
            lstm_output = lstm_output.squeeze(0)
            
        features = self.tensor_features_to_dict(lstm_output)
        
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs, actions, lstm_states, episodes_starts):
        features = self.extract_features(obs)
        features_tensor = self.dict_features_to_tensor(features)
        
        if self.lstm.batch_first:
            features_tensor = features_tensor.unsqueeze(1)
        else:
            features_tensor = features_tensor.unsqueeze(0)
            
        lstm_output, lstm_states = self._process_sequence(features_tensor, lstm_states, episodes_starts, self.lstm)
        
        if self.lstm.batch_first:
            lstm_output = lstm_output.squeeze(1)
        else:
            lstm_output = lstm_output.squeeze(0)
            
        features = self.tensor_features_to_dict(lstm_output)
        
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

def build_gnn_network(
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
    if lstm:
        algo = RecurrentPPO
        policy = GNNRecurrentActorCriticPolicy
    else:
        algo = PPO
        policy = GNNActorCriticPolicy

    kwargs = {
        "features_extractor_class": GNNFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }

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
        tensorboard_log=tensorboard_log,
        seed=seed,
        policy_kwargs=kwargs,
    )

def load_gnn_network(load_path, lstm=False):
    if lstm:
        algo = RecurrentPPO
    else:
        algo = PPO

    kwargs = {
        "features_extractor_class": GNNFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }

    return lambda env: algo.load(load_path, env, custom_objects={"policy_kwargs": kwargs})
