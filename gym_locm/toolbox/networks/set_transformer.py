import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
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


# from https://github.com/juho-lee/set_transformer/blob/6fdae7f/modules.py
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        bs, q_len = Q.size(0), Q.size(1)
        k_len = K.size(1)

        Q_proj = self.fc_q(Q)
        K_proj, V_proj = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads

        # Reshape to 4D tensors for native SDPA: [bs, num_heads, seq_len, dim_split]
        Q_ = Q_proj.view(bs, q_len, self.num_heads, dim_split).transpose(1, 2)
        K_ = K_proj.view(bs, k_len, self.num_heads, dim_split).transpose(1, 2)
        V_ = V_proj.view(bs, k_len, self.num_heads, dim_split).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q_, K_, V_, scale=1.0 / math.sqrt(self.dim_V)
        )
        
        # Reshape back to 3D and apply residual connection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, q_len, self.dim_V)
        O = Q_proj + attn_output

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(th.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.expand(X.size(0), -1, -1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(th.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.expand(X.size(0), -1, -1), X)
#####

class SetTransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: Dict, 
        card_dim: int = 17, 
        player_dim: int = 5, 
        creature_dim: int = 17,
        card_emb_dim: int = 32,
        zone_emb_dim: int = 32,
        player_emb_dim: int = 16,
        creature_emb_dim: int = 16,
        lane_emb_dim: int = 16,
        state_emb_dim: int = 256,
    ):
        super().__init__(
            observation_space, 
            features_dim=state_emb_dim
        )  # features_dim is used to calculate LSTM input size

        self.player_embedding = nn.Sequential(
            nn.Linear(player_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        ) # 5 * 16 + 16 * 16 = 336 parameters

        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        ) # 17 * 32 + 32 * 32 = 1,568 parameters

        self.card_zone_embedding = nn.Sequential(
            SAB(32, 32, num_heads=4, ln=True),
            SAB(32, 32, num_heads=4, ln=True),
        )

        self.card_zone_pool = PMA(
            dim=32,
            num_heads=4,
            num_seeds=1,
            ln=True
        )

        self.creature_embedding = nn.Sequential(
            nn.Linear(creature_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        ) # 17 * 16 + 16 * 16 = 528 parameters
        
        self.lane_embedding = nn.Sequential(
            SAB(16, 16, num_heads=4, ln=True),
            SAB(16, 16, num_heads=4, ln=True),
        )

        self.lane_pool = PMA(
            dim=16,
            num_heads=4,
            num_seeds=1,
            ln=True
        )
        
        # 2 * 16 player + 32 hand + 32 deck + 4 * 16 lane = 160 features
        self.state_embedding = nn.Sequential(
            nn.Linear(160, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        ) # 160 * 256 + 256 * 256 = 106,496 parameters

    def forward(self, observations) -> dict[str, th.Tensor]:
        # embedding of both players
        p = self.player_embedding(observations["player_stats"])
        op = self.player_embedding(observations["opponent_stats"])
        
        p_deck_cards = observations["player_deck"]
        
        # embedding of individual deck cards
        p_deck_cards = self.card_embedding(p_deck_cards)
        p_deck_cards = self.card_zone_embedding(p_deck_cards)
        
        # embedding of the whole deck
        p_deck = self.card_zone_pool(p_deck_cards).squeeze(1)

        p_hand_cards = observations["player_hand"]

        # embedding of individual hand cards
        p_hand_cards = self.card_embedding(p_hand_cards)
        p_hand_cards = self.card_zone_embedding(p_hand_cards)
        
        # embedding of the whole hand
        p_hand = self.card_zone_pool(p_hand_cards).squeeze(1)
        
        p_lane0_creatures = observations["player_lane0"]
        
        # embedding of individual player lane 0 creatures
        p_lane0_creatures = self.creature_embedding(p_lane0_creatures)
        p_lane0_creatures = self.lane_embedding(p_lane0_creatures)

        # embedding of the whole player lane 0
        p_lane0 = self.lane_pool(p_lane0_creatures).squeeze(1)

        p_lane1_creatures = observations["player_lane1"]
        
        # embedding of individual player lane 1 creatures
        p_lane1_creatures = self.creature_embedding(p_lane1_creatures)
        p_lane1_creatures = self.lane_embedding(p_lane1_creatures)

        # embedding of the whole player lane 1
        p_lane1 = self.lane_pool(p_lane1_creatures).squeeze(1)

        op_lane0_creatures = observations["opponent_lane0"]
        
        # embedding of individual opponent lane 0 creatures
        op_lane0_creatures = self.creature_embedding(op_lane0_creatures)
        op_lane0_creatures = self.lane_embedding(op_lane0_creatures)

        # embedding of the whole opponent lane 0
        op_lane0 = self.lane_pool(op_lane0_creatures).squeeze(1)

        op_lane1_creatures = observations["opponent_lane1"]
        
        # embedding of individual opponent lane 1 creatures
        op_lane1_creatures = self.creature_embedding(op_lane1_creatures)
        op_lane1_creatures = self.lane_embedding(op_lane1_creatures)

        # embedding of the whole opponent lane 1
        op_lane1 = self.lane_pool(op_lane1_creatures).squeeze(1)
        
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
            action_mask=observations["action_mask"],
        )
        
        return embeddings


class SetTransformerLOCMNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        player_dim: int = 5,
        card_dim: int = 17,
        creature_dim: int = 17,
        last_layer_dim_pi: int = 145,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        hidden_dim = 64
        
        # input: state (256)
        self.pass_action = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, last_layer_dim_vf)
        )
        
        # input: source card (32) + target lane (16) + state (256) = 304
        self.summon_source_card = nn.Linear(32, hidden_dim)
        self.summon_target_lane = nn.Linear(16, hidden_dim)
        self.summon_state = nn.Linear(256, hidden_dim)
        
        self.summon_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # input: source card (32) + target creature (16) + state (256) = 304
        self.use_source_card = nn.Linear(32, hidden_dim)
        self.use_target_creature = nn.Linear(16, hidden_dim)
        self.use_state = nn.Linear(256, hidden_dim)
        
        self.use_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

        # input: source card (16) + target creature (16) + state (256) = 288
        self.attack_source_creature = nn.Linear(16, hidden_dim)
        self.attack_target_creature = nn.Linear(16, hidden_dim)
        self.attack_state = nn.Linear(256, hidden_dim)
        
        self.attack_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # value function head
        # input: state (256)
        self.value_net = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, last_layer_dim_vf)
        )

        self.null_target = nn.Parameter(
            th.randn(1, 1, 16) * 0.02
        )  # for the "no target" option in use and attack actions

    def forward(self, features: dict) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, embeddings: dict) -> th.Tensor:
        bs = embeddings["state"].size(0)
        null_target = self.null_target.expand(bs, -1, -1)  # [bs, 1, 16]
        
        # PASS logit
        pass_logit = self.pass_action(embeddings["state"])  # [bs, 1]

        # SUMMON logits
        hand = embeddings["hand_cards"]  # [bs, max_hand_size, card_dim]

        lanes = th.stack((embeddings["p_lane0"], embeddings["p_lane1"]), dim=1)  # [bs, 2, lane_dim]
        
        state = embeddings["state"]  # [bs, state_dim]
        
        summon_card = self.summon_source_card(hand)  # [bs, max_hand_size, hidden_dim]
        summon_lane = self.summon_target_lane(lanes)  # [bs, 2, hidden_dim]
        summon_state = self.summon_state(state)  # [bs, hidden_dim]
        
        summon_input = (
            summon_card[:, :, None, :]  # [bs, max_hand_size, 1, hidden_dim]
            + summon_lane[:, None, :, :]  # [bs, 1, 2, hidden_dim]
            + summon_state[:, None, None, :]  # [bs, 1, 1, hidden_dim]
        )
        
        summon_logits = self.summon_action(summon_input)  # [bs, max_hand_size, 2, 1]
        summon_logits = summon_logits.squeeze(-1).reshape(bs, -1)  # [bs, max_hand_size * 2]
        
        # USE logits
        hand = embeddings["hand_cards"]  # [bs, max_hand_size, card_dim]
        
        targets = th.cat((
            null_target,
            embeddings["p_lane0_creatures"], embeddings["p_lane1_creatures"],
            embeddings["op_lane0_creatures"], embeddings["op_lane1_creatures"],
        ), dim=1)  # [bs, 13, creature_dim]
        
        state = embeddings["state"]  # [bs, state_dim]
        
        use_input = (
            self.use_source_card(hand)[:, :, None, :]  # [bs, max_hand_size, 1, hidden_dim]
            + self.use_target_creature(targets)[:, None, :, :]  # [bs, 1, 13, hidden_dim]
            + self.use_state(state)[:, None, None, :]  # [bs, 1, 1, hidden_dim]
        )
        
        use_logits = self.use_action(use_input)  # [bs * 13 * max_hand_size, 1]
        use_logits = use_logits.squeeze(-1).reshape(use_logits.size(0), -1)  # [bs, max_hand_size * 13]

        # ATTACK lane 0 logits
        op_lane0_creatures = embeddings["op_lane0_creatures"]  # [bs, 3, creature_dim]
        op_lane0_creatures = th.cat((null_target, op_lane0_creatures), dim=1)  # [bs, 4, creature_dim]
        
        p_lane0_creatures = embeddings["p_lane0_creatures"]  # [bs, 3, creature_dim]
        
        state = embeddings["state"]  # [bs, state_dim]
        
        attack_lane0_input = (
            self.attack_source_creature(p_lane0_creatures)[:, :, None, :]  # [bs, 3, 1, hidden_dim]
            + self.attack_target_creature(op_lane0_creatures)[:, None, :, :]  # [bs, 1, 4, hidden_dim]
            + self.attack_state(state)[:, None, None, :]  # [bs, 1, 1, hidden_dim]
        )

        attack_lane0_logits = self.attack_action(attack_lane0_input)  # [bs, 3, 4, 1]
        attack_lane0_logits = attack_lane0_logits.squeeze(-1).reshape(bs, -1)  # [bs, 3 * 4]
        
        # ATTACK lane 1 logits
        op_lane1_creatures = embeddings["op_lane1_creatures"]  # [bs, 3, creature_dim]
        op_lane1_creatures = th.cat((null_target, op_lane1_creatures), dim=1)  # [bs, 4, creature_dim]

        p_lane1_creatures = embeddings["p_lane1_creatures"]  # [bs, 3, creature_dim]
        
        state = embeddings["state"]  # [bs, state_dim]

        attack_lane1_input = (
            self.attack_source_creature(p_lane1_creatures)[:, :, None, :]  # [bs, 3, 1, hidden_dim]
            + self.attack_target_creature(op_lane1_creatures)[:, None, :, :]  # [bs, 1, 4, hidden_dim]
            + self.attack_state(state)[:, None, None, :]  # [bs, 1, 1, hidden_dim]
        )

        attack_lane1_logits = self.attack_action(attack_lane1_input)  # [bs, 3, 4, 1]
        attack_lane1_logits = attack_lane1_logits.squeeze(-1).reshape(bs, -1)  # [bs, 3 * 4]

        # concat all action logits
        logits = th.cat((
            pass_logit,
            summon_logits,
            use_logits,
            attack_lane0_logits,
            attack_lane1_logits,
        ), dim=1)  # [bs, 145]
        action_mask = embeddings["action_mask"]
        
        # prevent invalid actions
        logits = logits.masked_fill(action_mask == 0, -1e9)
        
        return logits

    def forward_critic(self, embeddings: dict) -> th.Tensor:
        return self.value_net(embeddings["state"])


class SetTransformerRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
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
            features_extractor_class=SetTransformerFeaturesExtractor,
            features_extractor_kwargs=dict(),
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
            *args,
            **kwargs,
        )
        
    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        self.features_extractor = th.compile(self.features_extractor)
        
        # do not add a nn.Linear layer on top of what we return at BaselineLOCMNetwork
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
        self.mlp_extractor = SetTransformerLOCMNetwork(256, last_layer_dim_pi=145, last_layer_dim_vf=1)
        self.mlp_extractor = th.compile(self.mlp_extractor)

    def dict_features_to_tensor(self, features: dict[str, th.Tensor]) -> th.Tensor:
        bs = features["player"].size(0)
        
        return th.cat(list([v.view(bs, -1) for k, v in features.items() if k != "action_mask"]), dim=1)
    
    def tensor_features_to_dict(self, features: th.Tensor) -> dict[str, th.Tensor]:
        return dict(
            player=features[:, :16],  # [bs, 16]
            opponent=features[:, 16:32],  # [bs, 16]
            deck_cards=features[:, 32:992].view(-1, 30, 32),  # [bs, 30, 32]
            deck=features[:, 992:1024],  # [bs, 32]
            hand_cards=features[:, 1024:1280].view(-1, 8, 32),  # [bs, 8, 32]
            hand=features[:, 1280:1312],  # [bs, 32]
            p_lane0_creatures=features[:, 1312:1360].view(-1, 3, 16),  # [bs, 3, 16]
            p_lane0=features[:, 1360:1376],  # [bs, 16]
            p_lane1_creatures=features[:, 1376:1424].view(-1, 3, 16),  # [bs, 3, 16]
            p_lane1=features[:, 1424:1440],  # [bs, 16]
            op_lane0_creatures=features[:, 1440:1488].view(-1, 3, 16),  # [bs, 3, 16]
            op_lane0=features[:, 1488:1504],  # [bs, 16]
            op_lane1_creatures=features[:, 1504:1552].view(-1, 3, 16),  # [bs, 3, 16]
            op_lane1=features[:, 1552:1568],  # [bs, 16]
            state=features[:, 1568:1824],  # [bs, 256]
        )

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
            state_features = features.get("state")
                
            # latent_pi, latent_vf = self.mlp_extractor(features)
            latent_pi, lstm_states_pi = self._process_sequence(state_features, lstm_states.pi, episode_starts, self.lstm_actor)

            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
            
            features["state"] = latent_pi
            latent_pi = self.mlp_extractor.forward_actor(features)
            
            features["state"] = latent_vf
            latent_vf = self.mlp_extractor.forward_critic(features)

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
        state_features = features.get("state")
        latent_pi, lstm_states = self._process_sequence(state_features, lstm_states, episode_starts, self.lstm_actor)
        
        features["state"] = latent_pi
        latent_pi = self.mlp_extractor.forward_actor(features)
        
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
        state_features = features.get("state")

        # Use LSTM from the actor
        latent_pi, _ = self._process_sequence(state_features, lstm_states, episode_starts, self.lstm_actor)
        latent_vf = latent_pi.detach()

        features["state"] = latent_vf
        latent_vf = self.mlp_extractor.forward_critic(features)
        
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
        state_features = features.get("state")

        latent_pi, _ = self._process_sequence(state_features, lstm_states.pi, episode_starts, self.lstm_actor)
        latent_vf = latent_pi.detach()

        features["state"] = latent_pi
        latent_pi = self.mlp_extractor.forward_actor(features)
        
        features["state"] = latent_vf
        latent_vf = self.mlp_extractor.forward_critic(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class SetTransformerActorCriticPolicy(ActorCriticPolicy):
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
            features_extractor_class=SetTransformerFeaturesExtractor,
            features_extractor_kwargs=dict(),
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        self.features_extractor = th.compile(self.features_extractor)
        
        # do not add a nn.Linear layer on top of what we return at SetTransformerLOCMNetwork
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
        self.mlp_extractor = SetTransformerLOCMNetwork(self.features_dim, last_layer_dim_pi=145, last_layer_dim_vf=1)
        self.mlp_extractor = th.compile(self.mlp_extractor)


def build_set_transformer_network(
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
        policy = SetTransformerRecurrentActorCriticPolicy
        kwargs = dict(policy_kwargs=dict(lstm_hidden_size=lstm))
    else:
        algo = PPO
        policy = SetTransformerActorCriticPolicy
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
        tensorboard_log=tensorboard_log,
        **kwargs,
    )


def load_set_transformer_network(path, lstm=False):
    algo = RecurrentPPO if lstm else PPO
    
    def loaded_model_builder(env, seed, *args, **kwargs):
        return algo.load(path + ".zip", env=env, force_reset=True, seed=seed)

    return loaded_model_builder
