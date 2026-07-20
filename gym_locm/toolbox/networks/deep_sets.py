import torch as th
from gym_locm.toolbox.networks.utils import safely_compile
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

class DeepSetsFeaturesExtractor(BaseFeaturesExtractor):
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

    def forward(self, observations) -> dict[str, th.Tensor]:
        # embedding of both players
        p = self.player_embedding(observations["player_stats"])
        op = self.player_embedding(observations["opponent_stats"])
        
        p_deck_cards = observations["player_deck"]
        
        # embedding of individual deck cards
        p_deck_cards = self.card_embedding(p_deck_cards)
        
        # embedding of the whole deck
        p_deck = p_deck_cards.sum(dim=1)
        p_deck = self.card_zone_embedding(p_deck)

        p_hand_cards = observations["player_hand"]

        # embedding of individual hand cards
        p_hand_cards = self.card_embedding(p_hand_cards)
        
        # embedding of the whole hand
        p_hand = p_hand_cards.sum(dim=1)
        p_hand = self.card_zone_embedding(p_hand)
        
        p_lane0_creatures = observations["player_lane0"]
        
        # embedding of individual player lane 0 creatures
        p_lane0_creatures = self.creature_embedding(p_lane0_creatures)

        # embedding of the whole player lane 0
        p_lane0 = p_lane0_creatures.sum(dim=1)
        p_lane0 = self.lane_embedding(p_lane0)

        p_lane1_creatures = observations["player_lane1"]
        
        # embedding of individual player lane 1 creatures
        p_lane1_creatures = self.creature_embedding(p_lane1_creatures)

        # embedding of the whole player lane 1
        p_lane1 = p_lane1_creatures.sum(dim=1)
        p_lane1 = self.lane_embedding(p_lane1)

        op_lane0_creatures = observations["opponent_lane0"]
        
        # embedding of individual opponent lane 0 creatures
        op_lane0_creatures = self.creature_embedding(op_lane0_creatures)

        # embedding of the whole opponent lane 0
        op_lane0 = op_lane0_creatures.sum(dim=1)
        op_lane0 = self.lane_embedding(op_lane0)

        op_lane1_creatures = observations["opponent_lane1"]
        
        # embedding of individual opponent lane 1 creatures
        op_lane1_creatures = self.creature_embedding(op_lane1_creatures)

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
            action_mask=observations["action_mask"],
        )
        
        return embeddings


class DeepSetsLOCMNetwork(nn.Module):
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
        
        hidden_dim = 64
        
        # PASS action head
        # input: state (256)
        self.pass_action = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # SOURCE head
        self.source_state = nn.Linear(256, hidden_dim)
        self.source_card = nn.Linear(32, hidden_dim)
        self.source_creature = nn.Linear(16, hidden_dim)
        
        self.source_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # TARGET head
        self.target_state = nn.Linear(256, hidden_dim)
        self.target_lane = nn.Linear(16, hidden_dim)
        self.target_creature = nn.Linear(16, hidden_dim)
        
        self.target_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Low-rank bilinear interaction between source and target
        interaction_dim = 8
        self.source_interaction = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.target_interaction = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.interaction_scale = interaction_dim ** -0.5
        
        # Action-context biases
        self.summon_bias = nn.Parameter(th.zeros(1))
        self.use_bias = nn.Parameter(th.zeros(1))
        self.attack_bias = nn.Parameter(th.zeros(1))
        
        # value function head
        # input: state (256)
        self.value_net = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, last_layer_dim_vf)
        )

        self.null_use_target = nn.Parameter(
            th.randn(1, 1, 16) * 0.02
        )  # for the "no target" option in use actions
        
        self.null_attack_target = nn.Parameter(
            th.randn(1, 1, 16) * 0.02
        )  # for the "attack opponent directly" option in attack actions

    def forward(self, features: dict) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, embeddings: dict) -> th.Tensor:
        bs = embeddings["state"].size(0)
        state = embeddings["state"]  # [bs, 256]
        
        # PASS logit
        pass_logit = self.pass_action(state)  # [bs, 1]

        # Source hidden embeddings and logits
        hand = embeddings["hand_cards"]  # [bs, 8, 32]
        p_lane0_creatures = embeddings["p_lane0_creatures"]  # [bs, 3, 16]
        p_lane1_creatures = embeddings["p_lane1_creatures"]  # [bs, 3, 16]
        
        source_state_emb = self.source_state(state)[:, None, :]  # [bs, 1, hidden_dim]
        
        source_hand_emb = self.source_card(hand) + source_state_emb  # [bs, 8, hidden_dim]
        source_hand_logits = self.source_action(source_hand_emb).squeeze(-1)  # [bs, 8]
        
        source_p_lane0_emb = self.source_creature(p_lane0_creatures) + source_state_emb  # [bs, 3, hidden_dim]
        source_p_lane0_logits = self.source_action(source_p_lane0_emb).squeeze(-1)  # [bs, 3]
        
        source_p_lane1_emb = self.source_creature(p_lane1_creatures) + source_state_emb  # [bs, 3, hidden_dim]
        source_p_lane1_logits = self.source_action(source_p_lane1_emb).squeeze(-1)  # [bs, 3]
        
        # Target hidden embeddings and logits
        lanes = th.stack((embeddings["p_lane0"], embeddings["p_lane1"]), dim=1)  # [bs, 2, 16]
        
        op_lane0_creatures = embeddings["op_lane0_creatures"]  # [bs, 3, 16]
        op_lane1_creatures = embeddings["op_lane1_creatures"]  # [bs, 3, 16]
        
        null_use_target = self.null_use_target.expand(bs, -1, -1)  # [bs, 1, 16]
        null_attack_target = self.null_attack_target.expand(bs, -1, -1)  # [bs, 1, 16]
        
        target_state_emb = self.target_state(state)[:, None, :]  # [bs, 1, hidden_dim]
        
        target_lane_emb = self.target_lane(lanes) + target_state_emb  # [bs, 2, hidden_dim]
        target_lane_logits = self.target_action(target_lane_emb).squeeze(-1)  # [bs, 2]
        
        target_null_use_emb = self.target_creature(null_use_target) + target_state_emb  # [bs, 1, hidden_dim]
        target_null_use_logits = self.target_action(target_null_use_emb).squeeze(-1)  # [bs, 1]
        
        target_null_attack_emb = self.target_creature(null_attack_target) + target_state_emb  # [bs, 1, hidden_dim]
        target_null_attack_logits = self.target_action(target_null_attack_emb).squeeze(-1)  # [bs, 1]
        
        target_p_lane0_emb = self.target_creature(p_lane0_creatures) + target_state_emb  # [bs, 3, hidden_dim]
        target_p_lane0_logits = self.target_action(target_p_lane0_emb).squeeze(-1)  # [bs, 3]
        
        target_p_lane1_emb = self.target_creature(p_lane1_creatures) + target_state_emb  # [bs, 3, hidden_dim]
        target_p_lane1_logits = self.target_action(target_p_lane1_emb).squeeze(-1)  # [bs, 3]
        
        target_op_lane0_emb = self.target_creature(op_lane0_creatures) + target_state_emb  # [bs, 3, hidden_dim]
        target_op_lane0_logits = self.target_action(target_op_lane0_emb).squeeze(-1)  # [bs, 3]
        
        target_op_lane1_emb = self.target_creature(op_lane1_creatures) + target_state_emb  # [bs, 3, hidden_dim]
        target_op_lane1_logits = self.target_action(target_op_lane1_emb).squeeze(-1)  # [bs, 3]
        
        # Source and target interaction projections
        src_inter_hand = self.source_interaction(source_hand_emb)  # [bs, 8, inter_dim]
        src_inter_p_lane0 = self.source_interaction(source_p_lane0_emb)  # [bs, 3, inter_dim]
        src_inter_p_lane1 = self.source_interaction(source_p_lane1_emb)  # [bs, 3, inter_dim]
        
        # Action Logits Assembly
        # 1. SUMMON logits (source: hand, target: lane)
        tgt_inter_lane = self.target_interaction(target_lane_emb)  # [bs, 2, inter_dim]
        summon_interaction = th.bmm(
            src_inter_hand, tgt_inter_lane.transpose(1, 2)
        ) * self.interaction_scale  # [bs, 8, 2]
        
        summon_logits = (
            source_hand_logits[:, :, None] 
            + target_lane_logits[:, None, :]
            + summon_interaction
            + self.summon_bias
        ).reshape(bs, -1)  # [bs, 16]
        
        # 2. USE logits (source: hand, target: null_use, p_lane0, p_lane1, op_lane0, op_lane1)
        use_target_embs = th.cat((
            target_null_use_emb, target_p_lane0_emb, target_p_lane1_emb,
            target_op_lane0_emb, target_op_lane1_emb,
        ), dim=1)  # [bs, 13, hidden_dim]
        
        use_targets_logits = th.cat((
            target_null_use_logits,
            target_p_lane0_logits,
            target_p_lane1_logits,
            target_op_lane0_logits,
            target_op_lane1_logits,
        ), dim=1)  # [bs, 13]
        
        tgt_inter_use = self.target_interaction(use_target_embs)  # [bs, 13, inter_dim]
        use_interaction = th.bmm(
            src_inter_hand, tgt_inter_use.transpose(1, 2)
        ) * self.interaction_scale  # [bs, 8, 13]
        
        use_logits = (
            source_hand_logits[:, :, None]
            + use_targets_logits[:, None, :]
            + use_interaction
            + self.use_bias
        ).reshape(bs, -1)  # [bs, 104]
        
        # 3. ATTACK lane 0 logits (source: p_lane0, target: null_attack, op_lane0)
        attack_lane0_target_embs = th.cat((
            target_null_attack_emb, target_op_lane0_emb,
        ), dim=1)  # [bs, 4, hidden_dim]
        
        attack_lane0_targets_logits = th.cat((
            target_null_attack_logits,
            target_op_lane0_logits,
        ), dim=1)  # [bs, 4]
        
        tgt_inter_attack_l0 = self.target_interaction(attack_lane0_target_embs)  # [bs, 4, inter_dim]
        attack_lane0_interaction = th.bmm(
            src_inter_p_lane0, tgt_inter_attack_l0.transpose(1, 2)
        ) * self.interaction_scale  # [bs, 3, 4]
        
        attack_lane0_logits = (
            source_p_lane0_logits[:, :, None]
            + attack_lane0_targets_logits[:, None, :]
            + attack_lane0_interaction
            + self.attack_bias
        ).reshape(bs, -1)  # [bs, 12]
        
        # 4. ATTACK lane 1 logits (source: p_lane1, target: null_attack, op_lane1)
        attack_lane1_target_embs = th.cat((
            target_null_attack_emb, target_op_lane1_emb,
        ), dim=1)  # [bs, 4, hidden_dim]
        
        attack_lane1_targets_logits = th.cat((
            target_null_attack_logits,
            target_op_lane1_logits,
        ), dim=1)  # [bs, 4]
        
        tgt_inter_attack_l1 = self.target_interaction(attack_lane1_target_embs)  # [bs, 4, inter_dim]
        attack_lane1_interaction = th.bmm(
            src_inter_p_lane1, tgt_inter_attack_l1.transpose(1, 2)
        ) * self.interaction_scale  # [bs, 3, 4]
        
        attack_lane1_logits = (
            source_p_lane1_logits[:, :, None]
            + attack_lane1_targets_logits[:, None, :]
            + attack_lane1_interaction
            + self.attack_bias
        ).reshape(bs, -1)  # [bs, 12]
        
        # Concat all action logits
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



class DeepSetsRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
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
            features_extractor_class=DeepSetsFeaturesExtractor,
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
        self.features_extractor = safely_compile(self.features_extractor)
        
        # do not add a nn.Linear layer on top of what we return at BaselineLOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # initialize weights of the new output layers (which are not initialized by the base class)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor.pass_action: 0.01,
                self.mlp_extractor.source_action: 0.01,
                self.mlp_extractor.target_action: 0.01,
                self.mlp_extractor.value_net: 1,
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DeepSetsLOCMNetwork(256, last_layer_dim_pi=145, last_layer_dim_vf=1)
        self.mlp_extractor = safely_compile(self.mlp_extractor)

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


class DeepSetsActorCriticPolicy(ActorCriticPolicy):
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
            features_extractor_class=DeepSetsFeaturesExtractor,
            features_extractor_kwargs=dict(),
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        self.features_extractor = safely_compile(self.features_extractor)
        
        # do not add a nn.Linear layer on top of what we return at DeepSetsLOCMNetwork
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # initialize weights of the new output layers (which are not initialized by the base class)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor.pass_action: 0.01,
                self.mlp_extractor.source_action: 0.01,
                self.mlp_extractor.target_action: 0.01,
                self.mlp_extractor.value_net: 1,
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DeepSetsLOCMNetwork(self.features_dim, last_layer_dim_pi=145, last_layer_dim_vf=1)
        self.mlp_extractor = safely_compile(self.mlp_extractor)


def build_deep_sets_network(
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
        policy = DeepSetsRecurrentActorCriticPolicy
        kwargs = dict(policy_kwargs=dict(lstm_hidden_size=lstm))
    else:
        algo = PPO
        policy = DeepSetsActorCriticPolicy
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


def load_deep_sets_network(path, lstm=False):
    algo = RecurrentPPO if lstm else PPO
    
    def loaded_model_builder(env, seed, *args, **kwargs):
        return algo.load(path + ".zip", env=env, force_reset=True, seed=seed)

    return loaded_model_builder
