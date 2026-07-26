import torch as th
import torch.nn as nn
from typing import Tuple, List

from gym_locm.toolbox.networks.actor_critics.base import LOCMActorCriticNetwork


class TypeSpecificLOCMNetwork(LOCMActorCriticNetwork):
    def __init__(self, state_dim: int, card_emb_dim: int, creature_emb_dim: int, lane_emb_dim: int, 
                 hidden_dim: int = 64, last_layer_dim_pi: int = 145, last_layer_dim_vf: int = 1):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # input: state
        self.pass_action = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.summon_source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.summon_target_lane = nn.Linear(lane_emb_dim, hidden_dim)
        self.summon_state = nn.Linear(state_dim, hidden_dim)
        
        self.summon_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.use_source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.use_target_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.use_state = nn.Linear(state_dim, hidden_dim)
        
        self.use_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.attack_source_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.attack_target_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.attack_state = nn.Linear(state_dim, hidden_dim)
        
        self.attack_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # value function head
        # input: state
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, last_layer_dim_vf)
        )

        self.null_target = nn.Parameter(
            th.randn(1, 1, creature_emb_dim) * 0.02
        )  # for the "no target" option in use and attack actions

    def get_policy_modules(self) -> List[nn.Module]:
        return [self.pass_action, self.summon_action, self.use_action, self.attack_action]

    def get_value_modules(self) -> List[nn.Module]:
        return [self.value_net]

    def forward_actor(self, embeddings: dict) -> th.Tensor:
        bs = embeddings["state"].size(0)
        null_target = self.null_target.expand(bs, -1, -1)
        
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
        action_mask = embeddings.get("action_mask")
        
        if action_mask is not None:
            # prevent invalid actions
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        return logits

    def forward_critic(self, embeddings: dict) -> th.Tensor:
        return self.value_net(embeddings["state"])
