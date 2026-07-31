"""
Bilinear actor-critic for LOCM.

Decomposes action logits into independent source and target scores,
coupled through a low-rank bilinear interaction term:

    logit(a) = s_src(src) + s_tgt(tgt)
             + (1/√d) · (W_src · e_src)ᵀ(W_tgt · e_tgt)
             + bias_action_type

The bilinear interaction models pairwise source-target compatibility
with minimal parameter overhead. If the interaction is not useful,
the network can learn near-zero interaction weights, effectively
reducing to a factored (independent) decomposition.
"""

import torch as th
import torch.nn as nn

from gym_locm.toolbox.networks.actor_critics.base import LOCMActorCriticNetwork


class BilinearLOCMNetwork(LOCMActorCriticNetwork):
    """Bilinear source-target interaction actor-critic.

    Computes action logits as the sum of independent source and target scores
    plus a low-rank bilinear interaction term between source and target entities.
    """

    def __init__(self, state_dim, card_emb_dim, creature_emb_dim, lane_emb_dim,
                 hidden_dim=64, interaction_dim=8, last_layer_dim_pi=145, last_layer_dim_vf=1):
        super().__init__(last_layer_dim_pi, last_layer_dim_vf)

        # PASS action head
        # input: state (state_dim)
        self.pass_action = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # SOURCE head
        self.source_state = nn.Linear(state_dim, hidden_dim)
        self.source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.source_creature = nn.Linear(creature_emb_dim, hidden_dim)

        self.source_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # TARGET head
        self.target_state = nn.Linear(state_dim, hidden_dim)
        self.target_lane = nn.Linear(lane_emb_dim, hidden_dim)
        self.target_creature = nn.Linear(creature_emb_dim, hidden_dim)

        self.target_action = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Low-rank bilinear interaction between source and target
        self.source_interaction = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.target_interaction = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.interaction_scale = interaction_dim ** -0.5

        # Action-context biases
        self.summon_bias = nn.Parameter(th.zeros(1))
        self.use_bias = nn.Parameter(th.zeros(1))
        self.attack_bias = nn.Parameter(th.zeros(1))

        # value function head
        # input: state (state_dim)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, last_layer_dim_vf)
        )

        self.null_use_target = nn.Parameter(
            th.randn(1, 1, creature_emb_dim) * 0.02
        )  # for the "no target" option in use actions

        self.null_attack_target = nn.Parameter(
            th.randn(1, 1, creature_emb_dim) * 0.02
        )  # for the "attack opponent directly" option in attack actions

    def forward_actor(self, embeddings: dict) -> th.Tensor:
        bs = embeddings["state"].size(0)
        state = embeddings["state"]

        # PASS logit
        pass_logit = self.pass_action(state)

        # Source hidden embeddings and logits
        hand = embeddings["hand_cards"]
        p_lane0_creatures = embeddings["p_lane0_creatures"]
        p_lane1_creatures = embeddings["p_lane1_creatures"]

        source_state_emb = self.source_state(state)[:, None, :]

        source_hand_emb = self.source_card(hand) + source_state_emb
        source_hand_logits = self.source_action(source_hand_emb).squeeze(-1)

        source_p_lane0_emb = self.source_creature(p_lane0_creatures) + source_state_emb
        source_p_lane0_logits = self.source_action(source_p_lane0_emb).squeeze(-1)

        source_p_lane1_emb = self.source_creature(p_lane1_creatures) + source_state_emb
        source_p_lane1_logits = self.source_action(source_p_lane1_emb).squeeze(-1)

        # Target hidden embeddings and logits
        lanes = th.stack((embeddings["p_lane0"], embeddings["p_lane1"]), dim=1)

        op_lane0_creatures = embeddings["op_lane0_creatures"]
        op_lane1_creatures = embeddings["op_lane1_creatures"]

        null_use_target = self.null_use_target.expand(bs, -1, -1)
        null_attack_target = self.null_attack_target.expand(bs, -1, -1)

        target_state_emb = self.target_state(state)[:, None, :]

        target_lane_emb = self.target_lane(lanes) + target_state_emb
        target_lane_logits = self.target_action(target_lane_emb).squeeze(-1)

        target_null_use_emb = self.target_creature(null_use_target) + target_state_emb
        target_null_use_logits = self.target_action(target_null_use_emb).squeeze(-1)

        target_null_attack_emb = self.target_creature(null_attack_target) + target_state_emb
        target_null_attack_logits = self.target_action(target_null_attack_emb).squeeze(-1)

        target_p_lane0_emb = self.target_creature(p_lane0_creatures) + target_state_emb
        target_p_lane0_logits = self.target_action(target_p_lane0_emb).squeeze(-1)

        target_p_lane1_emb = self.target_creature(p_lane1_creatures) + target_state_emb
        target_p_lane1_logits = self.target_action(target_p_lane1_emb).squeeze(-1)

        target_op_lane0_emb = self.target_creature(op_lane0_creatures) + target_state_emb
        target_op_lane0_logits = self.target_action(target_op_lane0_emb).squeeze(-1)

        target_op_lane1_emb = self.target_creature(op_lane1_creatures) + target_state_emb
        target_op_lane1_logits = self.target_action(target_op_lane1_emb).squeeze(-1)

        # Source and target interaction projections
        src_inter_hand = self.source_interaction(source_hand_emb)
        src_inter_p_lane0 = self.source_interaction(source_p_lane0_emb)
        src_inter_p_lane1 = self.source_interaction(source_p_lane1_emb)

        # Action Logits Assembly
        # 1. SUMMON logits (source: hand, target: lane)
        tgt_inter_lane = self.target_interaction(target_lane_emb)
        summon_interaction = th.bmm(
            src_inter_hand, tgt_inter_lane.transpose(1, 2)
        ) * self.interaction_scale

        summon_logits = (
            source_hand_logits[:, :, None]
            + target_lane_logits[:, None, :]
            + summon_interaction
            + self.summon_bias
        ).reshape(bs, -1)

        # 2. USE logits (source: hand, target: null_use, p_lane0, p_lane1, op_lane0, op_lane1)
        use_target_embs = th.cat((
            target_null_use_emb, target_p_lane0_emb, target_p_lane1_emb,
            target_op_lane0_emb, target_op_lane1_emb,
        ), dim=1)

        use_targets_logits = th.cat((
            target_null_use_logits,
            target_p_lane0_logits,
            target_p_lane1_logits,
            target_op_lane0_logits,
            target_op_lane1_logits,
        ), dim=1)

        tgt_inter_use = self.target_interaction(use_target_embs)
        use_interaction = th.bmm(
            src_inter_hand, tgt_inter_use.transpose(1, 2)
        ) * self.interaction_scale

        use_logits = (
            source_hand_logits[:, :, None]
            + use_targets_logits[:, None, :]
            + use_interaction
            + self.use_bias
        ).reshape(bs, -1)

        # 3. ATTACK lane 0 logits (source: p_lane0, target: null_attack, op_lane0)
        attack_lane0_target_embs = th.cat((
            target_null_attack_emb, target_op_lane0_emb,
        ), dim=1)

        attack_lane0_targets_logits = th.cat((
            target_null_attack_logits,
            target_op_lane0_logits,
        ), dim=1)

        tgt_inter_attack_l0 = self.target_interaction(attack_lane0_target_embs)
        attack_lane0_interaction = th.bmm(
            src_inter_p_lane0, tgt_inter_attack_l0.transpose(1, 2)
        ) * self.interaction_scale

        attack_lane0_logits = (
            source_p_lane0_logits[:, :, None]
            + attack_lane0_targets_logits[:, None, :]
            + attack_lane0_interaction
            + self.attack_bias
        ).reshape(bs, -1)

        # 4. ATTACK lane 1 logits (source: p_lane1, target: null_attack, op_lane1)
        attack_lane1_target_embs = th.cat((
            target_null_attack_emb, target_op_lane1_emb,
        ), dim=1)

        attack_lane1_targets_logits = th.cat((
            target_null_attack_logits,
            target_op_lane1_logits,
        ), dim=1)

        tgt_inter_attack_l1 = self.target_interaction(attack_lane1_target_embs)
        attack_lane1_interaction = th.bmm(
            src_inter_p_lane1, tgt_inter_attack_l1.transpose(1, 2)
        ) * self.interaction_scale

        attack_lane1_logits = (
            source_p_lane1_logits[:, :, None]
            + attack_lane1_targets_logits[:, None, :]
            + attack_lane1_interaction
            + self.attack_bias
        ).reshape(bs, -1)

        # Concat all action logits
        logits = th.cat((
            pass_logit,
            summon_logits,
            use_logits,
            attack_lane0_logits,
            attack_lane1_logits,
        ), dim=1)

        action_mask = embeddings["action_mask"]

        # prevent invalid actions
        logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def forward_critic(self, embeddings: dict) -> th.Tensor:
        return self.value_net(embeddings["state"])

    def get_policy_modules(self) -> list[nn.Module]:
        return [self.pass_action[-1], self.source_action[-1], self.target_action[-1]]

    def get_value_modules(self) -> list[nn.Module]:
        return [self.value_net[-1]]
