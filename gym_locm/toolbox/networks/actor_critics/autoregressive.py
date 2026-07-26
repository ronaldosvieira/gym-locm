"""
Autoregressive actor-critic for LOCM.

Inspired by the Hearthstone paper (arXiv 2303.05197), this decomposes actions
into source entity selection followed by target entity selection:

    logit(action) = f_source(src) + f_target(tgt | src)

In LOCM, the action type is deterministic given the source entity:
- Hand card (creature) -> SUMMON
- Hand card (item) -> USE  
- Friendly creature on board -> ATTACK
- No source -> PASS

So we skip explicit action-type selection and decompose into:
    π(a|s) ≈ softmax(f_src(src) + f_tgt(tgt | src, state))

Source entities: {PASS} ∪ {8 hand cards} ∪ {3 p_lane0 creatures} ∪ {3 p_lane1 creatures}
Target entities (conditioned on source type):
    - PASS: no target (terminal action)
    - SUMMON: {lane0, lane1} = 2 targets  
    - USE: {null, 3 p_lane0, 3 p_lane1, 3 op_lane0, 3 op_lane1} = 13 targets
    - ATTACK L0: {opponent_direct, 3 op_lane0} = 4 targets
    - ATTACK L1: {opponent_direct, 3 op_lane1} = 4 targets

The full 145-dim logit vector is computed in a SINGLE forward pass — no
sequential autoregressive sampling during training. The factored structure
provides source-conditioned target scoring while maintaining compatibility
with SB3's standard PPO and action masking.
"""

import torch as th
import torch.nn as nn
from typing import Tuple

from gym_locm.toolbox.networks.actor_critics.base import LOCMActorCriticNetwork


class AutoregressiveLOCMNetwork(LOCMActorCriticNetwork):
    """
    Autoregressive action decomposition: source selection + conditioned target selection.
    
    Computes logit(action_i) = source_score(src_i) + target_score(tgt_i | src_i, state)
    """

    def __init__(
        self,
        state_dim: int,
        card_emb_dim: int,
        creature_emb_dim: int,
        lane_emb_dim: int,
        hidden_dim: int = 64,
        last_layer_dim_pi: int = 145,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__(last_layer_dim_pi, last_layer_dim_vf)

        # ── Source scoring ──
        # Projects each source entity + state context into a scalar source score
        self.source_state_proj = nn.Linear(state_dim, hidden_dim)
        self.source_card_proj = nn.Linear(card_emb_dim, hidden_dim)
        self.source_creature_proj = nn.Linear(creature_emb_dim, hidden_dim)
        self.source_score = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # PASS source: dedicated head since PASS has no target
        self.pass_source = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Source embedding for conditioning ──
        # Produces a hidden vector per source entity used to condition target scoring
        self.source_emb_state_proj = nn.Linear(state_dim, hidden_dim)
        self.source_emb_card_proj = nn.Linear(card_emb_dim, hidden_dim)
        self.source_emb_creature_proj = nn.Linear(creature_emb_dim, hidden_dim)

        # ── Target scoring (conditioned on source) ──
        # target_score = f(target_proj + state_proj + source_conditioning_proj)
        self.target_state_proj = nn.Linear(state_dim, hidden_dim)
        self.target_lane_proj = nn.Linear(lane_emb_dim, hidden_dim)
        self.target_creature_proj = nn.Linear(creature_emb_dim, hidden_dim)
        self.target_source_cond = nn.Linear(hidden_dim, hidden_dim)  # conditions on source emb
        self.target_score = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable null target embeddings
        self.null_use_target = nn.Parameter(
            th.randn(1, 1, creature_emb_dim) * 0.02
        )  # "no target" for USE actions
        self.null_attack_target = nn.Parameter(
            th.randn(1, 1, creature_emb_dim) * 0.02
        )  # "attack opponent directly" for ATTACK actions

        # ── Value function head ──
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, last_layer_dim_vf),
        )

    def forward_actor(self, embeddings: dict) -> th.Tensor:
        bs = embeddings["state"].size(0)
        state = embeddings["state"]  # [bs, state_dim]

        hand_cards = embeddings["hand_cards"]  # [bs, 8, card_emb_dim]
        p_lane0_creatures = embeddings["p_lane0_creatures"]  # [bs, 3, creature_emb_dim]
        p_lane1_creatures = embeddings["p_lane1_creatures"]  # [bs, 3, creature_emb_dim]
        op_lane0_creatures = embeddings["op_lane0_creatures"]  # [bs, 3, creature_emb_dim]
        op_lane1_creatures = embeddings["op_lane1_creatures"]  # [bs, 3, creature_emb_dim]
        p_lane0 = embeddings["p_lane0"]  # [bs, lane_emb_dim]
        p_lane1 = embeddings["p_lane1"]  # [bs, lane_emb_dim]

        # ── Source scores ──
        state_ctx = self.source_state_proj(state)[:, None, :]  # [bs, 1, hidden]

        # Hand card source scores
        hand_src_hidden = self.source_card_proj(hand_cards) + state_ctx  # [bs, 8, hidden]
        hand_src_scores = self.source_score(hand_src_hidden).squeeze(-1)  # [bs, 8]

        # Lane creature source scores
        p_l0_src_hidden = self.source_creature_proj(p_lane0_creatures) + state_ctx  # [bs, 3, hidden]
        p_l0_src_scores = self.source_score(p_l0_src_hidden).squeeze(-1)  # [bs, 3]

        p_l1_src_hidden = self.source_creature_proj(p_lane1_creatures) + state_ctx  # [bs, 3, hidden]
        p_l1_src_scores = self.source_score(p_l1_src_hidden).squeeze(-1)  # [bs, 3]

        # PASS source score
        pass_score = self.pass_source(state)  # [bs, 1]

        # ── Source embeddings for conditioning ──
        emb_state_ctx = self.source_emb_state_proj(state)[:, None, :]  # [bs, 1, hidden]

        hand_src_emb = self.source_emb_card_proj(hand_cards) + emb_state_ctx  # [bs, 8, hidden]
        p_l0_src_emb = self.source_emb_creature_proj(p_lane0_creatures) + emb_state_ctx  # [bs, 3, hidden]
        p_l1_src_emb = self.source_emb_creature_proj(p_lane1_creatures) + emb_state_ctx  # [bs, 3, hidden]

        # ── Target scoring context ──
        target_state_ctx = self.target_state_proj(state)[:, None, :]  # [bs, 1, hidden]

        # ── SUMMON target logits (hand_card -> lane) ──
        # Each hand card can be summoned to lane 0 or lane 1
        lanes = th.stack((p_lane0, p_lane1), dim=1)  # [bs, 2, lane_emb_dim]
        target_lane_hidden = self.target_lane_proj(lanes)  # [bs, 2, hidden]

        # Source conditioning: each hand card conditions the target scoring
        summon_src_cond = self.target_source_cond(hand_src_emb)  # [bs, 8, hidden]

        summon_target_input = (
            target_lane_hidden[:, None, :, :]  # [bs, 1, 2, hidden]
            + target_state_ctx[:, None, :, :]   # [bs, 1, 1, hidden]
            + summon_src_cond[:, :, None, :]    # [bs, 8, 1, hidden]
        )  # [bs, 8, 2, hidden]
        summon_target_scores = self.target_score(summon_target_input).squeeze(-1)  # [bs, 8, 2]

        # Combine: source_score + target_score
        summon_logits = (
            hand_src_scores[:, :, None]  # [bs, 8, 1]
            + summon_target_scores       # [bs, 8, 2]
        ).reshape(bs, -1)  # [bs, 16]

        # ── USE target logits (hand_card -> creature/null) ──
        null_use = self.null_use_target.expand(bs, -1, -1)  # [bs, 1, creature_emb_dim]
        use_targets = th.cat((
            null_use,
            p_lane0_creatures, p_lane1_creatures,
            op_lane0_creatures, op_lane1_creatures,
        ), dim=1)  # [bs, 13, creature_emb_dim]
        target_use_hidden = self.target_creature_proj(use_targets)  # [bs, 13, hidden]

        use_src_cond = self.target_source_cond(hand_src_emb)  # [bs, 8, hidden]

        use_target_input = (
            target_use_hidden[:, None, :, :]  # [bs, 1, 13, hidden]
            + target_state_ctx[:, None, :, :]  # [bs, 1, 1, hidden]
            + use_src_cond[:, :, None, :]      # [bs, 8, 1, hidden]
        )  # [bs, 8, 13, hidden]
        use_target_scores = self.target_score(use_target_input).squeeze(-1)  # [bs, 8, 13]

        use_logits = (
            hand_src_scores[:, :, None]  # [bs, 8, 1]
            + use_target_scores          # [bs, 8, 13]
        ).reshape(bs, -1)  # [bs, 104]

        # ── ATTACK lane 0 target logits (p_lane0_creature -> op_lane0/opponent) ──
        null_attack = self.null_attack_target.expand(bs, -1, -1)  # [bs, 1, creature_emb_dim]

        attack_l0_targets = th.cat((null_attack, op_lane0_creatures), dim=1)  # [bs, 4, creature_emb_dim]
        target_atk_l0_hidden = self.target_creature_proj(attack_l0_targets)  # [bs, 4, hidden]

        atk_l0_src_cond = self.target_source_cond(p_l0_src_emb)  # [bs, 3, hidden]

        atk_l0_target_input = (
            target_atk_l0_hidden[:, None, :, :]  # [bs, 1, 4, hidden]
            + target_state_ctx[:, None, :, :]      # [bs, 1, 1, hidden]
            + atk_l0_src_cond[:, :, None, :]       # [bs, 3, 1, hidden]
        )  # [bs, 3, 4, hidden]
        atk_l0_target_scores = self.target_score(atk_l0_target_input).squeeze(-1)  # [bs, 3, 4]

        attack_l0_logits = (
            p_l0_src_scores[:, :, None]  # [bs, 3, 1]
            + atk_l0_target_scores       # [bs, 3, 4]
        ).reshape(bs, -1)  # [bs, 12]

        # ── ATTACK lane 1 target logits (p_lane1_creature -> op_lane1/opponent) ──
        attack_l1_targets = th.cat((null_attack, op_lane1_creatures), dim=1)  # [bs, 4, creature_emb_dim]
        target_atk_l1_hidden = self.target_creature_proj(attack_l1_targets)  # [bs, 4, hidden]

        atk_l1_src_cond = self.target_source_cond(p_l1_src_emb)  # [bs, 3, hidden]

        atk_l1_target_input = (
            target_atk_l1_hidden[:, None, :, :]  # [bs, 1, 4, hidden]
            + target_state_ctx[:, None, :, :]      # [bs, 1, 1, hidden]
            + atk_l1_src_cond[:, :, None, :]       # [bs, 3, 1, hidden]
        )  # [bs, 3, 4, hidden]
        atk_l1_target_scores = self.target_score(atk_l1_target_input).squeeze(-1)  # [bs, 3, 4]

        attack_l1_logits = (
            p_l1_src_scores[:, :, None]  # [bs, 3, 1]
            + atk_l1_target_scores       # [bs, 3, 4]
        ).reshape(bs, -1)  # [bs, 12]

        # ── Concat all action logits ──
        logits = th.cat((
            pass_score,        # [bs, 1]
            summon_logits,     # [bs, 16]
            use_logits,        # [bs, 104]
            attack_l0_logits,  # [bs, 12]
            attack_l1_logits,  # [bs, 12]
        ), dim=1)  # [bs, 145]

        action_mask = embeddings["action_mask"]
        logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def forward_critic(self, embeddings: dict) -> th.Tensor:
        return self.value_net(embeddings["state"])

    def get_policy_modules(self) -> list[nn.Module]:
        return [self.source_score, self.target_score, self.pass_source]

    def get_value_modules(self) -> list[nn.Module]:
        return [self.value_net]
