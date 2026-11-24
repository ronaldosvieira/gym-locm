"""
Typed auto-regressive actor-critic for LOCM.

Decomposes the flat Discrete(145) action space into three sequential decisions
with separate source and target scoring networks for each action type:

    π(a|s) = π(type|s) · π_type(src|s) · π_type(tgt|src,s)

Action types:
    0 = PASS (terminal, no source/target)
    1 = SUMMON (source: hand card, target: lane)
    2 = USE (source: hand card, target: creature or null)
    3 = ATTACK (source: board creature, target: opponent creature or face)

Flat action layout:
    [0]:       PASS
    [1..16]:   SUMMON = 1 + hand_idx*2 + lane_idx
    [17..120]: USE    = 17 + hand_idx*13 + target_idx
    [121..144]: ATTACK = 121 + creature_idx*4 + target_idx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_locm.toolbox.networks.actor_critics.base import LOCMActorCriticNetwork


def _build_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, output_dim),
    )


def _build_head(hidden_dim, output_dim=1):
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, output_dim),
    )


class TypedLOCMNetwork(LOCMActorCriticNetwork):
    """
    Auto-regressive action decomposition actor-critic head with separate
    per-action-type source and target networks.

    Decomposes the flat 145 action space into 3 sequential decisions:
    1. Action TYPE (PASS=0, SUMMON=1, USE=2, ATTACK=3)
    2. SOURCE entity (scored by the chosen action type's source head)
    3. TARGET entity (scored by the chosen action type's target head conditioned on source)
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
        super().__init__(
            last_layer_dim_pi=last_layer_dim_pi,
            last_layer_dim_vf=last_layer_dim_vf,
        )

        # TYPE HEAD
        self.type_head = _build_mlp(state_dim, hidden_dim, 4)

        # SOURCE HEADS (per action type)
        self.summon_source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.summon_source_state = nn.Linear(state_dim, hidden_dim)
        self.summon_source = _build_head(hidden_dim, 1)

        self.use_source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.use_source_state = nn.Linear(state_dim, hidden_dim)
        self.use_source = _build_head(hidden_dim, 1)

        self.attack_source_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.attack_source_state = nn.Linear(state_dim, hidden_dim)
        self.attack_source = _build_head(hidden_dim, 1)

        # TARGET HEADS (per action type, conditioned on chosen source)
        self.summon_target_lane = nn.Linear(lane_emb_dim, hidden_dim)
        self.summon_target_source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.summon_target_state = nn.Linear(state_dim, hidden_dim)
        self.summon_target = _build_head(hidden_dim, 1)

        self.use_target_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.use_target_source_card = nn.Linear(card_emb_dim, hidden_dim)
        self.use_target_state = nn.Linear(state_dim, hidden_dim)
        self.use_target = _build_head(hidden_dim, 1)

        self.attack_target_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.attack_target_source_creature = nn.Linear(creature_emb_dim, hidden_dim)
        self.attack_target_state = nn.Linear(state_dim, hidden_dim)
        self.attack_target = _build_head(hidden_dim, 1)

        # Null targets (for USE face target and ATTACK face target)
        self.use_null_target = nn.Parameter(
            torch.randn(1, 1, creature_emb_dim) * 0.02
        )
        self.attack_null_target = nn.Parameter(
            torch.randn(1, 1, creature_emb_dim) * 0.02
        )

        # VALUE HEAD
        self.value_net = _build_mlp(state_dim, hidden_dim, last_layer_dim_vf)

    @property
    def is_autoregressive(self) -> bool:
        return True

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        raise NotImplementedError(
            "Autoregressive actor-critic does not support forward_actor. "
            "Use sample_actions or evaluate_autoregressive."
        )

    def forward_critic(self, features):
        return self.value_net(features["state"])

    def get_policy_modules(self) -> list[nn.Module]:
        return [
            self.type_head[-1],
            self.summon_source[-1],
            self.use_source[-1],
            self.attack_source[-1],
            self.summon_target[-1],
            self.use_target[-1],
            self.attack_target[-1],
        ]

    def get_value_modules(self) -> list[nn.Module]:
        return [self.value_net[-1]]

    @staticmethod
    def _decompose_actions(flat_actions):
        """Vectorized mapping from flat to (type, src, tgt)"""
        types = torch.zeros_like(flat_actions)
        sources = torch.zeros_like(flat_actions)
        targets = torch.zeros_like(flat_actions)

        # PASS
        # flat_actions == 0 -> type 0, src 0, tgt 0

        # SUMMON
        is_summon = (flat_actions >= 1) & (flat_actions <= 16)
        summon_idx = flat_actions[is_summon] - 1
        types[is_summon] = 1
        sources[is_summon] = summon_idx // 2
        targets[is_summon] = summon_idx % 2

        # USE
        is_use = (flat_actions >= 17) & (flat_actions <= 120)
        use_idx = flat_actions[is_use] - 17
        types[is_use] = 2
        sources[is_use] = use_idx // 13
        targets[is_use] = use_idx % 13

        # ATTACK
        is_attack = (flat_actions >= 121) & (flat_actions <= 144)
        attack_idx = flat_actions[is_attack] - 121
        types[is_attack] = 3
        sources[is_attack] = attack_idx // 4
        targets[is_attack] = attack_idx % 4

        return types, sources, targets

    @staticmethod
    def _compose_actions(types, sources, targets):
        """Vectorized mapping from (type, src, tgt) to flat"""
        flat_actions = torch.zeros_like(types)

        is_summon = types == 1
        flat_actions[is_summon] = 1 + sources[is_summon] * 2 + targets[is_summon]

        is_use = types == 2
        flat_actions[is_use] = 17 + sources[is_use] * 13 + targets[is_use]

        is_attack = types == 3
        flat_actions[is_attack] = 121 + sources[is_attack] * 4 + targets[is_attack]

        return flat_actions

    def _derive_masks(self, action_mask):
        bs = action_mask.shape[0]
        action_mask = action_mask.bool()  # ensure bool; SB3 may pass float obs

        # Reshape flat mask into per-type-action 2D masks
        pass_mask = action_mask[:, 0:1]  # [bs, 1]
        summon_mask_2d = action_mask[:, 1:17].reshape(bs, 8, 2)  # [bs, 8, 2]
        use_mask_2d = action_mask[:, 17:121].reshape(bs, 8, 13)  # [bs, 8, 13]
        attack_mask_2d = action_mask[:, 121:145].reshape(bs, 6, 4)  # [bs, 6, 4]

        # Type mask
        type_mask = torch.stack(
            [
                pass_mask.squeeze(-1),
                summon_mask_2d.flatten(1).any(dim=1),
                use_mask_2d.flatten(1).any(dim=1),
                attack_mask_2d.flatten(1).any(dim=1),
            ],
            dim=1,
        )  # [bs, 4]

        # Source masks
        summon_src_mask = summon_mask_2d.any(dim=-1)  # [bs, 8]
        use_src_mask = use_mask_2d.any(dim=-1)  # [bs, 8]
        attack_src_mask = attack_mask_2d.any(dim=-1)  # [bs, 6]

        return (
            type_mask,
            summon_mask_2d,
            use_mask_2d,
            attack_mask_2d,
            summon_src_mask,
            use_src_mask,
            attack_src_mask,
        )

    def _score_sources(self, features):
        state = features["state"]  # [bs, state_dim]
        hand_cards = features["hand_cards"]  # [bs, 8, card_emb_dim]

        p_lane0_c = features["p_lane0_creatures"]  # [bs, 3, c_dim]
        p_lane1_c = features["p_lane1_creatures"]  # [bs, 3, c_dim]
        board_creatures = torch.cat([p_lane0_c, p_lane1_c], dim=1)  # [bs, 6, c_dim]

        summon_input = (
            self.summon_source_card(hand_cards)
            + self.summon_source_state(state)[:, None, :]
        )
        summon_scores = self.summon_source(summon_input).squeeze(-1)  # [bs, 8]

        use_input = (
            self.use_source_card(hand_cards)
            + self.use_source_state(state)[:, None, :]
        )
        use_scores = self.use_source(use_input).squeeze(-1)  # [bs, 8]

        attack_input = (
            self.attack_source_creature(board_creatures)
            + self.attack_source_state(state)[:, None, :]
        )
        attack_scores = self.attack_source(attack_input).squeeze(-1)  # [bs, 6]

        return summon_scores, use_scores, attack_scores, board_creatures

    def _compute_source_logprob_entropy(
        self,
        features,
        types,
        summon_src_mask,
        use_src_mask,
        attack_src_mask,
        sources=None,
        deterministic=False,
    ):
        bs = features["state"].shape[0]
        device = features["state"].device

        is_pass = types == 0

        summon_scores, use_scores, attack_scores, board_creatures = (
            self._score_sources(features)
        )

        # Pad creature scores (6) to match hand size (8)
        padded_attack_scores = F.pad(attack_scores, (0, 2), value=-1e9)
        padded_attack_mask = F.pad(attack_src_mask, (0, 2), value=False)

        # Stack by type, then gather for known type
        zeros_scores = torch.full((bs, 8), -1e9, device=device)
        zeros_mask = torch.zeros((bs, 8), dtype=torch.bool, device=device)

        all_src_scores = torch.stack(
            [zeros_scores, summon_scores, use_scores, padded_attack_scores],
            dim=1,
        )  # [bs, 4, 8]
        all_src_masks = torch.stack(
            [zeros_mask, summon_src_mask, use_src_mask, padded_attack_mask],
            dim=1,
        )  # [bs, 4, 8]

        b_idx = torch.arange(bs, device=device)
        src_scores = all_src_scores[b_idx, types]  # [bs, 8]
        src_masks = all_src_masks[b_idx, types]  # [bs, 8]

        src_scores = src_scores.masked_fill(~src_masks, -1e9)

        # Handle PASS rows (all masks False) to avoid NaN in Categorical:
        # give them a dummy uniform distribution, then zero out their log_probs/entropy
        valid_rows = src_masks.any(dim=1)  # [bs]
        safe_src_scores = torch.where(
            valid_rows.unsqueeze(1), src_scores, torch.zeros_like(src_scores)
        )

        dist = torch.distributions.Categorical(logits=safe_src_scores)

        if sources is None:
            if deterministic:
                sources = safe_src_scores.argmax(dim=-1)
            else:
                sources = dist.sample()

        log_probs = dist.log_prob(sources)
        entropy = dist.entropy()

        # Zero out for PASS (no source selection)
        log_probs = log_probs.masked_fill(is_pass, 0.0)
        entropy = entropy.masked_fill(is_pass, 0.0)

        return sources, log_probs, entropy, board_creatures

    def _compute_target_logprob_entropy(
        self,
        features,
        types,
        sources,
        summon_mask_2d,
        use_mask_2d,
        attack_mask_2d,
        board_creatures,
        targets=None,
        deterministic=False,
    ):
        bs = features["state"].shape[0]
        device = features["state"].device

        is_pass = types == 0
        state = features["state"]  # [bs, state_dim]

        # 1. Chosen source embeddings for target conditioning
        safe_hand_src = torch.clamp(sources, 0, 7)
        safe_creature_src = torch.clamp(sources, 0, 5)

        b_idx = torch.arange(bs, device=device)
        chosen_hand_card = features["hand_cards"][b_idx, safe_hand_src]  # [bs, card_emb_dim]
        chosen_board_creature = board_creatures[b_idx, safe_creature_src]  # [bs, creature_emb_dim]

        # 2. Extract targets and compute per-type target scores
        # SUMMON targets: lanes (2)
        p_lane0 = features["p_lane0"]
        p_lane1 = features["p_lane1"]
        lane_targets = torch.stack([p_lane0, p_lane1], dim=1)  # [bs, 2, lane_emb_dim]
        summon_tgt_input = (
            self.summon_target_lane(lane_targets)
            + self.summon_target_source_card(chosen_hand_card)[:, None, :]
            + self.summon_target_state(state)[:, None, :]
        )
        summon_tgt_scores = self.summon_target(summon_tgt_input).squeeze(-1)  # [bs, 2]

        # USE targets: face (0) + 12 creatures
        all_creatures = torch.cat(
            [
                features["p_lane0_creatures"],
                features["p_lane1_creatures"],
                features["op_lane0_creatures"],
                features["op_lane1_creatures"],
            ],
            dim=1,
        )  # [bs, 12, creature_emb_dim]
        use_targets = torch.cat(
            [self.use_null_target.expand(bs, -1, -1), all_creatures], dim=1
        )  # [bs, 13, creature_emb_dim]
        use_tgt_input = (
            self.use_target_creature(use_targets)
            + self.use_target_source_card(chosen_hand_card)[:, None, :]
            + self.use_target_state(state)[:, None, :]
        )
        use_tgt_scores = self.use_target(use_tgt_input).squeeze(-1)  # [bs, 13]

        # ATTACK targets: conditioned on lane
        # If source < 3 (lane 0), targets are op_face + op_lane0. If source >= 3 (lane 1), op_face + op_lane1.
        is_lane1 = sources >= 3
        op_lane0_c = features["op_lane0_creatures"]  # [bs, 3, c_dim]
        op_lane1_c = features["op_lane1_creatures"]  # [bs, 3, c_dim]
        attack_target_creatures = torch.where(
            is_lane1.unsqueeze(-1).unsqueeze(-1),
            op_lane1_c,
            op_lane0_c,
        )  # [bs, 3, c_dim]
        attack_targets = torch.cat(
            [self.attack_null_target.expand(bs, -1, -1), attack_target_creatures],
            dim=1,
        )  # [bs, 4, c_dim]
        attack_tgt_input = (
            self.attack_target_creature(attack_targets)
            + self.attack_target_source_creature(chosen_board_creature)[:, None, :]
            + self.attack_target_state(state)[:, None, :]
        )
        attack_tgt_scores = self.attack_target(attack_tgt_input).squeeze(-1)  # [bs, 4]

        # 3. Extract masks for specific sources
        summon_tgt_mask = summon_mask_2d[b_idx, safe_hand_src]  # [bs, 2]
        use_tgt_mask = use_mask_2d[b_idx, safe_hand_src]  # [bs, 13]
        attack_tgt_mask = attack_mask_2d[b_idx, safe_creature_src]  # [bs, 4]

        # 4. Pad to 13 and stack
        padded_summon_scores = F.pad(summon_tgt_scores, (0, 11), value=-1e9)
        padded_summon_mask = F.pad(summon_tgt_mask, (0, 11), value=False)
        padded_attack_scores = F.pad(attack_tgt_scores, (0, 9), value=-1e9)
        padded_attack_mask = F.pad(attack_tgt_mask, (0, 9), value=False)

        zeros_scores = torch.full((bs, 13), -1e9, device=device)
        zeros_mask = torch.zeros((bs, 13), dtype=torch.bool, device=device)

        all_tgt_scores = torch.stack(
            [zeros_scores, padded_summon_scores, use_tgt_scores, padded_attack_scores],
            dim=1,
        )  # [bs, 4, 13]
        all_tgt_masks = torch.stack(
            [zeros_mask, padded_summon_mask, use_tgt_mask, padded_attack_mask],
            dim=1,
        )  # [bs, 4, 13]

        tgt_scores = all_tgt_scores[b_idx, types]  # [bs, 13]
        tgt_masks = all_tgt_masks[b_idx, types]  # [bs, 13]

        tgt_scores = tgt_scores.masked_fill(~tgt_masks, -1e9)

        # 5. Handle PASS rows to avoid NaN in Categorical
        valid_rows = tgt_masks.any(dim=1)
        safe_tgt_scores = torch.where(
            valid_rows.unsqueeze(1), tgt_scores, torch.zeros_like(tgt_scores)
        )

        dist = torch.distributions.Categorical(logits=safe_tgt_scores)

        if targets is None:
            if deterministic:
                targets = safe_tgt_scores.argmax(dim=-1)
            else:
                targets = dist.sample()

        log_probs = dist.log_prob(targets)
        entropy = dist.entropy()

        # Zero out for PASS (no target selection)
        log_probs = log_probs.masked_fill(is_pass, 0.0)
        entropy = entropy.masked_fill(is_pass, 0.0)

        return targets, log_probs, entropy

    def sample_actions(self, features, deterministic=False):
        """Sequential sampling for inference"""
        action_mask = features["action_mask"]
        (
            type_mask,
            summon_mask_2d,
            use_mask_2d,
            attack_mask_2d,
            summon_src_mask,
            use_src_mask,
            attack_src_mask,
        ) = self._derive_masks(action_mask)

        # TYPE
        type_logits = self.type_head(features["state"])
        type_logits = type_logits.masked_fill(~type_mask, -1e9)
        type_dist = torch.distributions.Categorical(logits=type_logits)

        if deterministic:
            types = type_logits.argmax(dim=-1)
        else:
            types = type_dist.sample()

        type_log_probs = type_dist.log_prob(types)

        # SOURCE
        sources, src_log_probs, _, board_creatures = (
            self._compute_source_logprob_entropy(
                features,
                types,
                summon_src_mask,
                use_src_mask,
                attack_src_mask,
                sources=None,
                deterministic=deterministic,
            )
        )

        # TARGET
        targets, tgt_log_probs, _ = self._compute_target_logprob_entropy(
            features,
            types,
            sources,
            summon_mask_2d,
            use_mask_2d,
            attack_mask_2d,
            board_creatures,
            targets=None,
            deterministic=deterministic,
        )

        flat_actions = self._compose_actions(types, sources, targets)
        log_probs = type_log_probs + src_log_probs + tgt_log_probs

        return flat_actions, log_probs

    def evaluate_autoregressive(self, features, actions):
        """Parallel evaluation for training"""
        types, sources, targets = self._decompose_actions(actions)
        action_mask = features["action_mask"]
        (
            type_mask,
            summon_mask_2d,
            use_mask_2d,
            attack_mask_2d,
            summon_src_mask,
            use_src_mask,
            attack_src_mask,
        ) = self._derive_masks(action_mask)

        # TYPE
        type_logits = self.type_head(features["state"])
        type_logits = type_logits.masked_fill(~type_mask, -1e9)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        type_log_probs = type_dist.log_prob(types)
        type_entropy = type_dist.entropy()

        # SOURCE
        _, src_log_probs, src_entropy, board_creatures = (
            self._compute_source_logprob_entropy(
                features,
                types,
                summon_src_mask,
                use_src_mask,
                attack_src_mask,
                sources=sources,
                deterministic=False,
            )
        )

        # TARGET
        _, tgt_log_probs, tgt_entropy = self._compute_target_logprob_entropy(
            features,
            types,
            sources,
            summon_mask_2d,
            use_mask_2d,
            attack_mask_2d,
            board_creatures,
            targets=targets,
            deterministic=False,
        )

        log_probs = type_log_probs + src_log_probs + tgt_log_probs
        entropy = type_entropy + src_entropy + tgt_entropy

        return log_probs, entropy
