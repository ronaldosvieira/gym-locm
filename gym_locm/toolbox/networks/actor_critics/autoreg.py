"""
Auto-regressive actor-critic for LOCM.

Inspired by OpenAI Five's factored action decomposition. Decomposes the
flat Discrete(145) action space into three sequential decisions:

    π(a|s) = π(type|s) · π(src|type,s) · π(tgt|type,src,s)

Action types:
    0 = PASS (terminal, no source/target)
    1 = SUMMON (source: hand card, target: lane)
    2 = USE (source: hand card, target: creature or null)
    3 = ATTACK (source: board creature, target: opponent creature or face)

Conditioning uses concatenation (not additive bias) — the source entity's
embedding is concatenated to the target head's input, matching Five's
approach for maximal expressiveness.

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
        nn.Linear(hidden_dim // 2, output_dim)
    )


class AutoRegressiveLOCMNetwork(LOCMActorCriticNetwork):
    """
    OpenAI Five-style auto-regressive action decomposition actor-critic head.
    
    Decomposes the flat 145 action space into 3 sequential decisions:
    1. Action TYPE (PASS=0, SUMMON=1, USE=2, ATTACK=3)
    2. SOURCE entity (conditioned on type)
    3. TARGET entity (conditioned on type + source)
    """

    def __init__(self, state_dim, card_emb_dim, creature_emb_dim, lane_emb_dim,
                 hidden_dim=64, type_emb_dim=16, last_layer_dim_pi=145, last_layer_dim_vf=1):
        super().__init__(last_layer_dim_pi=last_layer_dim_pi, last_layer_dim_vf=last_layer_dim_vf)
        
        self.type_emb_dim = type_emb_dim
        
        # TYPE HEAD
        self.type_head = _build_mlp(state_dim, hidden_dim, 4)
        self.type_emb = nn.Embedding(4, type_emb_dim)
        
        # SOURCE HEAD
        self.source_card_proj = nn.Linear(card_emb_dim, hidden_dim)
        self.source_creature_proj = nn.Linear(creature_emb_dim, hidden_dim)
        self.source_score = _build_mlp(hidden_dim + state_dim + type_emb_dim, hidden_dim, 1)
        
        # TARGET HEAD
        self.target_lane_proj = nn.Linear(lane_emb_dim, hidden_dim)
        self.target_creature_proj = nn.Linear(creature_emb_dim, hidden_dim)
        
        self.target_source_card_proj = nn.Linear(card_emb_dim, hidden_dim)
        self.target_source_creature_proj = nn.Linear(creature_emb_dim, hidden_dim)
        
        self.target_score = _build_mlp(2 * hidden_dim + state_dim + type_emb_dim, hidden_dim, 1)
        
        # Null targets (for USE face target and ATTACK face target)
        self.use_null_target = nn.Parameter(torch.randn(1, 1, creature_emb_dim) * 0.02)
        self.attack_null_target = nn.Parameter(torch.randn(1, 1, creature_emb_dim) * 0.02)
        
        # VALUE HEAD
        self.value_net = _build_mlp(state_dim, hidden_dim, last_layer_dim_vf)

    @property
    def is_autoregressive(self) -> bool:
        return True

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        raise NotImplementedError("Autoregressive actor-critic does not support forward_actor. Use sample_actions or evaluate_autoregressive.")

    def forward_critic(self, features):
        state = features['state']
        return self.value_net(state)

    def get_policy_modules(self) -> list[nn.Module]:
        return [self.type_head[-1], self.source_score[-1], self.target_score[-1]]

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
        
        is_summon = (types == 1)
        flat_actions[is_summon] = 1 + sources[is_summon] * 2 + targets[is_summon]
        
        is_use = (types == 2)
        flat_actions[is_use] = 17 + sources[is_use] * 13 + targets[is_use]
        
        is_attack = (types == 3)
        flat_actions[is_attack] = 121 + sources[is_attack] * 4 + targets[is_attack]
        
        return flat_actions

    def _derive_masks(self, action_mask):
        bs = action_mask.shape[0]
        
        # Reshape flat mask into per-type-action 2D masks
        pass_mask = action_mask[:, 0:1]  # [bs, 1]
        summon_mask_2d = action_mask[:, 1:17].reshape(bs, 8, 2)  # [bs, 8, 2] 
        use_mask_2d = action_mask[:, 17:121].reshape(bs, 8, 13)  # [bs, 8, 13]
        attack_mask_2d = action_mask[:, 121:145].reshape(bs, 6, 4)  # [bs, 6, 4]
        
        # Type mask
        type_mask = torch.stack([
            pass_mask.squeeze(-1),
            summon_mask_2d.flatten(1).any(dim=1),
            use_mask_2d.flatten(1).any(dim=1),
            attack_mask_2d.flatten(1).any(dim=1),
        ], dim=1)  # [bs, 4]
        
        # Source masks
        summon_src_mask = summon_mask_2d.any(dim=-1)  # [bs, 8]
        use_src_mask = use_mask_2d.any(dim=-1)        # [bs, 8]
        attack_src_mask = attack_mask_2d.any(dim=-1)  # [bs, 6]
        
        return type_mask, summon_mask_2d, use_mask_2d, attack_mask_2d, summon_src_mask, use_src_mask, attack_src_mask

    def _score_sources(self, features, type_emb_cards, type_emb_creatures):
        bs = features['state'].shape[0]
        state = features['state'].unsqueeze(1)  # [bs, 1, state_dim]
        
        # Hand cards (for SUMMON and USE)
        hand_cards = features['hand_cards']  # [bs, 8, card_emb_dim]
        card_proj = self.source_card_proj(hand_cards)  # [bs, 8, hidden_dim]
        
        # Board creatures (for ATTACK)
        p_lane0_c = features['p_lane0_creatures']  # [bs, 3, c_dim]
        p_lane1_c = features['p_lane1_creatures']  # [bs, 3, c_dim]
        board_creatures = torch.cat([p_lane0_c, p_lane1_c], dim=1)  # [bs, 6, c_dim]
        creature_proj = self.source_creature_proj(board_creatures)  # [bs, 6, hidden_dim]
        
        # Score hand cards
        # Context for cards: proj + state + type_emb (which could be SUMMON or USE type emb)
        card_ctx = torch.cat([
            card_proj,
            state.expand(-1, 8, -1),
            type_emb_cards.unsqueeze(1).expand(-1, 8, -1)
        ], dim=-1)
        card_scores = self.source_score(card_ctx).squeeze(-1)  # [bs, 8]
        
        # Score board creatures
        creature_ctx = torch.cat([
            creature_proj,
            state.expand(-1, 6, -1),
            type_emb_creatures.unsqueeze(1).expand(-1, 6, -1)
        ], dim=-1)
        creature_scores = self.source_score(creature_ctx).squeeze(-1)  # [bs, 6]
        
        return card_scores, creature_scores, board_creatures

    def _compute_source_logprob_entropy(self, features, types, summon_src_mask, use_src_mask, attack_src_mask, sources=None, deterministic=False):
        bs = features['state'].shape[0]
        device = features['state'].device
        
        is_pass = (types == 0)
        
        # Get type embeddings for the actual selected types
        type_embs = self.type_emb(types)  # [bs, type_emb_dim]
        
        # We can evaluate cards and creatures with `type_embs` directly
        card_scores, creature_scores, board_creatures = self._score_sources(features, type_embs, type_embs)
        
        # Pad creature scores (6) to match hand size (8)
        padded_creature_scores = F.pad(creature_scores, (0, 2), value=-1e9)
        padded_attack_mask = F.pad(attack_src_mask, (0, 2), value=False)
        
        # Stack by type, then gather for known type
        zeros_scores = torch.full((bs, 8), -1e9, device=device)
        zeros_mask = torch.zeros((bs, 8), dtype=torch.bool, device=device)
        
        all_src_scores = torch.stack([zeros_scores, card_scores, card_scores, padded_creature_scores], dim=1)  # [bs, 4, 8]
        all_src_masks = torch.stack([zeros_mask, summon_src_mask, use_src_mask, padded_attack_mask], dim=1)  # [bs, 4, 8]
        
        b_idx = torch.arange(bs, device=device)
        src_scores = all_src_scores[b_idx, types]  # [bs, 8]
        src_masks = all_src_masks[b_idx, types]    # [bs, 8]
        
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

    def _compute_target_logprob_entropy(self, features, types, sources, summon_mask_2d, use_mask_2d, attack_mask_2d, board_creatures, targets=None, deterministic=False):
        bs = features['state'].shape[0]
        device = features['state'].device
        
        is_pass = (types == 0)
        
        # 1. Source embedding for target conditioning
        is_card_source = (types == 1) | (types == 2)
        safe_hand_src = torch.clamp(sources, 0, 7)
        safe_creature_src = torch.clamp(sources, 0, 5)
        
        b_idx = torch.arange(bs, device=device)
        source_card_proj = self.target_source_card_proj(features['hand_cards'][b_idx, safe_hand_src])  # [bs, hidden]
        source_creature_proj = self.target_source_creature_proj(board_creatures[b_idx, safe_creature_src])  # [bs, hidden]
        
        source_proj = torch.where(is_card_source.unsqueeze(1), source_card_proj, source_creature_proj)  # [bs, hidden]
        
        # 2. Extract targets
        # SUMMON targets: lanes (2)
        p_lane0 = features['p_lane0']
        p_lane1 = features['p_lane1']
        lane_targets = torch.stack([p_lane0, p_lane1], dim=1)  # [bs, 2, lane_emb_dim]
        lane_target_proj = self.target_lane_proj(lane_targets) # [bs, 2, hidden]
        
        # USE/ATTACK targets: creatures and null/face
        # All possible creatures: p_lane0, p_lane1, op_lane0, op_lane1
        all_creatures = torch.cat([
            features['p_lane0_creatures'],
            features['p_lane1_creatures'],
            features['op_lane0_creatures'],
            features['op_lane1_creatures']
        ], dim=1)  # [bs, 12, creature_emb_dim]
        
        # USE targets: face (0) + 12 creatures
        use_targets = torch.cat([self.use_null_target.expand(bs, -1, -1), all_creatures], dim=1)  # [bs, 13, creature_emb_dim]
        use_target_proj = self.target_creature_proj(use_targets)  # [bs, 13, hidden]
        
        # ATTACK targets: conditioned on lane.
        # If source < 3 (lane 0), targets are op_face + op_lane0. If source >= 3 (lane 1), op_face + op_lane1.
        is_lane1 = (sources >= 3)
        op_lane0_c = features['op_lane0_creatures']  # [bs, 3, c_dim]
        op_lane1_c = features['op_lane1_creatures']  # [bs, 3, c_dim]
        attack_target_creatures = torch.where(
            is_lane1.unsqueeze(-1).unsqueeze(-1),
            op_lane1_c,
            op_lane0_c
        )  # [bs, 3, c_dim]
        attack_targets = torch.cat([self.attack_null_target.expand(bs, -1, -1), attack_target_creatures], dim=1)  # [bs, 4, c_dim]
        attack_target_proj = self.target_creature_proj(attack_targets)  # [bs, 4, hidden]
        
        # 3. Compute scores
        state = features['state'].unsqueeze(1)
        type_embs = self.type_emb(types).unsqueeze(1)
        source_proj_exp = source_proj.unsqueeze(1)
        
        def score_targets(proj):
            n = proj.shape[1]
            ctx = torch.cat([
                proj,
                state.expand(-1, n, -1),
                type_embs.expand(-1, n, -1),
                source_proj_exp.expand(-1, n, -1)
            ], dim=-1)
            return self.target_score(ctx).squeeze(-1)
            
        summon_tgt_scores = score_targets(lane_target_proj)  # [bs, 2]
        use_tgt_scores = score_targets(use_target_proj)      # [bs, 13]
        attack_tgt_scores = score_targets(attack_target_proj) # [bs, 4]
        
        # 4. Extract masks for specific sources
        summon_tgt_mask = summon_mask_2d[b_idx, safe_hand_src]  # [bs, 2]
        use_tgt_mask = use_mask_2d[b_idx, safe_hand_src]        # [bs, 13]
        attack_tgt_mask = attack_mask_2d[b_idx, safe_creature_src] # [bs, 4]
        
        # 5. Pad to 13 and stack
        padded_summon_scores = F.pad(summon_tgt_scores, (0, 11), value=-1e9)
        padded_summon_mask = F.pad(summon_tgt_mask, (0, 11), value=False)
        padded_attack_scores = F.pad(attack_tgt_scores, (0, 9), value=-1e9)
        padded_attack_mask = F.pad(attack_tgt_mask, (0, 9), value=False)
        
        zeros_scores = torch.full((bs, 13), -1e9, device=device)
        zeros_mask = torch.zeros((bs, 13), dtype=torch.bool, device=device)
        
        all_tgt_scores = torch.stack([zeros_scores, padded_summon_scores, use_tgt_scores, padded_attack_scores], dim=1)  # [bs, 4, 13]
        all_tgt_masks = torch.stack([zeros_mask, padded_summon_mask, use_tgt_mask, padded_attack_mask], dim=1)  # [bs, 4, 13]
        
        tgt_scores = all_tgt_scores[b_idx, types]  # [bs, 13]
        tgt_masks = all_tgt_masks[b_idx, types]    # [bs, 13]
        
        tgt_scores = tgt_scores.masked_fill(~tgt_masks, -1e9)
        
        # 6. Handle PASS rows to avoid NaN in Categorical
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
        action_mask = features['action_mask']
        type_mask, summon_mask_2d, use_mask_2d, attack_mask_2d, summon_src_mask, use_src_mask, attack_src_mask = self._derive_masks(action_mask)
        
        # TYPE
        type_logits = self.type_head(features['state'])
        type_logits = type_logits.masked_fill(~type_mask, -1e9)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        
        if deterministic:
            types = type_logits.argmax(dim=-1)
        else:
            types = type_dist.sample()
            
        type_log_probs = type_dist.log_prob(types)
        
        # SOURCE
        sources, src_log_probs, _, board_creatures = self._compute_source_logprob_entropy(
            features, types, summon_src_mask, use_src_mask, attack_src_mask,
            sources=None, deterministic=deterministic
        )
        
        # TARGET
        targets, tgt_log_probs, _ = self._compute_target_logprob_entropy(
            features, types, sources, summon_mask_2d, use_mask_2d, attack_mask_2d, board_creatures,
            targets=None, deterministic=deterministic
        )
        
        flat_actions = self._compose_actions(types, sources, targets)
        log_probs = type_log_probs + src_log_probs + tgt_log_probs
        
        return flat_actions, log_probs

    def evaluate_autoregressive(self, features, actions):
        """Parallel evaluation for training"""
        types, sources, targets = self._decompose_actions(actions)
        action_mask = features['action_mask']
        type_mask, summon_mask_2d, use_mask_2d, attack_mask_2d, summon_src_mask, use_src_mask, attack_src_mask = self._derive_masks(action_mask)
        
        # TYPE
        type_logits = self.type_head(features['state'])
        type_logits = type_logits.masked_fill(~type_mask, -1e9)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        type_log_probs = type_dist.log_prob(types)
        type_entropy = type_dist.entropy()
        
        # SOURCE
        _, src_log_probs, src_entropy, board_creatures = self._compute_source_logprob_entropy(
            features, types, summon_src_mask, use_src_mask, attack_src_mask,
            sources=sources, deterministic=False
        )
        
        # TARGET
        _, tgt_log_probs, tgt_entropy = self._compute_target_logprob_entropy(
            features, types, sources, summon_mask_2d, use_mask_2d, attack_mask_2d, board_creatures,
            targets=targets, deterministic=False
        )
        
        log_probs = type_log_probs + src_log_probs + tgt_log_probs
        entropy = type_entropy + src_entropy + tgt_entropy
        
        return log_probs, entropy
