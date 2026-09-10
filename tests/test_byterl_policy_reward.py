"""Tests for ByteRLPolicyRewardFunction.

Invariants verified:
1. get_policy_probs() returns a valid probability distribution.
2. step() reward equals ByteRL prob for chosen action.
3. Rewards are always in [0, 1].
4. ByteRL LSTM advances across steps.
5. reset() reinitialises LSTM state.
6. Full episode completes without error.
7. Mixed reward functions preserve declaration order in raw_rewards.
8. action_id=None returns 0.0 gracefully.
"""
import numpy as np
import pytest
from gym_locm.agents import RandomDraftAgent, RandomBattleAgent
from gym_locm.agents.byterl.agent import ByterlAgent
from gym_locm.agents.byterl.main import ensure_weights_are_available
from gym_locm.engine import Phase, PlayerOrder
from gym_locm.envs.battle import LOCMBattleSingleEnv
from gym_locm.envs.rewards import ByteRLPolicyRewardFunction

SEED = 42


def _make_env(reward_functions=("byterl-policy",), reward_weights=(1.0,), seed=SEED):
    return LOCMBattleSingleEnv(
        deck_building_agents=(RandomDraftAgent(), RandomDraftAgent()),
        battle_agent=RandomBattleAgent(),
        reward_functions=reward_functions,
        reward_weights=reward_weights,
        play_first=True,
        seed=seed,
    )


def _first_battle_state(seed=SEED):
    env = _make_env(seed=seed)
    env.reset()
    return env.unwrapped.state


def _random_valid_action(state):
    """Return a random *integer* action index that is valid in the given state."""
    mask = np.asarray(state.action_mask, dtype=bool)
    valid = np.where(mask)[0]
    return int(np.random.default_rng(SEED).choice(valid))


class TestGetPolicyProbs:
    def setup_method(self):
        ensure_weights_are_available()

    def test_returns_array_of_length_145(self):
        state = _first_battle_state()
        agent = ByterlAgent()
        probs = agent.get_policy_probs(state)
        assert probs.shape == (145,)

    def test_probs_sum_to_one(self):
        state = _first_battle_state()
        agent = ByterlAgent()
        probs = agent.get_policy_probs(state)
        assert abs(probs.sum() - 1.0) < 1e-5, f"probs sum to {probs.sum()}"

    def test_all_probs_in_unit_interval(self):
        state = _first_battle_state()
        agent = ByterlAgent()
        probs = agent.get_policy_probs(state)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_invalid_actions_have_zero_probability(self):
        state = _first_battle_state()
        action_mask = state.action_mask
        agent = ByterlAgent()
        probs = agent.get_policy_probs(state)
        invalid_mask = ~np.asarray(action_mask, dtype=bool)
        assert np.all(probs[invalid_mask] == 0.0)

    def test_at_least_one_nonzero_prob(self):
        state = _first_battle_state()
        agent = ByterlAgent()
        probs = agent.get_policy_probs(state)
        assert probs.max() > 0.0

    def test_advances_lstm_state(self):
        """Two sequential get_policy_probs() calls on different states should
        yield different distributions because the LSTM hidden state advances."""
        env = _make_env()
        env.reset()
        state = env.unwrapped.state
        agent = ByterlAgent()
        probs1 = agent.get_policy_probs(state)
        action_id = _random_valid_action(state)
        env.step(action_id)
        state2 = env.unwrapped.state
        probs2 = agent.get_policy_probs(state2)
        assert not np.allclose(probs1, probs2)


class TestByterlPolicyRewardFunction:
    def setup_method(self):
        ensure_weights_are_available()

    def test_is_action_based_returns_true(self):
        fn = ByteRLPolicyRewardFunction()
        assert fn.is_action_based() is True

    def test_calculate_returns_zero(self):
        state = _first_battle_state()
        fn = ByteRLPolicyRewardFunction()
        assert fn.calculate(state, for_player=PlayerOrder.FIRST) == 0.0

    def test_action_id_none_returns_zero(self):
        state = _first_battle_state()
        fn = ByteRLPolicyRewardFunction()
        assert fn.calculate_action_reward(state, action_id=None) == 0.0

    def test_reward_is_valid_probability(self):
        state = _first_battle_state()
        fn = ByteRLPolicyRewardFunction()
        action_id = _random_valid_action(state)
        reward = fn.calculate_action_reward(state, action_id)
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0,1]"

    def test_reward_equals_byterl_prob_for_chosen_action(self):
        """The reward must equal the probability that a fresh ByterlAgent
        assigns to the same action on the same state."""
        state = _first_battle_state()
        # Reference: get probs from a separate fresh ByterlAgent
        ref_agent = ByterlAgent()
        expected_probs = ref_agent.get_policy_probs(state.clone())
        fn = ByteRLPolicyRewardFunction()
        for action_id in range(145):
            if not state.action_mask[action_id]:
                continue
            reward = fn.calculate_action_reward(state, action_id)
            expected = float(expected_probs[action_id])
            assert abs(reward - expected) < 1e-6, (
                f"action {action_id}: reward={reward}, expected={expected}"
            )
            break

    def test_reset_reinitialises_state(self):
        """After reset(), the first step of a new episode must give the same
        probability as a freshly constructed ByteRLPolicyRewardFunction."""
        state = _first_battle_state(seed=SEED)
        action_id = _random_valid_action(state)
        fn = ByteRLPolicyRewardFunction()
        _ = fn.calculate_action_reward(state, action_id)
        fn.reset()
        reward_after_reset = fn.calculate_action_reward(state, action_id)
        fn_fresh = ByteRLPolicyRewardFunction()
        reward_fresh = fn_fresh.calculate_action_reward(state, action_id)
        assert abs(reward_after_reset - reward_fresh) < 1e-6, (
            f"After reset: {reward_after_reset} != fresh: {reward_fresh}"
        )


class TestByterlPolicyRewardIntegration:
    """End-to-end tests through LOCMBattleSingleEnv.step() using integer actions.

    In RL training, the policy always outputs an integer action index.
    These tests replicate that by sampling a valid integer from the action mask.
    """

    def _run_full_episode(self, env, seed=SEED):
        """Run one full episode with random *integer* actions; return (reward, info) list."""
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        terminated = truncated = False
        steps = []
        while not (terminated or truncated):
            state = env.unwrapped.state
            mask = np.asarray(state.action_mask, dtype=bool)
            valid = np.where(mask)[0]
            action_id = int(rng.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action_id)
            steps.append((reward, info))
        return steps

    def test_full_episode_completes_without_error(self):
        env = _make_env()
        steps = self._run_full_episode(env)
        assert len(steps) > 0

    def test_all_step_rewards_in_unit_interval(self):
        env = _make_env()
        steps = self._run_full_episode(env)
        for i, (reward, _) in enumerate(steps):
            assert 0.0 <= reward <= 1.0, f"Step {i}: reward {reward} out of [0, 1]"

    def test_multiple_episodes_reset_correctly(self):
        env = _make_env()
        for episode in range(2):
            steps = self._run_full_episode(env, seed=SEED + episode)
            for i, (reward, _) in enumerate(steps):
                assert 0.0 <= reward <= 1.0, (
                    f"Episode {episode}, step {i}: reward {reward} out of [0, 1]"
                )

    def test_raw_rewards_ordering_with_mixed_functions(self):
        """raw_rewards[0] = win-loss delta (0 mid-episode);
        raw_rewards[1] = ByteRL policy prob in [0, 1]."""
        env = _make_env(
            reward_functions=("win-loss", "byterl-policy"),
            reward_weights=(1.0, 1.0),
        )
        rng = np.random.default_rng(SEED)
        env.reset()
        for _ in range(5):
            state = env.unwrapped.state
            if env.unwrapped._battle_is_finished:
                break
            mask = np.asarray(state.action_mask, dtype=bool)
            valid = np.where(mask)[0]
            action_id = int(rng.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action_id)
            if terminated:
                break
            raw = info["raw_rewards"]
            assert len(raw) == 2
            win_loss_delta, byterl_prob_reward = raw
            assert win_loss_delta == 0.0, f"Expected 0 win-loss delta, got {win_loss_delta}"
            assert 0.0 <= byterl_prob_reward <= 1.0, (
                f"ByteRL reward {byterl_prob_reward} out of [0, 1]"
            )

    def test_byterl_policy_only_positive_cumulative_reward(self):
        """Each step contributes a positive probability, so the episode total
        must be strictly greater than zero."""
        env = _make_env()
        steps = self._run_full_episode(env)
        total = sum(r for r, _ in steps)
        assert total > 0.0, f"Expected positive total reward, got {total}"

    def test_rewards_match_direct_byterl_probs(self):
        """For each step, the env reward must equal the ByteRL probability for
        the exact integer action taken, queried from a parallel ByterlAgent."""
        env = _make_env(seed=SEED)
        env.reset(seed=SEED)
        rng = np.random.default_rng(SEED)
        ref_agent = ByterlAgent()
        terminated = truncated = False
        step_idx = 0
        while not (terminated or truncated):
            state = env.unwrapped.state
            mask = np.asarray(state.action_mask, dtype=bool)
            valid = np.where(mask)[0]
            action_id = int(rng.choice(valid))
            expected_probs = ref_agent.get_policy_probs(state.clone())
            expected_reward = float(expected_probs[action_id])
            obs, reward, terminated, truncated, info = env.step(action_id)
            assert abs(reward - expected_reward) < 1e-5, (
                f"Step {step_idx}: env reward={reward}, expected ByteRL prob={expected_reward}"
            )
            step_idx += 1
