"""
Unified model builder for LOCM battle networks.

Usage:
    from gym_locm.toolbox.networks import build_network, load_network
    
    model = build_network(
        env, seed,
        feature_extractor="zone_attn",
        actor_critic="typed",
        lstm=256,  # or False for no LSTM
        n_steps=512,
        nminibatches=256,
        noptepochs=4,
        cliprange=0.2,
        vf_coef=1.0,
        ent_coef=0.01,
        learning_rate=3e-4,
    )
"""

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from gym_locm.toolbox.networks.feature_extractors import FEATURE_EXTRACTORS
from gym_locm.toolbox.networks.actor_critics import ACTOR_CRITICS
from gym_locm.toolbox.networks.policies import (
    LOCMActorCriticPolicy,
    LOCMRecurrentActorCriticPolicy,
)

# Valid combinations: simple FE can only be paired with simple AC
_SIMPLE_ONLY_AC = {"simple"}
_ENTITY_LEVEL_FE = {"deep_sets", "zone_attn", "full_attn", "gnn"}
_ENTITY_LEVEL_AC = {"typed", "bilinear", "conditional", "autoreg"}


def build_network(
    env,
    seed,
    feature_extractor: str,
    actor_critic: str,
    # PPO hyperparameters
    n_steps: int,
    nminibatches: int,
    noptepochs: int,
    cliprange: float,
    vf_coef: float,
    ent_coef: float,
    learning_rate: float,
    gamma: float = 1,
    gae_lambda: float = 0.95,
    tensorboard_log=None,
    lstm: bool | int = False,
    # Architecture params (passed through to feature extractor/actor-critic)
    feature_extractor_kwargs: dict | None = None,
    actor_critic_kwargs: dict | None = None,
    # Legacy params (accepted but ignored for non-simple extractors)
    neurons=None,
    layers=None,
    activation=None,
    compile_model: bool = True,
):
    """Build a PPO or RecurrentPPO model with the specified architecture.
    
    Args:
        env: Vectorized environment.
        seed: Random seed.
        feature_extractor: One of "simple", "deep_sets", "zone_attn",
            "full_attn", "gnn".
        actor_critic: One of "simple", "typed", "bilinear", "conditional".
        n_steps: Number of steps per rollout.
        nminibatches: Batch size.
        noptepochs: Number of optimization epochs.
        cliprange: PPO clip range.
        vf_coef: Value function coefficient.
        ent_coef: Entropy coefficient.
        learning_rate: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        tensorboard_log: TensorBoard log directory.
        lstm: False for no LSTM, or int for LSTM hidden size.
        feature_extractor_kwargs: Additional kwargs for the feature extractor constructor.
        actor_critic_kwargs: Additional kwargs for the actor-critic constructor.
        neurons: Legacy param for simple extractor (MLP width).
        layers: Legacy param for simple extractor (MLP depth).
        activation: Legacy param for simple extractor (activation function name).
    """
    # Validate combination
    if feature_extractor == "simple" and actor_critic not in _SIMPLE_ONLY_AC:
        raise ValueError(
            f"Simple feature extractor can only be paired with simple actor-critic, "
            f"got '{actor_critic}'"
        )
    if feature_extractor in _ENTITY_LEVEL_FE and actor_critic not in _ENTITY_LEVEL_AC:
        raise ValueError(
            f"Entity-level feature extractor '{feature_extractor}' must be paired with "
            f"an entity-level actor-critic (one of {_ENTITY_LEVEL_AC}), got '{actor_critic}'"
        )

    # Resolve classes
    fe_class = FEATURE_EXTRACTORS[feature_extractor]
    ac_class = ACTOR_CRITICS[actor_critic]

    # Build feature extractor kwargs
    fe_kwargs = dict(feature_extractor_kwargs or {})

    # Handle legacy simple extractor params
    if feature_extractor == "simple" and neurons is not None and layers is not None:
        import torch.nn as nn
        if isinstance(layers, int):
            net_arch = [neurons] * layers
        else:
            net_arch = layers
        act_fn = dict(tanh=nn.Tanh, relu=nn.ReLU, elu=nn.ELU).get(activation, nn.ReLU)
        fe_kwargs.setdefault("net_arch", net_arch)
        fe_kwargs.setdefault("activation_fn", act_fn)

    # Build actor-critic kwargs
    ac_kwargs = dict(actor_critic_kwargs or {})

    # Build policy kwargs
    policy_kwargs = dict(
        features_extractor_class=fe_class,
        features_extractor_kwargs=fe_kwargs,
        actor_critic_class=ac_class,
        actor_critic_kwargs=ac_kwargs,
        compile_model=compile_model,
    )

    if lstm:
        algo = RecurrentPPO
        policy = LOCMRecurrentActorCriticPolicy
        policy_kwargs["lstm_hidden_size"] = lstm if isinstance(lstm, int) else 256
    else:
        algo = PPO
        policy = LOCMActorCriticPolicy

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
        policy_kwargs=policy_kwargs,
    )


def load_network(path, feature_extractor: str = None, actor_critic: str = None, lstm: bool | int = False):
    """Load a saved PPO/RecurrentPPO model.
    
    Args:
        path: Path to the model file (without .zip extension).
        feature_extractor: Feature extractor type (for custom_objects if needed).
        actor_critic: Actor-critic type (for custom_objects if needed).
        lstm: Whether the model uses LSTM.
    
    Returns:
        A callable that loads the model: loaded_model_builder(env, seed, *args, **kwargs)
    """
    algo = RecurrentPPO if lstm else PPO

    def loaded_model_builder(env, seed, *args, **kwargs):
        return algo.load(path + ".zip", env=env, force_reset=True, seed=seed)

    return loaded_model_builder