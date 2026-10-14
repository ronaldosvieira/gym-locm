import os
import re
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from gym_locm.toolbox.trainer_battle import SelfPlay, FixedAdversary, FixedAndSelfPlayHybrid
from gym_locm.toolbox.networks import build_network, load_network
from gym_locm import agents
import gym_locm

# Default actor-critic pairing for each feature extractor
DEFAULT_ACTOR_CRITICS = {
    "simple": "simple",
    "deep_sets": "bilinear",
    "zone_attn": "typed",
    "full_attn": "typed",
    "gnn": "typed",
}


def get_env_parameters(cfg: DictConfig):
    """Generate dict parameters for various environment configurations."""
    # Handle reward functions: could be a space-separated string or a list
    if isinstance(cfg.reward_functions, str):
        reward_functions = cfg.reward_functions.split()
    else:
        reward_functions = list(cfg.reward_functions)

    # Handle reward weights: could be a space-separated string of floats or a list
    if isinstance(cfg.reward_weights, (str, float, int)):
        reward_weights = list(map(float, str(cfg.reward_weights).split()))
    else:
        reward_weights = list(map(float, cfg.reward_weights))
        
    # Instantiate agents
    byterl = agents.ByteRL()
    greedy = agents.GreedyBattleAgent()
    adversary_agent = agents.parse_battle_agent(cfg.adversary.battle_agent)()

    common_params = {
        "deck_building_agents": (byterl, byterl),
        "reward_functions": reward_functions,
        "reward_weights": reward_weights,
        "use_average_deck": cfg.use_average_deck,
        "version": cfg.version,
        "dict_observations": True,
    }
    
    # 1. Self Play parameters
    self_play = common_params.copy()
    
    # 2. Fixed Adversary parameters
    fixed = common_params.copy()
    fixed["battle_agent"] = adversary_agent
    
    # 3. Evaluation environments parameters
    evals = [
        # vs. OSL (Greedy)
        {
            **common_params,
            "battle_agent": greedy,
        },
        
        # vs. ByteRL
        {
            **common_params,
            "battle_agent": byterl,
        }
    ]
    
    return self_play, fixed, evals


def resolve_resume(cfg: DictConfig):
    """Return (checkpoint path without extension, episodes done, wandb run id), or None."""
    resume_ckpt = cfg.get("resume", {}).get("checkpoint")
    resume_id = cfg.get("resume", {}).get("wandb_run_id")

    if not resume_ckpt and not resume_id:
        return None
    if not resume_ckpt or not resume_id:
        raise ValueError("resume.checkpoint and resume.wandb_run_id must be set together")
    if cfg.adversary.type != "fixed":
        raise ValueError("Resuming is only supported for the fixed adversary")

    if str(resume_ckpt) == "latest":
        # same base directory the trainer uses for `path`
        run_dir = os.path.join(os.path.dirname(gym_locm.__file__), "..", cfg.path)
        episodes = [
            int(m.group(1))
            for f in os.listdir(run_dir)
            if (m := re.fullmatch(r"(\d+)\.zip", f))
        ]
        if not episodes:
            raise FileNotFoundError(f"No <episodes>.zip checkpoints found in {run_dir}")
        resume_ckpt = os.path.join(run_dir, f"{max(episodes)}.zip")

    m = re.fullmatch(r"(\d+)\.zip", os.path.basename(str(resume_ckpt)))
    if not m or not os.path.isfile(resume_ckpt):
        raise ValueError(f"Checkpoint must be an existing '<episodes>.zip' file: {resume_ckpt}")

    return str(resume_ckpt)[: -len(".zip")], int(m.group(1)), str(resume_id)

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Print the resolved configuration layout
    print("=" * 60)
    print("Resolved Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    resume = resolve_resume(cfg)
    resume_episodes = resume[1] if resume else 0

    # Get env settings
    self_play_env_params, fixed_adversary_env_params, eval_env_params = get_env_parameters(cfg)

    # Calculate PPO miniblocks
    nminibatches = cfg.n_steps // int(cfg.nminibatches_divider)

    # Structure PPO model inputs
    model_params = {
        "layers": cfg.model.layers,
        "neurons": cfg.model.neurons,
        "n_steps": cfg.n_steps,
        "nminibatches": nminibatches,
        "noptepochs": cfg.noptepochs,
        "cliprange": cfg.cliprange,
        "vf_coef": cfg.vf_coef,
        "ent_coef": cfg.ent_coef,
        "activation": cfg.model.act_fun,
        "learning_rate": cfg.learning_rate,
        "gae_lambda": cfg.gae_lambda,
        "tensorboard_log": os.path.join(cfg.path, "tf_logs"),
        "gamma": cfg.gamma,
        "lstm": cfg.model.lstm,
        "feature_extractor_kwargs": (
            OmegaConf.to_container(cfg.model.feature_extractor_kwargs, resolve=True)
            if cfg.model.get("feature_extractor_kwargs") is not None else None
        ),
        "actor_critic_kwargs": (
            OmegaConf.to_container(cfg.model.actor_critic_kwargs, resolve=True)
            if cfg.model.get("actor_critic_kwargs") is not None else None
        ),
    }

    # Resolve feature extractor and actor-critic names
    fe_name = cfg.model.name
    ac_name = cfg.model.get("actor_critic", DEFAULT_ACTOR_CRITICS.get(fe_name))
    if ac_name is None:
        raise ValueError(f"Unknown model architecture: {fe_name}")

    # Fetch builder or loader
    load_path = resume[0] if resume else cfg.model.get("load_path", None)
    if load_path:
        print(f"Loading pre-trained network from: {load_path}")
        model_builder = load_network(load_path, feature_extractor=fe_name, actor_critic=ac_name, lstm=cfg.model.lstm)
    else:
        model_builder = partial(build_network, feature_extractor=fe_name, actor_critic=ac_name)

    # Track with W&B
    run = None
    if cfg.task == "battle" and not cfg.get("no_tracking", False):
        # Convert DictConfig to primitive dict for W&B logging
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.get("wandb_group", None),
            sync_tensorboard=True,
            config=wandb_config,
            **(dict(id=resume[2], resume="must") if resume else {}),
        )
        if resume:
            run.config.update(
                dict(resumed_from_episodes=resume_episodes, resumed_from_checkpoint=resume[0]),
                allow_val_change=True,
            )

    # Decay params passed to trainers, which apply them episode-by-episode in callbacks
    decay_params = dict(
        lr_decay=cfg.get("learning_rate_decay"),
        lr_final=float(cfg.get("learning_rate_final", 0.0)),
        cr_decay=cfg.get("clip_range_decay"),
        cr_final=float(cfg.get("clip_range_final", 0.0)),
    )

    # Initialize and execute the chosen adversary scheme
    if cfg.adversary.type == "self-play":
        trainer = SelfPlay(
            cfg.task,
            model_builder,
            model_params,
            self_play_env_params,
            eval_env_params,
            cfg.train_episodes,
            cfg.eval_episodes,
            cfg.num_evals,
            cfg.adversary.role,
            cfg.adversary.switch_freq,
            cfg.path,
            cfg.seed,
            cfg.concurrency,
            wandb_run=run,
            **decay_params,
        )
    elif cfg.adversary.type == "fixed":
        trainer = FixedAdversary(
            cfg.task,
            model_builder,
            model_params,
            fixed_adversary_env_params,
            eval_env_params,
            cfg.train_episodes,
            cfg.eval_episodes,
            cfg.num_evals,
            cfg.adversary.role,
            cfg.path,
            cfg.seed,
            cfg.concurrency,
            wandb_run=run,
            resume_episodes=resume_episodes,
            **decay_params,
        )
    elif cfg.adversary.type == "hybrid":
        trainer = FixedAndSelfPlayHybrid(
            cfg.task,
            model_builder,
            model_params,
            self_play_env_params,
            fixed_adversary_env_params,
            eval_env_params,
            cfg.train_episodes,
            cfg.eval_episodes,
            cfg.num_evals,
            cfg.adversary.role,
            cfg.adversary.switch_freq,
            cfg.path,
            cfg.seed,
            cfg.adversary.num_self_play_envs,
            cfg.adversary.num_fixed_adversary_envs,
            wandb_run=run,
            **decay_params,
        )
    else:
        raise ValueError(f"Invalid adversary setting: {cfg.adversary.type}")

    trainer.run()

if __name__ == "__main__":
    main()
