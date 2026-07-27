import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from gym_locm.toolbox.trainer_battle import SelfPlay, FixedAdversary, FixedAndSelfPlayHybrid
from gym_locm.toolbox.networks import (
    build_simple_network, 
    build_deep_sets_network, 
    build_set_transformer_network,
    build_gnn_network,
    load_simple_network, 
    load_deep_sets_network,
    load_set_transformer_network,
    load_gnn_network,
)
from gym_locm import agents

# Map model name strings to their builder functions
MODEL_BUILDERS = {
    "simple": build_simple_network,
    "deep_sets": build_deep_sets_network,
    "set_transformer": build_set_transformer_network,
    "gnn": build_gnn_network,
}

MODEL_LOADERS = {
    "simple": load_simple_network,
    "deep_sets": load_deep_sets_network,
    "set_transformer": load_set_transformer_network,
    "gnn": load_gnn_network,
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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Print the resolved configuration layout
    print("=" * 60)
    print("Resolved Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

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
    }

    # Fetch builder or loader
    load_path = cfg.model.get("load_path", None)
    if load_path:
        loader_fn = MODEL_LOADERS.get(cfg.model.name)
        if not loader_fn:
            raise ValueError(f"Unknown model architecture for loading: {cfg.model.name}")
        print(f"Loading pre-trained network from: {load_path}")
        model_builder = loader_fn(load_path, lstm=cfg.model.lstm)
    else:
        builder_fn = MODEL_BUILDERS.get(cfg.model.name)
        if not builder_fn:
            raise ValueError(f"Unknown model architecture: {cfg.model.name}")
        model_builder = builder_fn

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
        )
    else:
        raise ValueError(f"Invalid adversary setting: {cfg.adversary.type}")

    trainer.run()

if __name__ == "__main__":
    main()
