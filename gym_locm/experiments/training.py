import argparse
import os
import sys
from datetime import datetime

import wandb

from gym_locm import agents
from gym_locm.toolbox.trainer import AsymmetricSelfPlay, model_builder_mlp, \
    model_builder_lstm, model_builder_mlp_masked, SelfPlay, FixedAdversary

_counter = 0


def get_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    tasks = ['draft', 'battle']
    approach = ['immediate', 'lstm', 'history']
    battle_agents = ['max-attack', 'greedy']
    adversary = ['fixed', 'self-play', 'asymmetric-self-play']

    p.add_argument("--task", "-t", choices=tasks, default="draft")
    p.add_argument("--approach", "-ap", choices=approach, default="immediate")
    p.add_argument("--adversary", "-ad", choices=adversary, default="asymmetric-self-play")
    p.add_argument("--draft-agent", "-d", choices=list(agents.draft_agents.keys()),
                   default="max-attack")
    p.add_argument("--battle-agent", "-b", choices=battle_agents,
                   default="max-attack")
    p.add_argument("--path", "-p", help="path to save models and results",
                   required=True)

    p.add_argument("--train-episodes", "-te", help="how many episodes to train",
                   default=30000, type=int)
    p.add_argument("--eval-episodes", "-ee", help="how many episodes to eval",
                   default=1000, type=int)
    p.add_argument("--num-evals", "-ne", type=int, default=12,
                   help="how many evaluations to perform throughout training")

    p.add_argument("--switch-freq", type=int, default=1000,
                   help="how many episodes to run before updating opponent networks")
    p.add_argument("--layers", type=int, default=1,
                   help="amount of layers in the network")
    p.add_argument("--neurons", type=int, default=169,
                   help="amount of neurons on each hidden layer in the network")
    p.add_argument("--act-fun", choices=['tanh', 'relu', 'elu'], default='elu',
                   help="activation function of neurons in hidden layers")
    p.add_argument("--n-steps", type=int, default=270,
                   help="batch size (in timesteps, 30 timesteps = 1 episode)")
    p.add_argument("--nminibatches", type=int, default=135,
                   help="amount of minibatches created from the batch")
    p.add_argument("--nminibatches-divider", type=str, choices=["1", "2", "4", "8", "n"],
                   help="amount of minibatches created from the batch"
                        " -- by dividing the n-steps parameter")
    p.add_argument("--noptepochs", type=int, default=20,
                   help="amount of epochs to train with all minibatches")
    p.add_argument("--cliprange", type=float, default=0.1,
                   help="clipping range of the loss function")
    p.add_argument("--vf-coef", type=float, default=1.0,
                   help="weight of the value function in the loss function")
    p.add_argument("--ent-coef", type=float, default=0.00595,
                   help="weight of the entropy term in the loss function")
    p.add_argument("--learning-rate", type=float, default=0.000228,
                   help="learning rate")

    p.add_argument("--seed", type=int, default=None,
                   help="seed to use on the model, envs and training")
    p.add_argument("--concurrency", type=int, default=1,
                   help="amount of environments to use")

    return p


def run():
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    args.path += "/" + args.task + "-" + str(args.seed) + "-" + datetime.now().strftime("%y%m%d%H%M")

    os.makedirs(args.path, exist_ok=True)

    if args.task == 'draft':

        if args.approach == 'lstm':
            model_builder = model_builder_lstm
        else:
            model_builder = model_builder_mlp

        if args.battle_agent == 'greedy':
            battle_agent = agents.GreedyBattleAgent
        else:
            battle_agent = agents.MaxAttackBattleAgent

        env_params = {
            'battle_agents': (battle_agent(), battle_agent()),
            'use_draft_history': args.approach == 'history'
        }

        eval_env_params = {
            'draft_agent': agents.MaxAttackDraftAgent(),
            'battle_agents': (battle_agent(), battle_agent()),
            'use_draft_history': args.approach == 'history'
        }

    elif args.task == 'battle':

        model_builder = model_builder_mlp_masked
        draft_agent = agents.parse_draft_agent(args.draft_agent)
        battle_agent = agents.parse_battle_agent(args.battle_agent)

        env_params = {
            'draft_agents': (draft_agent(), draft_agent())
        }

        if args.adversary == 'fixed':
            env_params['battle_agent'] = battle_agent()

        eval_env_params = {
            'draft_agents': (draft_agent(), draft_agent()),
            'battle_agent': agents.MaxAttackBattleAgent()
        }

    else:
        raise Exception("Invalid task")

    if args.nminibatches_divider == "n":
        args.nminibatches = args.n_steps
    elif args.nminibatches_divider is not None:
        args.nminibatches = args.n_steps // int(args.nminibatches_divider)

    model_params = {'layers': args.layers, 'neurons': args.neurons,
                    'n_steps': args.n_steps, 'nminibatches': args.nminibatches,
                    'noptepochs': args.noptepochs, 'cliprange': args.cliprange,
                    'vf_coef': args.vf_coef, 'ent_coef': args.ent_coef,
                    'activation': args.act_fun, 'learning_rate': args.learning_rate,
                    'tensorboard_log': args.path + '/tf_logs'}

    run = wandb.init(
        project='gym-locm',
        entity='ronaldosvieira',
        sync_tensorboard=True,
        config=vars(args)
    )

    # enable the use of wandb sweeps
    args = wandb.config

    if args.adversary == 'asymmetric-self-play':
        trainer = AsymmetricSelfPlay(
            args.task, model_builder, model_params, env_params, eval_env_params,
            args.train_episodes, args.eval_episodes, args.num_evals,
            args.switch_freq, args.path, args.seed, args.concurrency,
            wandb_run=run
        )
    elif args.adversary == 'self-play':
        trainer = SelfPlay(
            args.task, model_builder, model_params, env_params, eval_env_params,
            args.train_episodes, args.eval_episodes, args.num_evals,
            args.switch_freq, args.path, args.seed, args.concurrency,
            wandb_run=run
        )
    elif args.adversary == 'fixed':
        trainer = FixedAdversary(
            args.task, model_builder, model_params, env_params, eval_env_params,
            args.train_episodes, args.eval_episodes, args.num_evals,
            True, args.path, args.seed, args.concurrency, wandb_run=run
        )
    else:
        raise Exception("Invalid adversary")

    try:
        trainer.run()
    finally:
        run.finish()


if __name__ == "__main__":
    run()
