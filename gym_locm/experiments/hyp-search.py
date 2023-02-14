import argparse
import os
import pickle
import sys

import numpy as np
from hyperopt import hp, STATUS_OK, trials_from_docs, Trials, partial, tpe, fmin
from hyperopt.pyll import scope

from gym_locm.agents import MaxAttackBattleAgent, GreedyBattleAgent, MaxAttackDraftAgent
from gym_locm.toolbox.trainer_draft import (
    AsymmetricSelfPlay,
    model_builder_mlp,
    model_builder_lstm,
)

hyperparameter_space = {
    "switch_freq": hp.choice("switch_freq", [10, 100, 1000]),
    "layers": hp.uniformint("layers", 1, 3),
    "neurons": hp.uniformint("neurons", 24, 256),
    "activation": hp.choice("activation", ["tanh", "relu", "elu"]),
    "n_steps": scope.int(hp.quniform("n_steps", 30, 300, 30)),
    "nminibatches": scope.int(hp.quniform("nminibatches", 1, 300, 1)),
    "noptepochs": scope.int(hp.quniform("noptepochs", 3, 20, 1)),
    "cliprange": hp.quniform("cliprange", 0.1, 0.3, 0.1),
    "vf_coef": hp.quniform("vf_coef", 0.5, 1.0, 0.5),
    "ent_coef": hp.uniform("ent_coef", 0, 0.01),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.00005), np.log(0.01)),
}

_counter = 0


def get_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    approach = ["immediate", "lstm", "history", "curve"]
    battle_agents = ["max-attack", "greedy"]

    p.add_argument("--approach", "-a", choices=approach, default="immediate")
    p.add_argument("--battle-agent", "-b", choices=battle_agents, default="max-attack")
    p.add_argument(
        "--path", "-p", help="path to save models and results", required=True
    )

    p.add_argument(
        "--train-episodes",
        "-te",
        help="how many episodes to train",
        default=30000,
        type=int,
    )
    p.add_argument(
        "--eval-episodes",
        "-ee",
        help="how many episodes to eval",
        default=1000,
        type=int,
    )
    p.add_argument(
        "--num-evals",
        "-ne",
        type=int,
        default=12,
        help="how many evaluations to perform throughout training",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed to use on the model, envs and hyperparameter search",
    )
    p.add_argument(
        "--processes", type=int, help="amount of processes to use", default=1
    )

    p.add_argument(
        "--num-trials",
        "-n",
        type=int,
        default=50,
        help="amount of hyperparameter sets to test",
    )
    p.add_argument(
        "--num-warmup-trials",
        "-w",
        type=int,
        default=20,
        help="amount of random hyperparameter sets to test before "
        "starting optimizing",
    )

    return p


def load_run(path):
    # tries to load past trials
    try:
        with open(path + "/trials.p", "rb") as trials_file:
            trials = pickle.load(trials_file)

            if trials.trials[-1]["result"]["status"] != STATUS_OK:
                trials = trials_from_docs(trials.trials[:-1])

            print(
                f"Found previous incomplete run with {len(trials)} "
                f"trials. Continuing..."
            )
    except FileNotFoundError:
        trials = Trials()

    # tries to load past random state
    try:
        with open(path + "/rstate.p", "rb") as random_state_file:
            random_state = pickle.load(random_state_file)
    except FileNotFoundError:
        random_state = np.random.default_rng(seed=args.seed)

    return trials, random_state


def save_run(trials, random_state, path):
    with open(path + "/trials.p", "wb") as trials_file:
        pickle.dump(trials, trials_file)

    with open(path + "/rstate.p", "wb") as random_state_file:
        pickle.dump(random_state, random_state_file)


if __name__ == "__main__":
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    os.makedirs(args.path, exist_ok=True)

    trials, random_state = load_run(args.path)

    trials_so_far = len(trials)

    if args.approach == "lstm":
        model_builder = model_builder_lstm
        hyperparameter_space["nminibatches"] = hp.choice("nminibatches", [1])
    else:
        model_builder = model_builder_mlp

    if args.battle_agent == "max-attack":
        battle_agent = MaxAttackBattleAgent
    else:
        battle_agent = GreedyBattleAgent

    env_params = {
        "battle_agents": (battle_agent(), battle_agent()),
        "use_draft_history": args.approach == "history",
        "use_mana_curve": args.approach == "curve",
    }

    eval_env_params = {
        "draft_agent": MaxAttackDraftAgent(),
        "battle_agents": (battle_agent(), battle_agent()),
        "use_draft_history": args.approach == "history",
        "use_mana_curve": args.approach == "curve",
    }

    def wrapper(model_params: dict):
        global _counter

        save_run(trials, random_state, args.path)

        switch_freq = model_params.pop("switch_freq")

        # ensure integer hyperparams
        model_params["n_steps"] = int(model_params["n_steps"])
        model_params["nminibatches"] = int(model_params["nminibatches"])
        model_params["noptepochs"] = int(model_params["noptepochs"])

        # ensure nminibatches <= n_steps
        model_params["nminibatches"] = min(
            model_params["nminibatches"], model_params["n_steps"]
        )

        # ensure n_steps % nminibatches == 0
        while model_params["n_steps"] % model_params["nminibatches"] != 0:
            model_params["nminibatches"] -= 1

        _counter += 1
        trial_id = _counter

        trainer = AsymmetricSelfPlay(
            "draft",
            model_builder,
            model_params,
            env_params,
            eval_env_params,
            args.train_episodes,
            args.eval_episodes,
            args.num_evals,
            switch_freq,
            args.path + f"/{trial_id}",
            args.seed,
            args.processes,
        )
        trainer.run()

        best_win_rate = -max(max(player_wr) for player_wr in trainer.win_rates)

        return best_win_rate

    algo = partial(
        tpe.suggest, n_startup_jobs=max(0, args.num_warmup_trials - trials_so_far)
    )

    best_param = fmin(
        wrapper,
        hyperparameter_space,
        algo=algo,
        max_evals=args.num_trials,
        trials=trials,
        rstate=random_state,
    )

    save_run(trials, random_state, args.path)

    loss = [x["result"]["loss"] for x in trials.trials]

    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss) * -1)
    print("Best parameters: ", best_param)
