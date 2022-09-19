# Reproducing the experiments from our SBGames 2022 paper

This readme file contains the information necessary to reproduce the experiments 
from our paper in SBGames 2022 named "_Exploring Deep Reinforcement Learning for 
Battling in Collectible Card Games_." Although we mention in the paper that we use 
gym-locm's version 1.3.0, any future version should also suffice. Please contact 
me at [ronaldo.vieira@dcc.ufmg.br](mailto:ronaldo.vieira@dcc.ufmg.br) in case any 
of the instructions below do not work.

## Experiment 1: hyperparameter search

We use Weights and Biases (W&B) to orchestrate our hyperparameter search. The 
`hyp-search.yaml` file contains the search configuration, including hyperparameter
ranges. Having W&B installed, executing the following command on a terminal will
create a "sweep" on W&B:

```commandline
wandb sweep gym_locm/experiments/sbgames-2022/hyp-search.yaml
```

This command will output a _sweep ID_, including the entity and project names. 
Save it for the next step.
From this moment on, the hyperparameter search can be observed on W&B's website.
However, no training sessions will happen until you "recruit" one or more 
computers to run the training sessions. That can be done by executing the following
command on a terminal:

```commandline
wandb agent <sweep_id>
```

Where the `sweep_id` parameter should be the _sweep ID_ saved from the output of 
the previous command. From now on, the recruited computers will run training sessions
continuously until you tell them to stop. That can be done on W&B's website or by 
issuing a CTRL + C on the terminal where the training sessions are being executed. 
In our paper, we executed 35 training sessions. All the statistics can be seen on 
W&B's website, including which sets of hyperparameters yielded the best results. 
For more info on W&B sweeps, see [the docs](https://docs.wandb.ai/guides/sweeps).

## Experiment 2: training in self-play

Using the best set of hyperparameters found in the previous experiment, we executed 
five training sessions, each with a different random seed. To reproduce the training 
sessions we used for the paper, execute the following command on a terminal:

```commandline
python gym_locm/experiments/training.py --act-fun=relu --adversary=self-play \
--cliprange=0.2 --concurrency=4 --draft-agent=random --ent-coef=0.005 \
--eval-episodes=500 --gamma=0.99 --layers=7 --learning-rate=0.0041142387646692325 \
--n-steps=512 --neurons=455 --nminibatches-divider=1 --noptepochs=1 --num-evals=100 \
--path=gym_locm/experiments/papers/sbgames-2022/self-play --role=alternate \
--seed=<seed> --switch-freq=10 --task=battle --train-episodes=100000 --vf-coef=1
```

Repeating five times, each with a different `seed` parameter. The seeds we used were:
`91577453`, `688183`, `63008694`, `4662087`, and `58793266`. 

## Experiment 3: training against a fixed battle agent

This experiment uses almost the same command as the previous:

```commandline
python gym_locm/experiments/training.py --act-fun=relu --adversary=fixed \
--battle-agent=<battle_agent> --cliprange=0.2 --concurrency=4 --draft-agent=random \
--ent-coef=0.005 --eval-episodes=500 --gamma=0.99 --layers=7 \
--learning-rate=0.0041142387646692325 --n-steps=512 --neurons=455 \
--nminibatches-divider=1 --noptepochs=1 --num-evals=100 \
--path=gym_locm/experiments/papers/sbgames-2022/fixed --role=alternate --seed=<seed> \
 --switch-freq=10 --task=battle --train-episodes=100000 --vf-coef=1
```

Repeating ten times, each with a different combination of `battle_agent` and `seed` 
parameters. The seeds we used were: `91577453`, `688183`, `63008694`, `4662087`, 
and `58793266`. The battle agents we used were `max-attack` (MA) and `greedy` (OSL).
