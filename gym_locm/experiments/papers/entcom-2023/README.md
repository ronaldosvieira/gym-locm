# Reproducing the experiments from our Entertainment Computing paper

This readme file contains the information necessary to reproduce the experiments 
from our paper in Entertainment Computing named "_Towards Sample Efficient Deep 
Reinforcement Learning in Collectible Card Games_." Although we mention in the paper 
that we use gym-locm's version 1.4.0, any future version should also suffice. Please 
contact me at [ronaldo.vieira@dcc.ufmg.br](mailto:ronaldo.vieira@dcc.ufmg.br) in case 
any of the instructions below do not work.

Note that we use [Weights and Biases (W&B)](https://wandb.ai) to orchestrate the 
execution of all of our experiments. We provide the YAML files used, but additionally
provide instructions to run individual training sessions.

## Hyperparameter search

The `hyp-search.yaml` file contains the search configuration, including hyperparameter
ranges. Having W&B installed, executing the following command on a terminal will
create a "sweep" on W&B:

```commandline
wandb sweep gym_locm/experiments/papers/entcom-2023/hyp-search.yaml
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
In our paper, we executed 25 training sessions. All the statistics can be seen on 
W&B's website, including which sets of hyperparameters yielded the best results. 
For more info on W&B sweeps, see [the docs](https://docs.wandb.ai/guides/sweeps).

## Training the base approach

Using the best set of hyperparameters found in the previous experiment, we executed 
eight training sessions of our base approach, each with a different random seed. 
To reproduce the training sessions we used for the paper, execute the following
sweep in W&B:

```commandline
wandb sweep gym_locm/experiments/papers/entcom-2023/base.yaml
```

It will function exactly as the previous sweep, but instead of using different 
hyperparameters for each run, it will use different seeds. After using all seeds,
the sweep will finish. The seeds we used were: `73667418`, `74896946`, `28835729`, 
`38458274`, `68531181`, `34553231`, `8256697`, and `79863286`.

## Training the DTO, RS1, RS2, and DI approaches

Same as the base approach, except using

```commandline
wandb sweep gym_locm/experiments/papers/entcom-2023/{approach}.yaml
```

replacing `{approach}` with either `dto`, `rs1`, `rs2`, or `di`.

## Extra: running individual training sessions

If you wish to run an individual training session, use our training script
(which is exactly what W&B does under the hood):

```commandline
python gym_locm/experiments/training.py --task=battle --version=1.5 \
--adversary=self-play --role=alternate --draft-agent=inspirai --eval-battle-agents=greedy \
--train-episodes=100000 --eval-episodes=250 --num-evals=100 --switch-freq=100 \
--act-fun=relu --cliprange=0.2 --ent-coef=0.005 --gamma=0.99 --layers=1 \
--learning-rate=0.005838104376218821 --n-steps=4096 --neurons=501 \
--nminibatches-divider=1 --noptepochs=2 --vf-coef=1 \
--use-average-deck=False --reward-functions="win-loss" --reward-weights="1" \
--path=path/of/your/choice --seed=42 --concurrency=4
```

For a comprehensive list of parameters, use `python gym_locm/experiments/training -h`.
