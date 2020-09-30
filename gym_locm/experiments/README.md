# Experiments

This package contains the source-code for all experiments performed for our thesis <a href="#vieira2020a">[1]</a> 
and paper <a href="#vieira2020b">[2]</a>. They consist of: (i) a hyperparameter search script, (ii) a training 
script, (iii) a tournament runner script and a (iv) similarities script. Th steps to reproduce each experiment are
described as follows.


### How to reproduce the paper/thesis experiments

#### Section 5.3 or IV.D: Hyperparameter tuning

To perform a hyperparameter tuning, simply execute the [hyp-search.py](hyp-search.py) script:

```
usage: hyp-search.py [-h] [--approach {immediate,lstm,history}]
                     [--battle-agent {max-attack,greedy}] --path PATH
                     [--train-episodes TRAIN_EPISODES]
                     [--eval-episodes EVAL_EPISODES] [--num-evals NUM_EVALS]
                     [--seed SEED] [--processes PROCESSES]
                     [--num-trials NUM_TRIALS]
                     [--num-warmup-trials NUM_WARMUP_TRIALS]

  -h, --help            show this help message and exit
  --approach {immediate,lstm,history}, -a {immediate,lstm,history}
  --battle-agent {max-attack,greedy}, -b {max-attack,greedy}
  --path PATH, -p PATH  path to save models and results (default: None)
  --train-episodes TRAIN_EPISODES, -te TRAIN_EPISODES
                        how many episodes to train (default: 30000)
  --eval-episodes EVAL_EPISODES, -ee EVAL_EPISODES
                        how many episodes to eval (default: 1000)
  --num-evals NUM_EVALS, -ne NUM_EVALS
                        how many evaluations to perform throughout training (default: 12)
  --seed SEED           seed to use on the model, envs and hyperparameter search (default: None)
  --processes PROCESSES
                        amount of processes to use (default: 1)
  --num-trials NUM_TRIALS, -n NUM_TRIALS
                        amount of hyperparameter sets to test (default: 50)
  --num-warmup-trials NUM_WARMUP_TRIALS, -w NUM_WARMUP_TRIALS
                        amount of random hyperparameter sets to test before starting optimizing (default: 20)
```

The list and range of hyperparameted explored is available in the Appendix of our paper and in Attachment A of 
our thesis. we performed hyperparameter tunings for all combinations of `max-attack` and `greedy` battle agents 
and `immediate`, `history` and `lstm` draft approaches, using the script's default values for the optional 
parameters (except `--processes 4` and `--seed 96765`). Each run of the script took around 2 days with the
`max-attack` battle agent and more than a week with the `greedy` battle agent.

#### Section 5.4 or IV.E: Comparison between approaches

to do

#### Section 5.5 or IV.F: Comparison with other draft strategies

to do

#### Section 5.6 or IV.G: Agent improvement in the CoG 2019 LOCM tournament

We used the 
[source code of the Strategy Card Game AI competition](https://github.com/acatai/Strategy-Card-Game-AI-Competition/tree/master/contest-2019-08-COG) 
to re-run the matches, replacing the *max-attack* player (named Baseline2) with a personalized player featuring 
our best draft agent and the battle portion on the *max-attack* player. This can be reproduced by altering line 
11 of the runner script 
([run.sh](https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/contest-2019-08-COG/run.sh))
from `AGENTS[10]="python3 Baseline2/main.py"` to
```bash
AGENTS[10]="python3 gym_locm/toolbox/predictor.py --battle \"python3 Baseline2/main.py\" \
    --draft-1 path/to/gym_locm/trained_models/max-attack/immediate/1st/6.json \
    --draft-2 path/to/gym_locm/trained_models/max-attack/immediate/2nd/8.json
```
then, executing it. Paralellism can be achieved by running the script in multiple processes/machines. Save the 
output to text files named `out-*.txt` (with a number instead of `*`) in the same folder, then run 
[analyze.py](https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/contest-2019-08-COG/analyze.py) 
to extract win rates. The runner script can take up to several days, and the analyze script can take up to some hours.
See the [trained_models](https://github.com/ronaldosvieira/gym-locm/tree/master/gym_locm/trained_models) 
package for more information on the predictor script.

### References

1. <span id="vieira2020a">Vieira, R., Chaimowicz, L., Tavares, A. R. (2020). Drafting in Collectible Card Games via 
Reinforcement Learning. Master's thesis, Department of Computer Science, Federal University 
of Minas Gerais, Belo Horizonte, Brazil.</span>

2. <span id="vieira2020b">Vieira, R., Tavares, A. R., Chaimowicz, L. (2020). Drafting in 
Collectible Card Games via Reinforcement Learning. 19th Brazilian Symposium of Computer Games
and Digital Entertainment (SBGames).</span>
