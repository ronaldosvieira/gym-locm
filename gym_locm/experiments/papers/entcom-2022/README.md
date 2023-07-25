# Reproducing the experiments from our Entertainment Computing 2022 paper

This readme file contains the information necessary to reproduce the experiments
from our paper in Entertainment Computing 2022 named "_Exploring Deep Reinforcement Learning for
Drafting in Collectible Card Games_." Please contact
me at [ronaldo.vieira@dcc.ufmg.br](mailto:ronaldo.vieira@dcc.ufmg.br) in case any
of the instructions below do not work.

The game engine for LOCM 1.2 can be found at [engine.py](../../../engine/game_state.py), which is used by the OpenAI 
Gym environments (more info on the repository's main page). The implementation of our 
approaches can be found in the experiment files mentioned below. The resulting agents can be found in the
[trained_models](../../../trained_models) folder, along with instructions on how to use them.

## Section 4.1: hyperparameter search

To perform a hyperparameter tuning, simply execute the [hyp-search.py](../../../experiments/hyp-search.py) script:

```
python3 gym_locm/experiments/hyp-search.py --approach <approach> --battle-agent <battle_agent> \
    --path hyp_search_results/ --seed 96765 --processes 4
```

The list and range of hyperparameters explored is available in the Appendix A of our paper. we performed 
hyperparameter tunings for all combinations of `<approach>` (`immediate`, `history` and `lstm`) and 
`<battle_agent>` (`ma` and `osl`). To learn about the other script's parameters, execute it with the 
`--help` flag.

## Section 4.2: comparison between our approaches

To train **two** draft agents (a 1st player and a 2nd player) with a specific draft approach and battle agent,
in asymmetric self-play, simply execute the [training.py](../../../experiments/training.py) script:

```
python3 gym_locm/experiments/training.py --approach <approach> --battle-agent <battle_agent> \
    --path training_results/ --switch-freq <switch_freq> --layers <layers> --neurons <neurons> \
    --act-fun <activation_function> --n-steps <batch_size> --nminibatches <n_minibatches> \
    --noptepochs <n_epochs> --cliprange <cliprange> --vf-coef <vf_coef> --ent-coef <ent_coef> \
    --learning-rate <learning_rate> --seed 32359627 --concurrency 4
```

We trained ten draft agents (five 1st players and five 2nd players) of each combination of `<approach>` and
`<battle_agent>`, using the best sets of hyperparameters found for them in the previous experiment. That comprises
five runs of the script, in which we used the seeds `32359627`, `91615349`, `88803987`, `83140551` and `50731732`.

To learn about the other script's parameters, execute it with the `--help` flag. Running the script with all default
parameters will train a `immediate` drafter with the `ma` battler, using the best set of hyperparameters
we found for that combination. The best set of hyperparameters for the other combinations is available in the
Appendix A of our paper.

## Section 4.3: comparison with other draft agents

To run one of the tournaments, simply execute the [tournament.py](../../../experiments/tournament.py) script:
```
python3 gym_locm/experiments/tournament.py \
    --drafters random max-attack coac closet-ai icebox chad \
        gym_locm/trained_models/<battle_agent>/immediate-1M/ \
        gym_locm/trained_models/<battle_agent>/lstm-1M/ \
        gym_locm/trained_models/<battle_agent>/history-1M/ \
    --battler <battle_agent> --concurrency 4 --games 1000 --path tournament_results/ \
    --seeds 32359627 91615349 88803987 83140551 50731732
```
replacing `<battle_agent>` for either `ma` or `osl`, respectively, to run either tournament as
depicted in our paper. The script will create files at `tournament_results/` describing
the individual win rates of every set of matches, the aggregate win rates, average mana curves (section 4.3.2) 
and every individual draft choice made by every agent, in CSV format, for human inspection, and as serialized 
Pandas data frames (PKL format), for easy further data manipulation. To learn about the other script's
parameters, execute it with the `--help` flag.

To reproduce the table of agent similarities and the plot containing the agent's three-dimensional coordinates 
found via Principal Component Analysis and grouped via K-Means (section 4.3.3), simply execute the 
[similarities.py](../../../experiments/similarities.py) script:
```
python3 gym_locm/experiments/similarities.py \
  --files ma_tournament_results/choices.csv osl_tournament_results/choices.csv
```
which will result in files containing the similarities table (in CSV and PKL formats) and the plot (in PNG format)
created to the current folder.

## Section 4.4: agent improvement in the SCGAI competition

We used the source code of the Strategy Card Game AI competition
([2019](https://github.com/acatai/Strategy-Card-Game-AI-Competition/tree/master/contest-2019-08-COG) and
[2020](https://github.com/acatai/Strategy-Card-Game-AI-Competition/tree/master/contest-2020-08-COG) editions)
to re-run the matches, replacing the *max-attack* player (named Baseline2) with a personalized player featuring
our best draft agent and the battle portion on the *max-attack* player. This can be reproduced by altering line
11 (2019) or line 2 (2020) of the runner script (`run.sh`) from `AGENTS[10]="python3 Baseline2/main.py"` to
```bash
AGENTS[10]="python3 gym_locm/toolbox/predictor.py --battle \"python3 Baseline2/main.py\" \
    --draft-1 path/to/gym_locm/trained_models/max-attack/immediate-1M/1st/6.json \
    --draft-2 path/to/gym_locm/trained_models/max-attack/immediate-1M/2nd/8.json"
```
then, executing it. Parallelism can be achieved by running the script in multiple processes/machines. Save the
output to text files named `out-*.txt` (with a number instead of `*`) in the same folder, then run `analyze.py`
to extract win rates. The runner script can take up to several days, and the analyze script can take up to some hours.
See the [trained_models](../../../trained_models) package for more information on the predictor script.

## Section 4.5: importance of being history-aware in LOCM

This experiment is simply a re-execution of the OSL tournament from section 4.2, adding a new draft agent to the 
tournament (`historyless`). To reproduce it, execute the following script:
```
python3 gym_locm/experiments/tournament.py \
    --drafters random max-attack coac closet-ai icebox chad historyless \
        gym_locm/trained_models/<battle_agent>/immediate-1M/ \
        gym_locm/trained_models/<battle_agent>/lstm-1M/ \
        gym_locm/trained_models/<battle_agent>/history-1M/ \
    --battler osl --concurrency 4 --games 1000 --path osl_historyless_tournament_results/ \
    --seeds 32359627 91615349 88803987 83140551 50731732
```