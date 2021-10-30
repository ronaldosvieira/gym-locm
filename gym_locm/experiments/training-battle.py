import gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

import gym_locm

seed = 20211030
env = gym.make('LOCM-battle-v0', seed=seed)
model = MaskablePPO("MlpPolicy", env, gamma=1, seed=seed, verbose=1)

model.learn(1_000_000)

evaluate_policy(model, env, n_eval_episodes=100, reward_threshold=0, warn=True)

model.save('trained_models/battle/ppo_mask')

