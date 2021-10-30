import gym
import gym_locm
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

path = 'trained_models/battle/ppo_mask'
n_envs = 4

env = make_vec_env('LOCM-battle-v0', n_envs=n_envs)
eval_env = make_vec_env('LOCM-battle-v0', n_envs=n_envs)

model = MaskablePPO("MlpPolicy", env, gamma=1, verbose=1)

checkpoint_callback = CheckpointCallback(save_freq=10_000 // n_envs, save_path=path + '/models')

model.learn(1_000_000, callback=checkpoint_callback)

model.save('trained_models/battle/ppo_mask/final')
