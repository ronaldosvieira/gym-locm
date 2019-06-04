from gym.envs.registration import register

register(id='LoCM-v0',
         entry_point='gym_locm.envs:LoCMEnv')
