from gym.envs.registration import register

register(id='LoCM-draft-v0',
         entry_point='gym_locm.envs:LoCMDraftEnv')
register(id='LoCM-draft-single-v0',
         entry_point='gym_locm.envs:LoCMDraftSingleEnv')
