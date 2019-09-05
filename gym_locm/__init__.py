from gym.envs.registration import register

register(id='LoCM-draft-v0',
         entry_point='gym_locm.envs:LoCMDraftSingleEnv')
register(id='LoCM-draft-2p-v0',
         entry_point='gym_locm.envs:LoCMDraftEnv')
