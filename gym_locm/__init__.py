from gym.envs.registration import register

register(id="LOCM-draft-v0", entry_point="gym_locm.envs:LOCMDraftSingleEnv")

register(id="LOCM-draft-2p-v0", entry_point="gym_locm.envs:LOCMDraftEnv")

register(id="LOCM-battle-v0", entry_point="gym_locm.envs:LOCMBattleSingleEnv")

register(id="LOCM-battle-2p-v0", entry_point="gym_locm.envs:LOCMBattleEnv")
