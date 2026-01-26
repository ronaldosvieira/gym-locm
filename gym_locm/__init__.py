from gymnasium import register

register(id="LOCM-draft-v0", entry_point="gym_locm.envs:LOCMDraftSingleEnv")

register(id="LOCM-draft-2p-v0", entry_point="gym_locm.envs:LOCMDraftEnv")

register(id="LOCM-constructed-v0", entry_point="gym_locm.envs:LOCMConstructedSingleEnv")

register(id="LOCM-constructed-2p-v0", entry_point="gym_locm.envs:LOCMConstructedEnv")

register(id="LOCM-battle-v0", entry_point="gym_locm.envs:LOCMBattleSingleEnv")

register(id="LOCM-battle-2p-v0", entry_point="gym_locm.envs:LOCMBattleEnv")
