from gym.envs.registration import register

register(
    id='TradeSim-v0',
    entry_point='gym_tradesim.envs:TradeSimEnv',
    timestep_limit=1000,
)
