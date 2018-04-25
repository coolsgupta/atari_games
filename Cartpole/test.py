import gym

env = gym.make('MountainCar-v0')
state = env.reset()
print(state)
print(state[0])
print(env.observation_space)
print(env.step(0))
