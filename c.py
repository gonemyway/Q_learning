import gym

env = gym.make("MountainCar-v0")
env.reset()

print(env.state)
env.render()
env.close()