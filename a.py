import gym

env = gym.make("MountainCar-v0")
env.reset()

# Lấy state hiện tại sau khởi tạo
print(env.state)
print(env.action_space.n)

# Biên độ dao động và vận tốc
print(env.observation_space.high)
print(env.observation_space.low)


while True:
    action = 2  # Thực hiện hành động (ví dụ: nhấn ga)
    new_state, reward, done, info = env.step(action)  # Bước tiếp theo trong môi trường
    print("New state = {}, reward = {}".format(new_state, reward))
    env.render()  # Hiển thị môi trường

    if done:
        break


env.close()