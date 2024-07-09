import random

import gym

import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

c_learning_rate = 0.1
c_discount_value = 0.9
Q_table_size = [20, 20]
Q_table_segment_size = (env.observation_space.high - env.observation_space.low) / Q_table_size
c_no_of_eps = 10000
c_show_each = 10000

v_epsilon = 0.9
c_start_eps_epsilon_decay = 1
c_end_eps_epsilon_decay = c_no_of_eps // 2
v_epsilon_decay = v_epsilon / (c_end_eps_epsilon_decay - c_start_eps_epsilon_decay)
def convert_state(real_state):
    Q_state = (real_state - env.observation_space.low) // Q_table_segment_size
    return tuple(Q_state.astype(np.int_))

q_table = np.random.uniform(low=-2, high=0, size=(Q_table_size + [env.action_space.n]))

max_eps_reward = -999
max_eps_action_list = []
max_start_state = None
# print(q_table.shape)
# print(convert_state(env.reset()))
for eps in range(c_no_of_eps):
    print("Eps = ", eps)
    done = False
    current_state = convert_state(env.reset())

    eps_reward = 0
    action_list = []
    eps_start_state = current_state


    if eps % c_show_each == 0:
        show_hang = True
    else:
        show_hang = False

    while not done:
        if random.random() > v_epsilon:

            # Lấy argmax Q value của current_state
            action = np.argmax(q_table[current_state])
        else:
            action = random.randint(0, env.action_space.n - 1)

        action_list.append(action)

        # Hành động theo action đã lấy
        new_real_state, reward, done, _ = env.step(action=action)
        eps_reward += reward

        if show_hang:
            env.render()

        if done:
            if new_real_state[0] >= env.goal_position:
                print("Đã hoàn thành trò chơi tại eps = {}, reward = {}".format(eps, eps_reward))
                if eps_reward > max_eps_reward:
                    max_eps_reward = eps_reward
                    max_eps_action_list = action_list
                    max_start_state = eps_start_state

        else:

            # Convert về q_state
            new_q_state = convert_state(new_real_state)

            # Update Q value cho current_state và action
            current_q_value = q_table[current_state + (action,)]

            new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * np.max(q_table[new_q_state]))

            q_table[current_state + (action,)] = new_q_value

            current_state = new_q_state

    if c_end_eps_epsilon_decay >= eps > c_start_eps_epsilon_decay:
        v_epsilon = v_epsilon - v_epsilon_decay


print("Max reward = ", max_eps_reward)
print("Max action_list = ", max_eps_action_list)

env.reset()
env.state = max_start_state
for action in max_eps_action_list:
    env.step(action)
    env.render()

done = False
while not done:
    _, _, done, _ = env.step(0)
    env.render()