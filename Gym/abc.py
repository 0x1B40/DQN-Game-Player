import gym
import time  # Optional: to slow down the rendering for visibility

env = gym.make("CartPole-v1")

observation = env.reset(seed=42)

for _ in range(1000):
    env.render()  # 👈 This line opens a pop-up window with the game

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    time.sleep(0.01)  # 👈 Optional: slow down to see movement

    if done:
        observation = env.reset()

env.close()
