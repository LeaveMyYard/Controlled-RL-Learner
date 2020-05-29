import gym
import os
import matplotlib.pyplot as plt
import numpy as np

import stable_baselines.common.policies as policies
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from custom_gym_env import Pacman


def main():
    # multiprocess environment
    n_cpu = 1
    env = SubprocVecEnv(
        [
            lambda: gym.wrappers.Monitor(Pacman(), "./videos", force=True)
            for i in range(n_cpu)
        ]
    )
    rewards_list = []

    layers = [1024] * 5

    print("Initial testing...")

    model = PPO2.load("./results/Pacman_Resized_4Frames/PPO2/CnnPolicy/93.pkl", env)

    # Enjoy trained agent
    obs = env.reset()

    reward_total = 0

    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_total += sum(rewards)
        env.render()

    reward_total /= n_cpu

    rewards_list.append(reward_total)

    print(f"Initial testing finished, average reward: {reward_total}")

    env.close()


if __name__ == "__main__":
    main()
