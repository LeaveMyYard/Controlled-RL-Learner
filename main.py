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
    n_cpu = 64
    env = SubprocVecEnv([Pacman for i in range(n_cpu)])
    rewards_list = []

    layers = [1024] * 5

    # model = PPO2(
    #     policies.CnnPolicy,
    #     env,
    #     verbose=1,
    #     tensorboard_log="./tensorboard/",
    #     # policy_kwargs=dict()
    #     # policy_kwargs={"net_arch": [dict(vf=layers, pi=layers)]},
    # )

    model = PPO2.load("./results/86.pkl", env, tensorboard_log="./tensorboard/")

    print("Initial testing...")

    # Enjoy trained agent
    obs = env.reset()

    reward_total = 0

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_total += sum(rewards)
        env.render()

    reward_total /= n_cpu

    rewards_list.append(reward_total)

    print(f"Initial testing finished, average reward: {reward_total}")

    for epoch in range(87, 1000):
        print(f"\nRunning epoch {epoch}.")

        model.learn(1000000)

        model.save(f"./results/{epoch}.pkl")
        print("Epoch training finished, testing...")

        # Enjoy trained agent
        obs = env.reset()

        reward_total = 0

        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            reward_total += sum(rewards)
            env.render()

        reward_total /= n_cpu

        rewards_list.append(reward_total)

        plt.clf()
        plt.plot(rewards_list)
        plt.gcf().savefig("plot.png")

        print(f"Epoch testing finished, total reward: {reward_total}")


if __name__ == "__main__":
    main()
