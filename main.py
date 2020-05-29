import gym
import os
import matplotlib.pyplot as plt
import numpy as np

import stable_baselines.common.policies as policies
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from custom_gym_env import Pacman

from lib import LearningModel


def main():
    # multiprocess environment
    n_cpu = 64
    env = SubprocVecEnv([Pacman for i in range(n_cpu)])

    learner = LearningModel(
        env,
        PPO2,
        policies.CnnPolicy,
        save_location="Pacman_Resized_4Frames",
        reward_preprocess_function=lambda x: x / n_cpu,
    )

    learner.launch(
        train_size=10 ** 7, retrain_loss_amount=14, test_type="dones", test_epoches=6
    )


if __name__ == "__main__":
    main()
