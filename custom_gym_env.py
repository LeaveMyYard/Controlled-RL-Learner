import gym
from gym import error, spaces, utils
from gym.utils import seeding
from numba import jit
import numpy as np

from cuda_grayscale import grayscale


class Pacman(gym.ObservationWrapper):
    def __init__(self):
        super().__init__(gym.make("MsPacman-v4"))
        self.observation_space = gym.spaces.Box(0, 255, [70, 80, 4,],)
        self.action_space = self.env.action_space

        self._frames_list = []

    def observation(self, observation):
        frame = (
            np.mean(observation, 2, keepdims=True).reshape((70, 3, 80, 2)).max(3).max(1)
        )
        if self._frames_list == []:
            for _ in range(3):
                self._frames_list.append(frame)

        self._frames_list.append(frame)
        self._frames_list = self._frames_list[-4:]

        f = np.array(self._frames_list)

        return np.moveaxis(f, 0, -1)


class SpaceInvaders(gym.ObservationWrapper):
    def __init__(self):
        super().__init__(gym.make("SpaceInvaders-v0"))
        self.observation_space = gym.spaces.Box(0, 255, [210, 160, 4,],)
        self.action_space = self.env.action_space

        self._frames_list = []

    def observation(self, observation):
        # frame = np.mean(observation, 2)
        frame = grayscale(observation)
        if self._frames_list == []:
            for _ in range(3):
                self._frames_list.append(frame)

        self._frames_list.append(frame)
        self._frames_list = self._frames_list[-4:]

        f = np.array(self._frames_list)

        return np.moveaxis(f, 0, -1)
