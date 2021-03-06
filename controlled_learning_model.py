import os
import re
import pickle
import typing

import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime


class LearningModel:
    """
        A class that processes all the learning work with RL models, 
        including training in epochs, training in between, saving data,
        building plots and loading previous epoches if model started to
        get worse for some reason.
    """

    def __init__(
        self,
        environment,
        algorithm,
        policy,
        policy_kwargs: typing.Dict = None,
        load_saved: bool = True,
        save_location: typing.Union[str, None] = None,
        reward_preprocess_function: typing.Callable[[float], float] = lambda x: x,
    ):
        self.environment = environment
        self.algorithm = algorithm
        self.policy = policy

        self.save_location = save_location

        self.current_epoch = 1
        self.__reward_preprocess_function = reward_preprocess_function

        loaded = False
        if load_saved:
            if save_location is None:
                raise ValueError(
                    f"If load_saved is True you have to provide save_location"
                )

            self._directory = os.path.join(
                *[
                    "results",
                    save_location,
                    self.algorithm.__name__,
                    self.policy.__name__,
                ]
            )

            if os.path.exists(self._directory):
                files = [
                    int(f[:-4])
                    for f in os.listdir(self._directory)
                    if os.path.isfile(os.path.join(self._directory, f))
                    and re.search(r"^\d+.pkl$", f) is not None
                ]

                if files != []:
                    self.model = self.algorithm.load(
                        os.path.join(self._directory, f"{max(files)}.pkl"),
                        env=self.environment,
                    )
                    print(f"Loaded {max(files)}.pkl")
                    loaded = True
                    self.current_epoch = max(files) + 1

                try:
                    with open(os.path.join(self._directory, "data.pickle"), "rb") as f:
                        self._epoch_tests = pickle.load(f)
                except OSError:
                    self._epoch_tests = {}

        if not loaded:
            if policy_kwargs is not None:
                self.model = self.algorithm(
                    self.policy, self.environment, **policy_kwargs
                )
            else:
                self.model = self.algorithm(self.policy, self.environment, verbose=1)

    @staticmethod
    def __percentage_difference(a: float, b: float) -> float:
        return abs((a - b) / a) * 100 if b not in (a, 0) else 0

    def launch(
        self,
        verbose: int = 0,
        train_size: int = 10 ** 6,
        epoches: int = 1000,
        retrain_on_test_down: bool = False,
        retrain_loss_amount: float = 10,
        test_type: str = "dones",
        test_size: int = 10 ** 4,
        test_epoches: int = 1,
        reload_previous_max_amount: int = 5,
    ):
        result = self._test_epoch(
            test_type=test_type,
            test_num=self.current_epoch - 1,
            steps_amount=test_size,
            test_epoches=test_epoches,
        )

        print(f"Initial testing result: {result}")

        for epoch in range(self.current_epoch, epoches):
            print(f"Epoch {epoch}")
            reload_amount = 0
            while True:
                max_epoch_test_reward_num, max_epoch_test_reward = max(
                    self._epoch_tests.items(), key=lambda x: x[1]
                )
                self._train_epoch(
                    epoch_num=epoch,
                    epoch_size=train_size,
                    save_location=self.save_location,
                )
                test_result = self._test_epoch(
                    test_type=test_type,
                    test_num=epoch,
                    steps_amount=test_size,
                    test_epoches=test_epoches,
                )

                print(f"Epoch {epoch} result: {test_result}")

                if not retrain_on_test_down or epoch == 1:
                    break

                if (
                    self.__percentage_difference(test_result, max_epoch_test_reward)
                    < retrain_loss_amount
                    or test_result > max_epoch_test_reward
                ):
                    break

                reload_amount += 1

                print(
                    "Current maximum test value is for epoch"
                    + f" {max_epoch_test_reward_num}: {max_epoch_test_reward}"
                )

                directory = os.path.join(
                    *[
                        "results",
                        self.save_location,
                        self.algorithm.__name__,
                        self.policy.__name__,
                    ]
                )

                if reload_amount < reload_previous_max_amount:
                    self.model = self.algorithm.load(
                        os.path.join(directory, f"{epoch-1}.pkl"), env=self.environment,
                    )
                    print(f"Reloaded previous {epoch-1}.pkl")
                else:
                    self.model = self.algorithm.load(
                        os.path.join(directory, f"{max_epoch_test_reward_num}.pkl"),
                        env=self.environment,
                    )
                    reload_amount = 0
                    print(f"Reloaded current best {max_epoch_test_reward_num}.pkl")

    def _train_epoch(
        self, epoch_num: int, epoch_size: int, save_location: typing.Union[str, None],
    ):
        self.model.learn(epoch_size)

        if save_location is not None:
            if not os.path.exists(self._directory):
                os.makedirs(self._directory)

            self.model.save(os.path.join(self._directory, f"{epoch_num}.pkl"))

    def _test_epoch(
        self,
        test_type: typing.Union["steps", "dones"],
        test_num: typing.Union[int, None] = None,
        steps_amount: int = 1000,
        test_epoches: int = 1,
    ) -> float:
        print("Running test...")
        if test_type == "steps":
            reward_total = (
                sum(
                    [
                        self.__test_epoch_steps_type(steps_amount, test_num)
                        for _ in range(test_epoches)
                    ]
                )
                / test_epoches
            )
        elif test_type == "dones":
            reward_total = (
                sum(
                    [
                        self.__test_epoch_dones_type(test_num)
                        for _ in range(test_epoches)
                    ]
                )
                / test_epoches
            )
        else:
            raise ValueError(
                f"test_type setting only accepts 'steps' or 'dones' type, but {test_type} was given"
            )

        if test_num is not None:
            self._epoch_tests[test_num] = reward_total

            with open(os.path.join(self._directory, "data.pickle"), "wb") as f:
                pickle.dump(self._epoch_tests, f)

        plt.clf()
        plt.plot(list(self._epoch_tests.keys()), list(self._epoch_tests.values()))
        plt.gcf().savefig(os.path.join(self._directory, "plot.png"))

        return reward_total

    def __test_epoch_dones_type(self, test_num) -> float:
        obs = self.environment.reset()

        reward_total = 0

        dones = None

        while dones is None or not all(dones):
            t = datetime.now()
            timer = {}

            action, _states = self.model.predict(obs)
            timer["predict"] = (datetime.now() - t).total_seconds()
            obs, rewards, current_dones, _ = self.environment.step(action)
            timer["step"] = (datetime.now() - t).total_seconds() - timer["predict"]

            if dones is None:
                dones = current_dones

            for i, current_done in enumerate(current_dones):
                dones[i] = dones[i] or current_done

            for reward, done in zip(rewards, dones):
                if not done:
                    reward_total += reward

            timer["other"] = (
                (datetime.now() - t).total_seconds() - timer["predict"] - timer["step"]
            )

            # self.environment.render()
            timer["FPS"] = 1 / (datetime.now() - t).total_seconds()

            print(
                f"\rFPS: {round(timer['FPS'], 4)},\tpredict: {round(timer['predict'], 4)},\tstep: {round(timer['step'], 4)},\tother: {round(timer['other'], 4)}\t[{100 * np.sum(dones) / len(dones)}%]",
                end="              ",
            )

        return self.__reward_preprocess_function(reward_total)

    def __test_epoch_steps_type(
        self, steps_amount: int, test_num: typing.Union[int, None] = None
    ) -> float:
        obs = self.environment.reset()

        reward_total = 0

        for i in range(steps_amount):
            t = datetime.now()
            timer = {}

            action, _states = self.model.predict(obs)
            timer["predict"] = (datetime.now() - t).total_seconds()
            obs, rewards, _, _ = self.environment.step(action)
            timer["step"] = (datetime.now() - t).total_seconds() - timer["predict"]
            reward_total += sum(rewards)
            timer["other"] = (
                (datetime.now() - t).total_seconds() - timer["predict"] - timer["step"]
            )

            # self.environment.render()

            timer["FPS"] = 1 / (datetime.now() - t).total_seconds()

            print(
                f"\rFPS: {round(timer['FPS'], 4)},\tpredict: {round(timer['predict'], 4)},\tstep: {round(timer['step'], 4)},\tother: {round(timer['other'], 4)},\t[{100 * i / steps_amount}%]",
                end="              ",
            )

        return self.__reward_preprocess_function(reward_total)
