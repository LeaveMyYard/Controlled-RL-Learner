import os
import re
import typing

import matplotlib.pyplot as plt


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

            directory = os.path.join(
                *[
                    "results",
                    save_location,
                    self.algorithm.__name__,
                    self.policy.__name__,
                ]
            )

            if os.path.exists(directory):
                files = [
                    int(f[:-4])
                    for f in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, f))
                    and re.search(r"^\d+.pkl$", f) is not None
                ]

                if files != []:
                    self.model = self.algorithm.load(
                        os.path.join(directory, f"{max(files)}.pkl"),
                        env=self.environment,
                    )
                    print(f"Loaded {max(files)}.pkl")
                    loaded = True
                    self.current_epoch = max(files) + 1

        if not loaded:
            if policy_kwargs is not None:
                self.model = self.algorithm(
                    self.policy, self.environment, **policy_kwargs
                )
            else:
                self.model = self.algorithm(self.policy, self.environment, verbose=1)

        self._epoch_tests = {}

    @staticmethod
    def __percentage_difference(a: float, b: float) -> float:
        return abs((a - b) / a) * 100 if b not in (a, 0) else 0

    def launch(
        self,
        verbose: int = 0,
        train_size: int = 10 ** 6,
        epoches: int = 1000,
        retrain_on_test_down: bool = True,
        retrain_loss_amount: float = 10,
        test_type: typing.Union["steps", "dones"] = "dones",
        test_size: int = 10 ** 4,
        reload_previous_max_amount: int = 5,
    ):
        result = self._test_epoch(
            test_type=test_type, test_num=self.current_epoch - 1, steps_amount=test_size
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
                    test_type=test_type, test_num=epoch, steps_amount=test_size
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
                    f"Current maximum test value is for epoch {max_epoch_test_reward_num}: {max_epoch_test_reward}"
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
            directory = os.path.join(
                *[
                    "results",
                    save_location,
                    self.algorithm.__name__,
                    self.policy.__name__,
                ]
            )

            if not os.path.exists(directory):
                os.makedirs(directory)

            self.model.save(os.path.join(directory, f"{epoch_num}.pkl"))

    def _test_epoch(
        self,
        test_type: typing.Union["steps", "dones"],
        test_num: typing.Union[int, None] = None,
        steps_amount: int = 1000,
    ) -> float:
        if test_type == "steps":
            return self.__test_epoch_steps_type(steps_amount, test_num)
        if test_type == "dones":
            return self.__test_epoch_dones_type(test_num)

        raise ValueError(
            f"test_type setting only accepts 'steps' or 'dones' type, but {test_type} was given"
        )

    def __test_epoch_dones_type(self, test_num) -> float:
        obs = self.environment.reset()

        reward_total = 0

        dones = None

        while dones is None or not all(dones):
            action, _states = self.model.predict(obs)
            obs, rewards, current_dones, _ = self.environment.step(action)

            if dones is None:
                dones = current_dones

            for i, current_done in enumerate(current_dones):
                dones[i] = dones[i] or current_done

            for reward, done in zip(rewards, dones):
                if not done:
                    reward_total += reward

            self.environment.render()

        reward_total = self.__reward_preprocess_function(reward_total)

        if test_num is not None:
            self._epoch_tests[test_num] = reward_total

        plt.clf()
        plt.plot(list(self._epoch_tests.keys()), list(self._epoch_tests.values()))
        plt.gcf().savefig("plot.png")

        return reward_total

    def __test_epoch_steps_type(
        self, steps_amount: int, test_num: typing.Union[int, None] = None
    ) -> float:
        obs = self.environment.reset()

        reward_total = 0

        for _ in range(steps_amount):
            action, _states = self.model.predict(obs)
            obs, rewards, _, _ = self.environment.step(action)
            reward_total += sum(rewards)
            self.environment.render()

        reward_total = self.__reward_preprocess_function(reward_total)

        if test_num is not None:
            self._epoch_tests[test_num] = reward_total

        plt.clf()
        plt.plot(list(self._epoch_tests.keys()), list(self._epoch_tests.values()))
        plt.gcf().savefig("plot.png")

        return reward_total
