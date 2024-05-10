import gym
import numpy as np
import itertools
import random

from gym import spaces

from typing import Dict, Any, Tuple, List, Set


class Hypothesis():
    @property
    def blickets(self) -> Set[int]:
        ...

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        ...


### Base Conjunctive Hypotheses for 3 blickets ###


class ConjunctiveHypothesis:
    blickets = None
    structure = "conjunctive"

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        return all(c in blickets for c in cls.blickets)  # type: ignore


class ABconj(ConjunctiveHypothesis):
    blickets = set([0, 1])


class ACconj(ConjunctiveHypothesis):
    blickets = set([0, 2])


class BCconj(ConjunctiveHypothesis):
    blickets = set([1, 2])


class ABCconj(ConjunctiveHypothesis):
    blickets = set([0, 1, 2])


### Base Disjunctive Hypotheses for 3 blickets ###


class DisjunctiveHypothesis:
    blickets = None
    structure = "disjunctive"

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        return any(c in blickets for c in cls.blickets)  # type: ignore


class Adisj(DisjunctiveHypothesis):
    blickets = set([0])


class Bdisj(DisjunctiveHypothesis):
    blickets = set([1])


class Cdisj(DisjunctiveHypothesis):
    blickets = set([2])


class ABdisj(DisjunctiveHypothesis):
    blickets = set([0, 1])


class ACdisj(DisjunctiveHypothesis):
    blickets = set([0, 2])


class BCdisj(DisjunctiveHypothesis):
    blickets = set([1, 2])


class ABCdisj(DisjunctiveHypothesis):
    blickets = set([0, 1, 2])


class CausalEnv_v0(gym.Env):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        """
        Representation of the Blicket environment, based on the exeperiments presente in the causal learning paper.

        Args:
            env_config (Dict[str, Any]): A dictionary representing the environment configuration.
                Keys: Values (Default)


        Action Space:
            => [Object A (on/off), Object B state (on/off), Object C state (on/off)]

        """

        self._n_blickets = env_config.get("n_blickets", 3)  # Start with 3 blickets
        self._reward_structure = env_config.get("reward_structure", "baseline")  # Start with baseline reward structure
        self._symbolic = env_config.get("symbolic", True)  # Start with symbolic observation space

        if self._reward_structure not in ("baseline", "quiz", "quiz-type", "quiz-typeonly"):
            raise ValueError(
                "Invalid reward structure: {}, must be one of (baseline, quiz, quiz-type, quiz-typeonly)".format(self._reward_structure)
            )

        # Setup penalties and reward structures
        self._add_step_reward_penalty = env_config.get("add_step_reward_penalty", False)
        self._add_detector_state_reward_for_quiz = env_config.get("add_detector_state_reward_for_quiz", False)
        self._step_reward_penalty = env_config.get("step_reward_penalty", 0.01)
        self._detector_reward = env_config.get("detector_reward", 1)
        self._quiz_positive_reward = env_config.get("quiz_positive_reward", 1)
        self._quiz_negative_reward = env_config.get("quiz_negative_reward", -1)
        self._max_baseline_steps = env_config.get("max_baseline_steps", 20)  # max number of steps agent can take to explore, before forced to enter quiz stage
        self._blicket_dim = env_config.get("blicket_dim", 3)
        self._quiz_disabled_steps = env_config.get("quiz_disabled_steps", -1)

        assert self._max_baseline_steps >= self._quiz_disabled_steps, "Max baseline steps must be greater than quiz-disabled steps."

        # Gym environment setup
        self.action_space = (
            spaces.MultiDiscrete([2] * self._n_blickets)
            if 'quiz' not in self._reward_structure
            else spaces.MultiDiscrete([2] * (self._n_blickets + 1))
        )

        if self._symbolic:
            if 'quiz' in self._reward_structure:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 2,), dtype=np.float32
                )  # The state of all of the blickets, plus the state of the detector plus the quiz indicator
            else:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 1,), dtype=np.float32
                )  # The state of all of the blickets, plus the state of the detector
        else:
            self._blicket_cmap = {i: np.random.uniform(self._blicket_dim) for i in range(self._n_blickets)}
            if 'quiz' in self._reward_structure:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 2, self._blicket_dim), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 1, self._blicket_dim), dtype=np.float32
                )

        # Add the hypothesis spaces
        self._hypotheses: List[Hypothesis] = env_config.get(
            "hypotheses",
            [
                ABconj,
                ACconj,
                BCconj,
                # ABCconj,
                Adisj,
                Bdisj,
                Cdisj,
                # ABdisj,
                # ACdisj,
                # BCdisj,
                # ABCdisj,
            ],
        )

        # Setup the environment by default
        self._current_gt_hypothesis = random.choice(self._hypotheses)

        self._steps = 0
        self._observations = 0
        self._quiz_step = None
        self.quiz_reward = 0
        self.type_reward = 0

        self.reset()

    def reset(self) -> np.ndarray:
        # Reset the color map for the blickets
        self._current_gt_hypothesis = random.choice(self._hypotheses)
        if "quiz" in self._reward_structure:
            # Reset the color map
            self._blicket_cmap = {i: np.random.uniform(self._blicket_dim) for i in range(self._n_blickets)}

        # Reset the step trackers
        self._steps = 0
        self._quiz_step = None
        self.quiz_reward = 0
        self.type_reward = 0

        # Get the baseline observation
        return self._get_observation(blickets=np.zeros(self._n_blickets))

    def _get_baseline_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._symbolic:
            return np.concatenate(
                [
                    blickets,
                    np.array([1]) if self._get_detector_state(blickets) else np.array([0]),
                ],
                axis=0,
            ) # type: ignore
        return np.concatenate(
            [
                np.stack(
                    [
                        self._blicket_cmap[i] if blickets[i] == 0 else np.zeros(self._blicket_dim)
                        for i in range(self._n_blickets)
                    ],
                    axis=0,
                ),
                np.ones((1, self._blicket_dim))
                if self._get_detector_state(blickets)
                else np.zeros((1, self._blicket_dim)),
            ],
            axis=0,
        ) # type: ignore

    def _get_quiz_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._quiz_step is not None:
            if self._symbolic:
                return np.concatenate(
                    [
                        np.array([1 if self._quiz_step == i else 0 for i in range(self._n_blickets)]),
                        np.array([0]),
                        np.array([1]),
                    ],
                    axis=0,
                )
            else:
                return np.concatenate(
                    [
                        np.stack(
                            [
                                self._blicket_cmap[i] if self._quiz_step == i else np.zeros(self._blicket_dim)
                                for i in range(self._n_blickets)
                            ],
                            axis=0,
                        ),
                        np.zeros((1, self._blicket_dim)),
                        np.ones((1, self._blicket_dim)),
                    ],
                    axis=0,
                )  # type: ignore

        if self._symbolic:
            return np.concatenate(
                [
                    blickets,  # Blickets
                    np.array([1] if self._get_detector_state(blickets) else [0]),  # Detector state
                    np.array([0]),  # Quiz indicator
                ],
                axis=0,
            )  # type: ignore
        return np.concatenate(
            [
                np.stack(
                    [
                        self._blicket_cmap[i] if blickets[i] == 1 else np.zeros(self._blicket_dim)
                        for i in range(self._n_blickets)
                    ],
                    axis=0,
                ),
                np.zeros((1, self._blicket_dim)),  # Detector state
                np.zeros((1, self._blicket_dim)),  # Quiz indicator
            ],
            axis=0,
        )  # type: ignore

    def _get_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._reward_structure == "baseline":
            return self._get_baseline_observation(blickets)
        elif 'quiz' in self._reward_structure:
            return self._get_quiz_observation(blickets)
        raise ValueError("Invalid reward structure: {}".format(self._reward_structure))

    def _get_detector_state(self, active_blickets: np.ndarray) -> bool:
        blickets_on = set()
        for i in range(len(active_blickets)):
            if active_blickets[i] == 1:
                blickets_on.add(i)
        return self._current_gt_hypothesis.test(blickets_on)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        observation, reward, done, info = (None, 0, False, {})

        # Generate the observations and reward
        if self._reward_structure == "baseline":
            observation = self._get_baseline_observation(action[: self._n_blickets])

            # Get the reward
            if self._add_step_reward_penalty:
                reward -= self._step_reward_penalty
            if self._get_detector_state(action[: self._n_blickets]):
                reward += self._detector_reward
            done = self._steps > self._max_baseline_steps

        elif 'quiz' in self._reward_structure:
            if self._quiz_step is not None:

                # Get the reward
                if self._reward_structure == 'quiz':
                    reward = (
                        self._quiz_positive_reward
                        if (
                            action[self._quiz_step] == 1
                            and self._quiz_step in self._current_gt_hypothesis.blickets
                            or action[self._quiz_step] == 0
                            and self._quiz_step not in self._current_gt_hypothesis.blickets
                        )
                        else self._quiz_negative_reward
                    )
                    self.quiz_reward = reward
                    self.type_reward = 0
                elif self._reward_structure in ('quiz-type', 'quiz-typeonly'):
                    if self._quiz_step < self._n_blickets:
                        reward = (
                        self._quiz_positive_reward
                        if (
                            action[self._quiz_step] == 1
                            and self._quiz_step in self._current_gt_hypothesis.blickets
                            or action[self._quiz_step] == 0
                            and self._quiz_step not in self._current_gt_hypothesis.blickets
                        )
                        else self._quiz_negative_reward
                    )
                        self.quiz_reward = reward
                        self.type_reward = 0
                    else:
                        reward = (
                            0.5
                            if (action[0] == 0 and issubclass(self._current_gt_hypothesis, ConjunctiveHypothesis) or action[0] == 1 and issubclass(self._current_gt_hypothesis, DisjunctiveHypothesis))
                            else -0.5
                        )
                        self.quiz_reward = 0
                        self.type_reward = reward



                # We're in the quiz phase.
                self._quiz_step += 1
                observation = self._get_quiz_observation(action[: self._n_blickets])

                if self._reward_structure in ('quiz-type', 'quiz-typeonly'):
                    if self._quiz_step > self._n_blickets:
                        done = True
                elif self._reward_structure == 'quiz':
                    if self._quiz_step >= self._n_blickets:
                        done = True
            else:
                # Check the action to see if we should go to quiz phase
                if (self._steps > self._max_baseline_steps) or (action[-1] == 1 and self._steps > self._quiz_disabled_steps):

                    if self._add_step_reward_penalty:
                        reward -= self._step_reward_penalty
                    if self._add_detector_state_reward_for_quiz and self._get_detector_state(
                        action[: self._n_blickets]
                    ):
                        reward += self._detector_reward

                    # Go to quiz phase
                    self._quiz_step = 0 if self._reward_structure != 'quiz-typeonly' else self._n_blickets
                    observation = self._get_quiz_observation(action[: self._n_blickets])
                else:
                    # We're in the standard action phase
                    observation = self._get_quiz_observation(action[: self._n_blickets])


        assert observation is not None

        self._steps += 1
        return observation, reward, done, info
    

    def get_quiz_reward(self):
        """
        Return the current quiz reward
        """
        return self.quiz_reward

    def get_type_reward(self):
        """
        Return the current type reward
        """
        return self.type_reward




class CausalEnv_v1(gym.Env):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        """
        Representation of the Blicket environment, based on the exeperiments presente in the causal learning paper.
        Compared to v0, here the action space is changed to be intervention. Reward structure for quiz setting is also changed.

        Args:
            env_config (Dict[str, Any]): A dictionary representing the environment configuration.
                Keys: Values (Default)


        Action Space: Intervention. Choose one of the three objects to intervene on by changing its state to 0 or 1
            => [Object A , Object B , Object C , target state]

        """

        self._n_blickets = env_config.get("n_blickets", 3)  # Start with 3 blickets
        self._reward_structure = env_config.get("reward_structure", "baseline")  # Start with baseline reward structure
        self._symbolic = env_config.get("symbolic", True)  # Start with symbolic observation space

        if self._reward_structure not in ("baseline", "quiz", "quiz-type", "quiz-typeonly"):
            raise ValueError(
                "Invalid reward structure: {}, must be one of (baseline, quiz, quiz-type, quiz-typeonly)".format(self._reward_structure)
            )

        # Setup penalties and reward structures
        self._add_step_reward_penalty = env_config.get("add_step_reward_penalty", False)
        self._add_detector_state_reward_for_quiz = env_config.get("add_detector_state_reward_for_quiz", False)
        self._step_reward_penalty = env_config.get("step_reward_penalty", 0.01)
        self._detector_reward = env_config.get("detector_reward", 0)  # No detector reward in this version
        self._quiz_positive_reward = env_config.get("quiz_positive_reward", 1)
        self._quiz_negative_reward = env_config.get("quiz_negative_reward", -1)
        self._max_steps = env_config.get("max_steps", 100)  # max number of steps agent can take to explore, before forced to enter quiz stage
        self._blicket_dim = env_config.get("blicket_dim", 3)
        self._quiz_disabled_steps = env_config.get("quiz_disabled_steps", -1)

        assert self._max_steps >= self._quiz_disabled_steps, "Max baseline steps must be greater than quiz-disabled steps."

        # Gym environment setup
        self.action_space = spaces.Tuple((
            spaces.Discrete(self._n_blickets),  # One-hot encoded selection of objects
            spaces.Discrete(2)                    # Binary decision for action
        ))  # assume baseline for now

        if self._symbolic:
            if 'quiz' in self._reward_structure:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 1,), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 1,), dtype=np.float32
                )  # The state of all of the blickets, plus the state of the detector
        else:
            pass  # assume symbolic for now

        # Add the hypothesis spaces
        self._hypotheses: List[Hypothesis] = env_config.get(
            "hypotheses",
            [
                ABconj,
                ACconj,
                BCconj,
                # ABCconj,
                Adisj,
                Bdisj,
                Cdisj,
                # ABdisj,
                # ACdisj,
                # BCdisj,
                # ABCdisj,
            ],
        )

        # Setup the environment by default
        self._current_gt_hypothesis = random.choice(self._hypotheses)
        self._steps = 0
        self._observations = 0
        self._quiz_step = None

        self.reset()

    def reset(self) -> np.ndarray:
        # Reset the color map for the blickets
        self._current_gt_hypothesis = random.choice(self._hypotheses)

        # Reset the step trackers
        self._steps = 0
        self._quiz_step = None
        self.blickets = np.zeros(self._n_blickets)  # states of the object

        # Get the baseline observation
        return self._get_observation(blickets=self.blickets)
    
    def _get_baseline_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._symbolic:
            return np.concatenate(
                [
                    blickets,
                    np.array([1]) if self._get_detector_state(blickets) else np.array([0]),
                ],
                axis=0,
            )
        else:
            pass  # assume symbolic for now

    def _get_quiz_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._symbolic:
            return np.concatenate(
                [
                    blickets,
                    np.array([1]) if self._get_detector_state(blickets) else np.array([0]),
                ],
                axis=0,
            )
        else:
            pass  # assume symbolic for now


    def _get_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._reward_structure == "baseline":
            return self._get_baseline_observation(blickets)
        elif 'quiz' in self._reward_structure:
            return self._get_quiz_observation(blickets)
        raise ValueError("Invalid reward structure: {}".format(self._reward_structure))
    
    def _get_detector_state(self, active_blickets: np.ndarray) -> bool:
        blickets_on = set()
        for i in range(len(active_blickets)):
            if active_blickets[i] == 1:
                blickets_on.add(i)
        return self._current_gt_hypothesis.test(blickets_on)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        observation, reward, done, info = (None, 0, False, {})

        # Generate the observations and reward
        if self._reward_structure == "baseline":
            # note that action is now intervention
            self.blickets[action[0]] = action[-1]
            observation = self._get_baseline_observation(self.blickets)

            # Get the reward
            if self._add_step_reward_penalty:
                reward -= self._step_reward_penalty
            if self._get_detector_state(self.blickets):
                reward += self._detector_reward
            done = self._steps > self._max_steps

        elif 'quiz' in self._reward_structure:
            # note that action is now intervention
            self.blickets[action[0]] = action[-1]
            observation = self._get_baseline_observation(self.blickets)

            if self._add_step_reward_penalty:
                reward -= self._step_reward_penalty
            if self._get_detector_state(self.blickets):
                reward += self._detector_reward

            # Give quiz reward
            if set(np.where(self.blickets == 1)[0]) == self._current_gt_hypothesis.blickets:
                reward += self._quiz_positive_reward
                done = True
            else:
                done = self._steps > self._max_steps

        assert observation is not None

        self._steps += 1
        return observation, reward, done, info
    

def test_env():
    args = {
        'num_trajectories': 10,
        'max_steps': 20,
        'quiz_disabled_steps': -1,
        'reward_structure': 'quiz',
        'add_step_reward_penalty': True
    }

    env = CausalEnv_v1({
        "reward_structure":  args['reward_structure'],
        "quiz_disabled_steps": args['quiz_disabled_steps'],
        "max_steps": args["max_steps"],
        "add_step_reward_penalty": args["add_step_reward_penalty"]
    })

    # Roll out the environment for k trajectories
    print('Collecting Trajectories...')
    trajectories = []
    for i in range(args['num_trajectories']):
        print(f"====== Episode {i+1} ==========")
        # Reset the environment
        obs = env.reset()
        gt = str(env._current_gt_hypothesis).split("'")[1].split('.')[-1]
        print(f"Ground truth is {gt}")

        # Roll out the environment for n steps
        steps = []
        for j in range(args['max_steps']):
            print(f"-- step {j+1} --")
            # Get the action from the model
            action = env.action_space.sample()
            print(f"sampled action {action}")

            # Step the environment
            n_obs, reward, done, info = env.step(action)
            print(f"received reward {reward}. Next observation will be {n_obs}.")

            steps.append((obs, action, reward, n_obs, done))
            obs = n_obs

            # Check if the episode has ended
            if done:
                break

if __name__ == "__main__":
    test_env()