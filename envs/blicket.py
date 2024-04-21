
class BlicketEnv:
    def __init__(self, true_hypothesis):
        # Represent the ground truth causal structure
        self.hypotheses_space = [
            "A-dis", "B-dis", "C-dis", "AB-dis", "AC-dis", "BC-dis", "ABC-dis",
            "AB-con", "AC-con", "BC-con", "ABC-con"]
        self.action_space = ["A", "B", "C", "AB", "BC", "AC", "ABC"]
        self.true_hypothesis = true_hypothesis
        assert self.true_hypothesis in self.hypotheses_space

    def step(self, action):
        assert action in self.action_space
        # Take action and return observation
        if self.true_hypothesis.endswith("con"):
            if len(action) < 2: # if true hypothesis is conjunctive, then it requires two blickets to turn on the detector
                return 0
            elif set(self.true_hypothesis[:-4]).issubset(set(action)):  # if action contains all the blickets in the hypothesis
                return 1
            else:
                return 0
        # if true hypothesis is disjunctive, then the detector turns on if any of the blickets in the hypothesis is present
        else:
            if set(action).intersection(set(self.true_hypothesis[:-4])):  # if action contains any of the blickets in the hypothesis
                return 1
            else:
                return 0

    def reset(self):
        pass # No need for resetting in the current setup

