# causal_rl
Includes code bases for a few projects/ideas related to causal structures in RL:
- `causal_overhypotheses` is mostly based on the paper "Towards Understanding How Machines Can Learn Causal Overhypotheses" from Gopnik Lab, and its codebase `https://github.com/CannyLab/causal_overhypotheses/tree/master`.
- `first_explore_then_exploit` is inspired by the overhypothesis environment, and solves continually-changing environment with first-explore-then-exploit strategy, where in the exploration stage the agent collects observations from the symbolic blicket detector environment, extracts causal graph with bayes net, then in the exploitation stage the agent maximizes reward using inference from the causal graph. 
- `conspec` includes code related to using conspec (Sun et al., 2023 NeurIPS) to discover and test causal structures in the environments, which in this codebase includes MiniGrid (main branch) and MiniWorld (miniworld branch). It contains two subfolders:
    - `grid_blicket` focuses on training regular conspec on MiniGrid/Miniworld
    - `hypothesis` focuses on post-conspec training, assuming we have mature prototypes to form hypotheses, and trains hypothesis-conditioned policies to test hypothesis in environments in order to study exploration and generalization. (STILL VERY MUCH WORK IN PROGRESS!)

See each subfolder's README for instructions.

