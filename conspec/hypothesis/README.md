# Hypothesis testing with ConSpec's prototypes

This subfolder uses scripts from ../grid_blicket, but instead of actor_critic networks that only takes in observations, it takes "hypothesis" as an additional input, which tells the agent (hypothesis-conditioned policy) which prototype to use (1), which to ignore (0), which to *not* use (-1). Hypothesis can be passed to main.py as an argument, eg. "11110022" means the first 4 prototypes are to use, 5th and 6th prototypes to ignore, last two prototypes to not use.

Still largely WIP, current challenge is to get interpretable prototypes in Minigrid so that doing prototype-based hypothesis testing makes sense.



