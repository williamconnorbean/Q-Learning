# Q-Learning

This is a demonstration of the reinforcement learning technique Q-Learning. Implemented using the [OpenAI Gym](https://gym.openai.com/) toolkit, using the [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/) environment.

The FrozenLake-v0 environment is modeled as:
```
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
```

## Usage
```
$ python qLearning.py
```

## Example Output
```
Average reward per 1000 episodes
===============================
1000: 0.05300000000000004
2000: 0.21500000000000016
3000: 0.4090000000000003
4000: 0.5670000000000004
5000: 0.6290000000000004
6000: 0.6430000000000005
7000: 0.6620000000000005
8000: 0.6790000000000005
9000: 0.6890000000000005
10000: 0.6950000000000005


Final Q-Table Values
    Left        Down        Right       Up
[[0.56010858 0.49226986 0.48920864 0.50856735]
 [0.27832833 0.40573726 0.31036374 0.45322622]
 [0.38643823 0.3084546  0.27337718 0.27981723]
 [0.08178549 0.20387031 0.04341759 0.09195054]
 [0.57824327 0.38639214 0.36705414 0.41405289]
 [0.         0.         0.         0.        ]
 [0.13736415 0.1841101  0.35332989 0.08445062]
 [0.         0.         0.         0.        ]
 [0.46408042 0.3572033  0.41852641 0.60969682]
 [0.43086111 0.6501073  0.52092162 0.48552732]
 [0.59208557 0.43273842 0.4484966  0.23052023]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.45910196 0.61855928 0.78023698 0.47311013]
 [0.73479435 0.87754405 0.78986941 0.7948204 ]
 [0.         0.         0.         0.        ]]
 ```

The average reward per 1000 episodes represents the average probability that the agent reaches the goal state within the 1000 episode time frame. This means that in the first 1000 episodes, the agent only reaches the goal state approximately 5% of the time. However, after 10,000 episodes of training, the agent reaches the goal state approximately 70% of the time. This is a significant improvement given that the agent does not know anything about the enviroment at the start of training and given it only receives a reward of +1 if it reaches the goal state.
