# Q-Learning Implementation

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

# Movement
# left: 0, down: 1, right: 2, up: 3

import gym
import numpy
import random

# Q-Learning global vars
epsilon = 1
learningRate = 0.7
gamma = 0.9

# initialize frozen lake environment
env = gym.make('FrozenLake-v0')
env.reset()

# initialize Q-Table
qTable = numpy.zeros((env.observation_space.n, env.action_space.n))

def selectAction(state):
    action = 0

    if random.random() <= epsilon:
        # Exploration (a.k.a. explore environment randomly)
        action = env.action_space.sample()
    else:
        # Exploitation (a.k.a. use our previously learned environment info)
        action = numpy.argmax(qTable[state, :])

    return action

def updateQTable(oldState, newState, action, reward):
    # Bellman Equation
    qTable[oldState, action] = qTable[oldState, action] + learningRate * (reward + gamma * numpy.max(qTable[newState, :]) - qTable[oldState, action])

# for _ in range(5):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

#     print(observation)

env.close()
