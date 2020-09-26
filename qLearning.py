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
maxEpsilon = 1
minEpsilon = 0.01
learningRate = 0.1 # was 0.7
gamma = 0.99
decayRate = 0.001
MAX_EPISODES = 10000 # TODO: change later for more learning
MAX_STEPS_PER_EPISODE = 100

# initialize frozen lake environment
env = gym.make('FrozenLake-v0')

# initialize Q-Table
qTable = numpy.zeros((env.observation_space.n, env.action_space.n))

def selectAction(state):
    action = 0

    if random.uniform(0, 1) <= epsilon:
        # Exploration (a.k.a. explore environment randomly)
        action = env.action_space.sample()
    else:
        # Exploitation (a.k.a. use our previously learned environment info)
        action = numpy.argmax(qTable[state, :])

    return action

def updateQTable(oldState, newState, action, reward):
    # Bellman Equation
    qTable[oldState, action] = qTable[oldState, action] + learningRate * (reward + gamma * numpy.max(qTable[newState, :]) - qTable[oldState, action])


# Main Method
rewardsAllEpisodes = []

for ep in range(MAX_EPISODES):
    state1 = env.reset()
    # env.render()

    done = False
    rewardsCurrentEp = 0
    i = 0

    while not done and i < MAX_STEPS_PER_EPISODE:
        action = selectAction(state1)
        state2, reward, done, info = env.step(action)

        # give negative reward for falling into hole
        # if done and reward == 0:
        #     reward -= 1

        updateQTable(state1, state2, action, reward)
        state1 = state2
        rewardsCurrentEp += reward

        i += 1
        # env.render()
    
    # decay epsilon before starting next learning episode
    epsilon = minEpsilon + (maxEpsilon - minEpsilon) * numpy.exp(-decayRate * ep)

    rewardsAllEpisodes.append(rewardsCurrentEp)

# Calculate the average reward per 1000 episodes
rewardsPer1000Eps = numpy.split(numpy.array(rewardsAllEpisodes), MAX_EPISODES/1000)
count = 1000
print("Average reward per 1000 episodes\n===============================")
for r in rewardsPer1000Eps:
    print("%s: %s" % (count, str(sum(r/1000))))
    count += 1000

# Print final Q-Table Values
print ("\n\nFinal Q-Table Values")
print ("    Left        Down        Right       Up")
print(qTable)

env.close()
