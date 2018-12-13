from Agent import *
from Environment import *
from State import *
import numpy as np
import matplotlib.pyplot as plt

###### parameters #######
learning_rate = 0.2
discount_factor = 0.8

# epsilon: greedy level
epsilon_training = 0.5
epsilon_experiment = 1

numEpisodes = 5000
numExperiments = 1
maxIter = 100
#########################

def drawMap(car, world):
    x, y = car.curState.crd
    n, m = world.shape
    print(x, y, n, m)
    for i in range(n):
        for j in range(m):
            if i == x and j == y:
                print('*  ', end = '')
            else:
                print(world[i, j], ' ', end = '')
        print('\n')

world1 = np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 1],
                   [1, 2, 0, 2, 1],
                   [1, 1, 0, 0, 1],
                   [1, 1, 1, 1, 1]])
env1 = Environment(world1)

world2 = np.array([[1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 0, 1],
                   [1, 2, 0, 2, 1, 1],
                   [1, 1, 0, 1, 0, 1],
                   [1, 1, 0, 0, 0, 1],
                   [1, 1, 1, 2, 0, 1],
                   [1, 1, 1, 1, 1, 1]])
env2 = Environment(world2)

envTrain = env2
envTrain.setDest(np.array([3, 4]))

envExp = env1
envExp.setDest(np.array([3, 3]))

initD = np.array([0, 1])
startPt = np.array([1, 1])
dists, destPos = envTrain.calDists(startPt, initD)
initS = State(startPt, 1, dists = dists, destPos = destPos)
numArrives = 0
rList = []

car = Agent(initS, initD)

# Start of numExperiments times of Experiments
for _ in range(numExperiments):

    car = Agent(initS, initD)

    # Train
    print("Start training...")
    for i in range(1, numEpisodes + 1):
        car.curState = initS
        car.direction = initD
        car.rTotal = 0
        car.terminate = False
        k = 0
        while not car.terminate and k < maxIter:
            action = car.takeAction(epsilon_training)
            s, r, t = car.interact(action, envTrain, lr = learning_rate, y = discount_factor)
            k += 1

        # rList.append(car.rTotal)
        if i % 100 == 0:
            print("Iter", i, car.rTotal)
    print("Training finished.\n")

    # Drive
    newCar = Agent(initS, initD)
    newCar.QTable = car.QTable
    for state in newCar.QTable:
        print(state.crd, state.dists, ':', newCar.QTable[state])
    k = 0
    while not newCar.terminate and k < maxIter:
        k += 1
        action = newCar.takeAction(epsilon_experiment)
        s, _, _ = newCar.interact(action, envExp, update = False)
        print(s.crd, action, s.v, s.dists, s.destPos, newCar.direction)

    if np.all(s.crd == envExp.dest):
        print("Arrive!")
        numArrives += 1
    print(newCar.rTotal)
    rList.append(newCar.rTotal)
    drawMap(newCar, world1)

# for state in newCar.QTable:
#     print(state.crd, ':', newCar.QTable[state])
print("Agent arrives the destination %d times in %d experiments."%(numArrives, numExperiments))
print(rList)
