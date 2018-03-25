from classes import *
import numpy as np
import matplotlib.pyplot as plt
import winsound

def drawMap(car, world):
    x, y = car.curState.crd
    n, m = world.shape
    print(x, y, n, m)
    for i in range(n):
        for j in range(m):
            if i == x and j == y:
                print('*  ')
            else:
                print(world[n, m], ' ')
        print('\n')

###### parameters #######
learning_rate = 0.8
discount_factor = 1

epsilon_training = 0.8
epsilon_experiment = 1

numEpisodes = 5000
numExperiments = 10
#########################

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
envTrain.setDest(np.array([5, 4]))

envExp = env1
envExp.setDest(np.array([3, 3]))

initD = np.array([0, 1])
startPt = np.array([1, 1])
dists, destPos = envTrain.calDists(startPt, initD)
initS = State(startPt, 0, dists = dists, destPos = destPos)
cnt = 0
rList = []
for _ in range(numExperiments):
    
    car = Agent(initS, initD)

    print("Start training...")
    for i in range(1, numEpisodes + 1):
        car.curState = initS
        car.direction = initD
        car.rTotal = 0
        car.terminate = False
        k = 0
        while not car.terminate and k < 100:
            action = car.takeAction(epsilon_training)
            s, r, t = car.interact(action, envTrain, lr = learning_rate, y = discount_factor)
            k += 1
            
##        rList.append(car.rTotal)
        if i % 100 == 0:
            print("Iter", i, car.rTotal)
    print("Training finished.")

    ##plt.plot(np.arange(numEpisodes), rList)
    ##plt.show()

    newCar = Agent(initS, initD)
    newCar.QTable = car.QTable
    k = 0
    while not newCar.terminate and k < 100:
        k += 1
        action = newCar.takeAction(epsilon_experiment)
        s, _, _ = newCar.interact(action, envExp, update = False)
        print(action, s.crd, s.v, s.destPos, newCar.direction)
        
    if np.all(s.crd == envExp.dest):
        print("Arrive!")
        cnt += 1
    print(newCar.rTotal)
    rList.append(newCar.rTotal)
print("Agent arrives the destination %d times in %d experiments."%(cnt, numExperiments))
print(rList)

##drawMap(newCar, world)

winsound.Beep(1000, 200)
