from Agent import *
from Environment import *
from State import *
from Parameters import *
import numpy as np
import matplotlib.pyplot as plt

def drawMap(car, world):
	x, y = car.curState.crd
	n, m = world.shape
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
env1.setDestination(np.array([3, 3]))

world2 = np.array([[1, 1, 1, 1, 1, 1],
				   [1, 0, 0, 1, 0, 1],
				   [1, 2, 0, 2, 1, 1],
				   [1, 1, 0, 1, 0, 1],
				   [1, 0, 0, 0, 0, 1],
				   [1, 0, 0, 2, 0, 1],
				   [1, 1, 1, 1, 1, 1]])
env2 = Environment(world2)
env2.setDestination(np.array([5, 1]))

world3 = np.array([[1, 1, 1, 1, 1, 1],
				   [1, 0, 0, 1, 0, 1],
				   [1, 2, 0, 2, 1, 1],
				   [1, 0, 0, 0, 0, 1],
				   [1, 0, 0, 1, 0, 1],
				   [1, 0, 1, 0, 0, 1],
				   [1, 0, 0, 0, 0, 1],
				   [1, 1, 1, 1, 1, 1]])
env3 = Environment(world3)
env3.setDestination(np.array([4, 3]))

envTrain = env3
envExp = env2

initD = np.array([0, 1])
startPt = np.array([1, 1])
dists, destPos = envTrain.calSurroundDists(startPt, initD)
initS = State(startPt, 1, dists = dists, destPos = destPos)
numArrives = 0
rList = []

car = Agent(initS, initD)

# Start of numExperiments times of Experiments
for _ in range(numExperiments):

	car = Agent(initS, initD)
	numSuccess = 0

	# Training
	dynamic_epsilon = 0
	print("Start training...")
	for i in range(1, numEpisodes + 1):

		# change destination randomly with probability of 10%
		if np.random.rand(1) < 0.2:
			curDestination = envTrain.dest
			envTrain.world[curDestination[0]][curDestination[1]] = 0
			x = np.random.randint(1, 6)
			y = np.random.randint(1, 5)
			while envTrain.world[x][y] > 0:
				x = np.random.randint(1, 7)
				y = np.random.randint(1, 5)
			envTrain.setDestination([x, y])

		car.curState = initS
		car.direction = initD
		car.rTotal = 0
		car.terminate = False
		k = 0
		while not car.terminate and k < maxIter:
			# action = car.takeAction(epsilon_training)
			action = car.takeAction(dynamic_epsilon)
			s, r, t = car.interact(action, envTrain, lr = learning_rate, y = discount_factor)
			k += 1
		numSuccess += np.all(s.crd == envTrain.dest)

		dynamic_epsilon += 1 / numEpisodes

		if i % (numEpisodes // 10) == 0:
			print("Iter", i, car.rTotal, numSuccess)
	print("Training finished.\n")

	# Testing
	newCar = Agent(initS, initD)
	newCar.QTable = car.QTable
	print("Learned Q-Table:")
	for state in newCar.QTable:
		print(state.dists, state.destPos, ':', newCar.QTable[state])
	k = 0

	print("\nTrajectories:")
	while not newCar.terminate and k < maxIter:
		k += 1
		action = newCar.takeAction(epsilon_experiment)
		s, _, _ = newCar.interact(action, envExp, update = False)
		print('crd', s.crd, 'act', action, 'v', s.v, 'dist', s.dists, 'destPos', s.destPos, 'dir', newCar.direction, newCar.QTable[s])

	if np.all(s.crd == envExp.dest):
		print("Arrive!")
		numArrives += 1
	print("\nFinal reward:", newCar.rTotal)
	rList.append(newCar.rTotal)
	drawMap(newCar, envExp.world)

print("Agent arrives the destination %d times in %d experiments."%(numArrives, numExperiments))
print(rList)