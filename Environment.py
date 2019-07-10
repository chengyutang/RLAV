from State import *
import numpy as np

class Environment(object):
    
    def __init__(self, world = [], visibility = [1, 1, 1], speedLimit = 1):
        self.world = world # 1 for car, 2 for people, 3 for destiny
        self.rewardTable = [-5, -500, -800, 1000]
        self.visibility = visibility
        self.speedLimit = speedLimit
        
    def stateTransition(self, curState, action, direction):
        # action: [(-1, 0, 1), (-1, 0, 1)], [steering wheel, gas pedal]
        
        newV = curState.v + action[1]
        newV = min(newV, self.speedLimit)
        newV = max(newV, 0)
        
        newX, newY = curState.crd + newV * direction
        newX = max(newX, 0)
        newY = max(newY, 0)
        newX = min(newX, self.world.shape[0] - 1)
        newY = min(newY, self.world.shape[1] - 1)
        
        newCrd = np.array([int(newX), int(newY)])
        
        return newV, newCrd

    def step(self, curState, action, direction):
        terminate = False
        reward = 0 # penalty for each step
        
        newV, newCrd = self.stateTransition(curState, action, direction)
        
        reward += self.rewardTable[self.world[newCrd[0], newCrd[1]]]
        newDists, dest = self.calSurroundDists(newCrd, direction)

        # check termination
        if self.world[newCrd[0], newCrd[1]] > 0:
            terminate = True

        newState = State(newCrd, newV, newDists, dest)
        return newState, reward, terminate

    def calSurroundDists(self, crd, direction):

        temp = np.sign(self.dest - crd)
        dest = np.dot(temp, np.array([[direction[0], direction[1]], [direction[1], -direction[0]]]))
        
        # calculate the absolute direction of right and left
        right = np.dot(direction, np.array([[0, -1], [ 1, 0]]))
        left  = np.dot(direction, np.array([[0,  1], [-1, 0]]))

        if self.world[crd[0], crd[1]] > 0:
            return np.array([0, 0, 0]), dest
        
        distFront, distRight, distLeft = 0, 0, 0
        
        # determine the distance to the closest obstacle on each direction
        crdF = crd + (distFront + 1) * direction
        while distFront + 1 <= self.visibility[0] and self.inMap(crdF) and self.world[crdF[0], crdF[1]]==0:#in (0, 3)
            distFront += 1
            crdF = crd + (distFront + 1) * direction
        
        crdR = crd + (distRight + 1) * right
        while distRight + 1 <= self.visibility[1] and self.inMap(crdR) and self.world[crdR[0], crdR[1]]==0:#in (0, 3)
            distRight += 1
            crdR = crd + (distRight + 1) * direction

        crdL = crd + (distLeft + 1) * left
        while distLeft + 1 <= self.visibility[2] and self.inMap(crdL) and self.world[crdL[0], crdL[1]]==0:#in (0, 3)
            distLeft += 1
            crdL = crd + (distLeft + 1) * direction

        return np.array([distFront, distRight, distLeft]), dest

    def inMap(self, crd):
        return np.all(crd >= 0) and np.all(crd < self.world.shape)

    def setDest(self, destCrd):
        self.world[destCrd[0], destCrd[1]] = 3
        self.dest = destCrd