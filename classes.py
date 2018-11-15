import numpy as np

class Environment(object):
    
    def __init__(self, world = [], visibility = [1, 1, 1], speedLimit = 1):
        self.world = world # 1 for car, 2 for people, 3 for destiny
        self.rewardTable = [0, -100, -800, 10000]
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
        reward = -2
        
        newV, newCrd = self.stateTransition(curState, action, direction)
        
        reward += self.rewardTable[self.world[newCrd[0], newCrd[1]]]
        newDists, dest = self.calDists(newCrd, direction)

        # check termination
        if self.world[newCrd[0], newCrd[1]] > 0:
            terminate = True

        newState = State(newCrd, newV, newDists, dest)
        return newState, reward, terminate

    def calDists(self, crd, direction):

        temp = np.sign(self.dest - crd)
        dest = np.dot(temp, np.array([[direction[0], direction[1]], [direction[1], -direction[0]]]))
        
        right = np.dot(direction, np.array([[0, -1], [ 1, 0]]))
        left  = np.dot(direction, np.array([[0,  1], [-1, 0]]))
        if self.world[crd[0], crd[1]] > 0:
            return np.array([0, 0, 0]), dest
        
        d0, d1, d2 = 0, 0, 0
        
        crdF = crd + (d0 + 1) * direction
        while d0 + 1 <= self.visibility[0] and self.inMap(crdF) and self.world[crdF[0], crdF[1]]==0:#in (0, 3)
            d0 += 1
            crdF = crd + (d0 + 1) * direction
        
        crdR = crd + (d1 + 1) * right
        while d1 + 1 <= self.visibility[1] and self.inMap(crdR) and self.world[crdR[0], crdR[1]]==0:#in (0, 3)
            d1 += 1
            crdR = crd + (d1 + 1) * direction

        crdL = crd + (d2 + 1) * left
        while d2 + 1 <= self.visibility[2] and self.inMap(crdL) and self.world[crdL[0], crdL[1]]==0:#in (0, 3)
            d2 += 1
            crdL = crd + (d2 + 1) * direction

        return np.array([d0, d1, d2]), dest

    def inMap(self, crd):
        return np.all(crd >= 0) and np.all(crd < self.world.shape)

    def setDest(self, destCrd):
        self.world[destCrd[0], destCrd[1]] = 3
        self.dest = destCrd
        
class Agent(object):
    
    def __init__(self, curState, direction, QTable = None):
        self.actions = np.array([[ 0, 0], [ 1, 0], [-1, 0]])
        # self.actions = np.array([[ 0, 0], [ 0, 1], [ 0, -1],
        #                          [ 1, 0], [ 1, 1], [ 1, -1],
        #                          [-1, 0], [-1, 1], [-1, -1]]) # [steering wheel, gas pedal]
        
        # self.actions = np.array([[ 0, 1], [ 1, 1], [-1, 1],
        #                         [ 0, 0], [ 1, 0], [-1, 0],
        #                         [ 0,-1], [ 1,-1], [-1,-1]]) # [steering wheel, gas pedal]

        # self.actions = np.array([[ 0, 1], [ 1, 1], [-1, 1],
        #                         [ 0, 0], [ 1, 0], [-1, 0]]) # [steering wheel, gas pedal]
        self.curState = curState
        self.direction = direction # np.array([+-1, 0]) or np.array([0, +-1]), invisible to the agent
        self.QTable = {curState:np.zeros(len(self.actions))}
        self.rTotal = 0
        self.terminate = False

    # epsilon-greedy
    def takeAction(self, epsilon = 0.8):
##        if self.curState.v > 0:
##            if np.random.rand() <= epsilon:
##                action = self.actions[np.argmax(self.QTable[self.curState])] # exploitation
##            else:
##                action = self.actions[np.random.randint(len(self.actions))] # exploration
##        else:
##            if np.random.rand() <= epsilon:
##                action = self.actions[np.argmax(self.QTable[self.curState][0:3])] # exploitation
##            else:
##                action = self.actions[np.random.randint(3)] # exploration

        if np.random.rand(1) <= epsilon:
            action = self.actions[np.argmax(self.QTable[self.curState])] # exploitation
        else:
            action = self.actions[np.random.randint(len(self.actions))] # exploration

        # if steering wheel is turned, change direction
        if action[0] != 0:
            self.direction = np.dot(self.direction, np.array([[0, -1], [1, 0]]) * action[0])
        
        return action

    def interact(self, action, env, update = True, lr = 0.8, y = 0.8):
        s, r, t = env.step(self.curState, action, self.direction)
        self.rTotal += r
        self.terminate = t
        if s not in self.QTable:
            self.QTable[s] = np.zeros(len(self.actions))
        if update:
            self.updateQTable(s, action, r, lr, y)
        self.curState = s
        return s, r, t
    
    def updateQTable(self, s, a, r, lr, y):
        actionHelper = self.actions.tolist()
        actionIdx = actionHelper.index(list(a))
        self.QTable[self.curState][actionIdx] = (1 - lr) * self.QTable[self.curState][actionIdx] + lr * (r + y * max(self.QTable[s]) - self.QTable[self.curState][actionIdx])

class State(object):
    
    def __init__(self, crd, v, dists, destPos):
        self.crd = crd #np.array([x, y])
        self.v = v
        self.dists = dists #np.array, front visibility:2, left&right visibility:1
        self.destPos = destPos # relative position of destination

    # make objects comparable by speed and perception of its surroundings
    # the agent doesn't know its coordinates
    def __eq__(self, other):
        # return np.all(self.dists == other.dists) and self.v == other.v and np.all(self.destPos == other.destPos)
        return np.all(self.dists == other.dists) #and np.all(self.destPos == other.destPos)

    def __hash__(self):
        # return hash(tuple(self.dists) + (self.v, ) + tuple(self.destPos))
        return hash(tuple(self.dists) + tuple(self.destPos))
