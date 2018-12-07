import numpy as np

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