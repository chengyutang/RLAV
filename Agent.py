"""
Authur: Chengyu Tang

"""

import numpy as np

class Agent(object):
	
	def __init__(self, curState, direction, QTable = None):
		"""
		actions: numpy array of action
			action[0] stands for steering wheel. 0 for going forward, 1 for turning right and -1 for turning left.
			action[1] stands for gas pedal. Value stands for the change of speed.
		curState: State
		direction: [1, 0] for down, [-1, 0] for up, [0, 1] for right, [0, -1] for left.
		QTable: {State: [float]}
		rTotal: accumulative reward
		terminate: boolean
		"""
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
		self.direction = direction
		self.QTable = {curState:np.zeros(len(self.actions))}
		self.rTotal = 0
		self.terminate = False

	# epsilon-greedy
	def takeAction(self, epsilon = 0.8):
		if self.curState.v > 0:
			if np.random.rand() <= epsilon:
				action = self.actions[np.argmax(self.QTable[self.curState])] # exploitation
			else:
				action = self.actions[np.random.randint(len(self.actions))] # exploration
		else:
			if np.random.rand() <= epsilon:
				action = self.actions[np.argmax(self.QTable[self.curState][0:3])] # exploitation
			else:
				action = self.actions[np.random.randint(3)] # exploration

		# human knowledge:
		# if np.all(self.curState.dists == [1, 0, 0]):
		#     action = np.array([0, 0])
		# elif np.all(self.curState.dists == [0, 1, 0]):
		#     action = np.array([1, 0])
		# elif np.all(self.curState.dists == [0, 0, 1]):
		#     action = np.array([-1, 0])

		# else:
		# exploit with possibility of epsilon
		if np.random.rand(1) <= epsilon:
			choices = np.argwhere(self.QTable[self.curState] == np.amax(self.QTable[self.curState]))
			choices = choices.reshape(len(choices))
			idx = np.random.choice(choices)
			action = self.actions[idx]
		
		else: # explore with possibility of 1 - epsilon
			action = self.actions[np.random.randint(self.actions.shape[0])]

		# if steering wheel is turned, change direction
		if action[0] != 0:
			self.direction = np.dot(self.direction, np.array([[0, -1], [1, 0]]) * action[0])
		
		return action

	def interact(self, action, env, update = True, lr = 0.001, y = 0.8):
		newState, reward, terminated = env.step(self.curState, action, self.direction)
		self.rTotal += reward
		self.terminate = terminated
		if newState not in self.QTable:
			self.QTable[newState] = np.zeros(len(self.actions))
		if update:
			self.updateQTable(newState, action, reward, lr, y)
		self.curState = newState
		return newState, reward, terminated
	
	def updateQTable(self, s, a, r, lr, y):
		"""
		s: resultant state of taking action a at state self.curState
		a: action
		r: reward of taking action a
		lr: learning rate
		y: discount factor
		"""
		actionHelper = self.actions.tolist()
		actionIdx = actionHelper.index(list(a))
		self.QTable[self.curState][actionIdx] = (1 - lr) * self.QTable[self.curState][actionIdx] + lr * (r + y * max(self.QTable[s]) - self.QTable[self.curState][actionIdx])