import numpy as np

class State(object):
    
    def __init__(self, crd, v, dists, destPos):
        """
        crd: np.array([x, y])
        v: integer
        dists: np.array([front, right, left]) distance to the closest object on the front, right and left
        destPos: np.array([front/befind, left/right]) relative position of destination. Values are 0, 1 or -1.
        """
        self.crd = crd
        self.v = v
        self.dists = dists
        self.destPos = destPos

    # make objects comparable by speed and perception of its surroundings
    # the agent doesn't know its coordinates
    def __eq__(self, other):
        # return np.all(self.dists == other.dists) and self.v == other.v and np.all(self.destPos == other.destPos)
        if isinstance(other, self.__class__):
            return np.all(self.dists == other.dists) and np.all(self.destPos == other.destPos)
        return False

    def __hash__(self):
        # return hash(tuple(self.dists) + (self.v, ) + tuple(self.destPos))
        return hash(tuple(self.dists) + tuple(self.destPos))
        # return hash(tuple(self.dists))