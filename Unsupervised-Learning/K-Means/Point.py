from math import sqrt

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
            
    def __repr__(self):
        return "Point(%d, 0)"

    def distance(self, other):
        X = self.x - other.x
        Y = self.y - other.y
        
        X = X*X
        Y = Y*Y
        
        return(sqrt(X + Y))