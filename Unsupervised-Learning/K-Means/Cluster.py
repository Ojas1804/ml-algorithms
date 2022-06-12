from statistics import mean
from Point import Point


class Cluster(object):
    def __init__(self, x, y):
        self.center = Point(x, y)
        self.points = []

    def update(self):
        if(len(self.points)==0):
            pass            
        else:
            center_x = mean([point.x for point in self.points])
            center_y = mean([point.y for point in self.points])
            self.center = Point(center_x, center_y)
            self.points = []
            return self.center

    def add_point(self, point):
        self.points.append(point)

    def __repr__(self):
        return str(self.center)

    def compute_result(points):
        points = [Point(*point) for point in points]
        a = Cluster(1,0)
        b = Cluster(-1,0)
        #this represents  the list of points that have beenn assigned previusly to the cluster
        a_old = []
        b_old = []
        for _ in range(10000): # max iterations
            for point in points:
            #are we closest to the a.centroid vs the b.centroid
                if point.distance(a.center) < point.distance(b.center):
                    a.add_point(point)
                else:
                    b.add_point(point)
            if a_old == a.points and b_old == b.points:
                break
            a_old = a.points
            b_old = b.points
            a.update()
            b.update()
        if (a.center.x>b.center.x):
            return [(a.center.x,a.center.y), (b.center.x, b.center.y)]
        else:
            return [(b.center.x,b.center.y), (a.center.x, a.center.y)]