import math
import numpy as np
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
    def setx(self, x):
        self.x = x
    def sety(self, y):
        self.y = y
    def setparent(self, parent):
        self.parent = parent

class RRT:
    def __init__(self, startNode, goalNode, gridMap, step_size, max_iter):
        self.startNode = startNode
        self.goalNode = goalNode
        self.gridMap = gridMap
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.startNode]
    
    def setMap(self):
        self.map = [[0 for _ in range(100)] for _ in range(100)]
    def euclid_distance(self, curNode, nextNode):
        dist = math.sqrt(math.pow(curNode.x - nextNode.x, 2) + math.pow(curNode.y - nextNode.y, 2))
        return dist

    def nearestNode(self, randomNode):
        closestNode = None
        closestdist = -1

        for node in self.tree:
            dist = self.euclid_distance(randomNode, node) 
            if dist < closestdist || closestdist == -1:
                closestNode = node
                closestdist = dist

        return closestNode


    def randomSampler(self):
        rand_x = random.randint(0, self.gridMap.shape[0])
        rand_y = random.randint(0, self.gridMap.shape[1])
        return [rand_x, rand_y]


    def collisionCheck(self, node, randomNode):
        tolerance = 0
        dx = randomNode.x - node.x 
        dy = randomNode.y - node.y
        stepx = node.x
        stepy = node.y

        if dx == 0 && dy == 0:
            return False
        if dx == 0:
            for i in range(0, abs(dy)):
                if (self.gridMap[round(node.y + i * stepy)][round(node.x)] == 1):
                    return True
            return False
        if dy == 0:
            for i in range(0, abs(dx)):
                if (self.gridMap[round(node.y)][round(node.x + i * stepx)] == 1):
                    return True
            return False

     
        error = 0
        residual = 0
        if abs(dy) > abs(dx):
            residual = abs(dx) / abs(dy)
        else:
            residual = abs(dy) / abs(dx)
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        while stepx != randomNode.x && stepy != randomNode.y:
            if error > 0.5:
                error -= 1
                if abs(dx) > abs(dy):
                    stepx += sx
                else:
                    stepy += sy
            if abs(dy) > abs(dx):
                stepy += sy
            else:
                stepx += sx
            if (self.gridMap[round(stepy)][round(stepx)] == 1):
                return True
            error += residual

        return False

    def goalCheck(self, newNode): 
        dist = self.euclid_distance(newNode, self.goalNode)
        if dist < self.step_size:
            return True
        else:
            return False
    def euclid_point(self, nodeA, randomPoint):
        return math.sqrt(math.pow(nodeA.x - randomPoint[0], 2) + math.pow(nodeA.y - randomPoint[1],2))

    def extend(self, nodeA, randomPoint):
        dist = self.euclid_point(nodeA, randomPoint)
        dx = randomPoint[0] - nodeA.x
        dy = randomPoint[1] - nodeA.y
        normaldx = dx / dist
        normaldy = dy / dist
        
        nodeC = Node(nodeA.x + (normaldx * self.step_size), nodeA.y + (normaldy * self.step_size))
        nodeC.parent = nodeA
        return nodeC


