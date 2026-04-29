import random

import matplotlib.pyplot as plt
import numpy as np


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

    def setMap(self, rows=100, cols=100):
        self.gridMap = np.zeros((rows, cols), dtype=np.int8)

    def euclid_distance(self, curNode, nextNode):
        return np.hypot(curNode.x - nextNode.x, curNode.y - nextNode.y)

    def nearestNode(self, randomNode):
        closestNode = None
        closestdist = -1

        for node in self.tree:
            dist = self.euclid_point(node, randomNode)
            if dist < closestdist or closestdist == -1:
                closestNode = node
                closestdist = dist

        return closestNode

    def randomSampler(self):
        rand_x = random.randint(0, self.gridMap.shape[0] - 1)
        rand_y = random.randint(0, self.gridMap.shape[1] - 1)
        return [rand_x, rand_y]

    def collisionCheck(self, node, randomNode):
        dx = randomNode.x - node.x
        dy = randomNode.y - node.y
        dist = np.hypot(dx, dy)
        if dist == 0:
            return False
        steps = int(np.ceil(dist))
        for s in range(steps + 1):
            t = s / steps
            cx = int(round(node.x + t * dx))
            cy = int(round(node.y + t * dy))
            if (
                cx < 0
                or cy < 0
                or cx >= self.gridMap.shape[1]
                or cy >= self.gridMap.shape[0]
            ):
                return True
            if self.gridMap[cy, cx] == 1:
                return True
        return False

    def goalCheck(self, newNode):
        dist = self.euclid_distance(newNode, self.goalNode)
        if dist < self.step_size:
            return True
        else:
            return False

    def euclid_point(self, nodeA, randomPoint):
        return np.hypot(nodeA.x - randomPoint[0], nodeA.y - randomPoint[1])

    def extend(self, nodeA, randomPoint):

        diff = np.array(
            [randomPoint[0] - nodeA.x, randomPoint[1] - nodeA.y], dtype=float
        )
        dist = np.linalg.norm(diff)
        if dist == 0:
            return nodeA
        direction = diff / dist
        new_pos = np.array([nodeA.x, nodeA.y]) + direction * self.step_size
        nodeC = Node(new_pos[0], new_pos[1])
        nodeC.parent = nodeA
        return nodeC

    def tracePath(self, node):
        path = []
        curnode = node
        while curnode != None:
            path.insert(0, curnode)
            curnode = curnode.parent
        return path

    def plan(self):
        for _ in range(0, self.max_iter):
            randompoint = self.randomSampler()
            nearNode = self.nearestNode(randompoint)
            nextNode = self.extend(nearNode, randompoint)
            if not self.collisionCheck(nearNode, nextNode):
                self.tree.append(nextNode)
                if self.goalCheck(nextNode):
                    return self.tracePath(nextNode)
        return None


if __name__ == "__main__":
    grid = np.zeros((100, 100), dtype=np.int8)

    # vertical wall near the middle with a gap
    grid[20:45, 40] = 1
    grid[55:80, 40] = 1
    # horizontal wall blocking upper path
    grid[30, 50:75] = 1

    goal = Node(63, 31)
    start = Node(10, 10)
    rrt = RRT(start, goal, grid, 1, 100000)

    path = rrt.plan()
    if path:
        print(f"Path found with {len(path)} nodes:")
        for node in path:
            print(f"  ({node.x:.1f}, {node.y:.1f})")
    else:
        print("No path found.")
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r")

    # draw tree
    for node in rrt.tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "b-", alpha=0.3)

    # draw path
    if path:
        px = [n.x for n in path]
        py = [n.y for n in path]
        plt.plot(px, py, "r-", linewidth=2)

    plt.plot(start.x, start.y, "go", markersize=10)
    plt.plot(goal.x, goal.y, "ro", markersize=10)
    plt.show()
