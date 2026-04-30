import random

import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.cost = 0
        self.theta = theta
        self.parent = None

    def setx(self, x):
        self.x = x

    def sety(self, y):
        self.y = y

    def setparent(self, parent):
        self.parent = parent


class Dubins:
    def __init__(self, nodeA, nodeB, r):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.r = r

    def LSL(self):
        cs_x = self.nodeA.x - self.r * np.sin(self.nodeA.theta)
        cs_y = self.nodeA.y + self.r * np.cos(self.nodeA.theta)
        cg_x = self.nodeB.x - self.r * np.sin(self.nodeB.theta)
        cg_y = self.nodeB.y + self.r * np.cos(self.nodeB.theta)

        diff_x = cg_x - cs_x
        diff_y = cg_y - cs_y
        dist_centers = np.sqrt(diff_x**2 + diff_y**2)
        phi = np.arctan2(diff_y, diff_x)

        t = (phi - self.nodeA.theta) % (2 * np.pi)
        p = dist_centers
        q = (self.nodeB.theta - phi) % (2 * np.pi)
        return t, p, q

    def RSR(self):
        cs_x = self.nodeA.x + self.r * np.sin(self.nodeA.theta)
        cs_y = self.nodeA.y - self.r * np.cos(self.nodeA.theta)
        cg_x = self.nodeB.x + self.r * np.sin(self.nodeB.theta)
        cg_y = self.nodeB.y - self.r * np.cos(self.nodeB.theta)

        diff_x = cg_x - cs_x
        diff_y = cg_y - cs_y
        dist_centers = np.sqrt(diff_x**2 + diff_y**2)
        phi = np.arctan2(diff_y, diff_x)

        t = (self.nodeA.theta - phi) % (2 * np.pi)
        p = dist_centers
        q = (phi - self.nodeB.theta) % (2 * np.pi)
        return t, p, q

    def LSR(self):
        cs_x = self.nodeA.x - self.r * np.sin(self.nodeA.theta)
        cs_y = self.nodeA.y + self.r * np.cos(self.nodeA.theta)
        cg_x = self.nodeB.x + self.r * np.sin(self.nodeB.theta)
        cg_y = self.nodeB.y - self.r * np.cos(self.nodeB.theta)

        diff_x = cg_x - cs_x
        diff_y = cg_y - cs_y
        dist_centers = np.sqrt(diff_x**2 + diff_y**2)
        phi = np.arctan2(diff_y, diff_x)

        val = dist_centers**2 - (2 * self.r) ** 2
        if val < 0:
            return None
        p = np.sqrt(val)

        phi2 = phi - np.arctan2(2 * self.r, p)
        t = (phi2 - self.nodeA.theta) % (2 * np.pi)
        q = (phi2 - self.nodeB.theta) % (2 * np.pi)
        return t, p, q

    def RSL(self):
        cs_x = self.nodeA.x + self.r * np.sin(self.nodeA.theta)
        cs_y = self.nodeA.y - self.r * np.cos(self.nodeA.theta)
        cg_x = self.nodeB.x - self.r * np.sin(self.nodeB.theta)
        cg_y = self.nodeB.y + self.r * np.cos(self.nodeB.theta)

        diff_x = cg_x - cs_x
        diff_y = cg_y - cs_y
        dist_centers = np.sqrt(diff_x**2 + diff_y**2)
        phi = np.arctan2(diff_y, diff_x)

        val = dist_centers**2 - (2 * self.r) ** 2
        if val < 0:
            return None
        p = np.sqrt(val)

        phi2 = phi + np.arctan2(2 * self.r, p)
        t = (self.nodeA.theta - phi2) % (2 * np.pi)
        q = (self.nodeB.theta - phi2) % (2 * np.pi)
        return t, p, q

    def LRL(self):
        cs_x = self.nodeA.x - self.r * np.sin(self.nodeA.theta)
        cs_y = self.nodeA.y + self.r * np.cos(self.nodeA.theta)
        cg_x = self.nodeB.x - self.r * np.sin(self.nodeB.theta)
        cg_y = self.nodeB.y + self.r * np.cos(self.nodeB.theta)

        diff_x = cg_x - cs_x
        diff_y = cg_y - cs_y
        dist_centers = np.sqrt(diff_x**2 + diff_y**2)
        phi = np.arctan2(diff_y, diff_x)

        val = (4 * self.r**2 - dist_centers**2) / (4 * self.r**2)
        if val < -1 or val > 1:
            return None
        angle = np.arccos(val)

        t = (phi - self.nodeA.theta + angle) % (2 * np.pi)
        p = 2 * angle
        q = (self.nodeB.theta - phi + angle) % (2 * np.pi)
        return t, p, q

    def RLR(self):
        cs_x = self.nodeA.x + self.r * np.sin(self.nodeA.theta)
        cs_y = self.nodeA.y - self.r * np.cos(self.nodeA.theta)
        cg_x = self.nodeB.x + self.r * np.sin(self.nodeB.theta)
        cg_y = self.nodeB.y - self.r * np.cos(self.nodeB.theta)

        diff_x = cg_x - cs_x
        diff_y = cg_y - cs_y
        dist_centers = np.sqrt(diff_x**2 + diff_y**2)
        phi = np.arctan2(diff_y, diff_x)

        val = (4 * self.r**2 - dist_centers**2) / (4 * self.r**2)
        if val < -1 or val > 1:
            return None
        angle = np.arccos(val)

        t = (self.nodeA.theta - phi + angle) % (2 * np.pi)
        p = 2 * angle
        q = (phi - self.nodeB.theta + angle) % (2 * np.pi)
        return t, p, q

    def shortest_path(self):
        results = []
        for fn in [self.LSL, self.RSR, self.LSR, self.RSL, self.LRL, self.RLR]:
            try:
                result = fn()
                if result is None:
                    continue
                t, p, q = result

                if fn.__name__ in ("LRL", "RLR"):
                    length = self.r * (t + p + q)
                else:
                    length = self.r * t + p + self.r * q
                if length >= 0:
                    results.append((length, t, p, q, fn.__name__))
            except Exception:
                continue
        if not results:
            return None
        return min(results, key=lambda x: x[0])


class RRT:
    def __init__(self, startNode, goalNode, gridMap, step_size, max_iter, radius):
        self.startNode = startNode
        self.goalNode = goalNode
        self.gridMap = gridMap
        self.step_size = step_size
        self.max_iter = max_iter
        self.radius = radius
        self.tree = [self.startNode]
        dx = self.goalNode.x - self.startNode.x
        dy = self.goalNode.y - self.goalNode.y
        self.startNode.theta = np.arctan2(dy, dx)

    def setMap(self, rows=100, cols=100):
        self.gridMap = np.zeros((rows, cols), dtype=np.int8)

    def chooseParent(self, newNode):
        cheapestParent = newNode.parent
        mincost = newNode.cost
        for node in self.tree:
            dist = self.euclid_distance(node, newNode)
            if dist >= self.step_size * 3:
                continue
            cost = node.cost + dist
            if self.collisionCheck(node, newNode):
                continue
            if cost < mincost or mincost == -1:
                cheapestParent = node
                mincost = cost
        newNode.parent = cheapestParent
        newNode.cost = mincost

    def rewire(self, newNode):
        for i in range(0, len(self.tree)):
            dist = self.euclid_distance(self.tree[i], newNode)
            if dist >= self.step_size * 3:
                continue
            if (self.tree[i].cost > (newNode.cost + dist)) and not self.collisionCheck(
                newNode, self.tree[i]
            ):
                self.tree[i].parent = newNode
                self.tree[i].cost = newNode.cost + dist

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
        if dist < self.step_size + 1:
            return True
        else:
            return False

    def euclid_point(self, nodeA, randomPoint):
        return np.hypot(nodeA.x - randomPoint[0], nodeA.y - randomPoint[1])

    def dubin_extension(self, nodeA, randomPoint):
        theta = np.arctan2(randomPoint[1] - nodeA.y, randomPoint[0] - nodeA.x)
        randomNode = Node(randomPoint[0], randomPoint[1], theta)
        dubin_path = Dubins(nodeA, randomNode, self.radius)
        length, t, p, q, path_type = dubin_path.shortest_path()
        nodeC = self.sample_dubins(length, t, p, q, path_type, nodeA, randomNode)
        nodeC.parent = nodeA
        nodeC.cost = nodeA.cost + length
        return nodeC

    def sample_dubins(self, length, t, p, q, path_type, nodeA, randomNode):
        dist = self.step_size
        seg1, seg2, seg3 = 0
        seg1 = self.radius * t
        seg2 = self.radius * p
        x0 = nodeA.x
        y0 = nodeA.y
        theta0 = nodeA.theta
        if path_type[1] == "S":
            seg2 = p
        seg3 = self.radius * q

        seg1 = min(dist, seg1)
        arc_angle = seg1 / self.radius
        x1, y1, theta1 = self.turn(x0, y0, theta0, arc_angle, path_type[0], self.radius)
        dist = dist - seg1
        if not dist == 0:
            if path_type[1] == "S":
                seg2 = min(seg2, dist)
                dist = dist - seg2
                x2 = x1 + seg2 * np.cos(theta1)
                y2 = y1 + seg2 * np.sin(theta1)
                theta2 = theta1
            else:
                seg2 = min(seg2, dist)
                arc_angle = seg2 / self.radius
                dist = dist - seg2
                x2, y2, theta2 = self.turn(
                    x1, y1, theta1, arc_angle, path_type[1], self.radius
                )

            if not dist == 0:
                seg3 = min(seg3, dist)
                arc_angle = seg3 / self.radius
                dist = dist - seg3
                x3, y3, theta3 = self.turn(
                    x2, y2, theta2, arc_angle, path_type[2], self.radius
                )
                nodeC = Node(x3, y3, theta3)
            else:
                nodeC = Node(x2, y2, theta2)
        else:
            nodeC = Node(x1, y1, theta1)

        return nodeC

    def turn(self, x, y, theta, angle, direction, r):
        cx = 0
        cy = 0
        theta_new = 0
        if direction == "L":
            cx = x - r * np.sin(theta)
            cy = y + r * np.cos(theta)
            theta_new = theta + angle
        if direction == "R":
            cx = x + r * np.sin(theta)
            cy = y - r * np.cos(theta)
            theta_new = theta - angle

        x_new = cx + r * np.sin(theta_new)
        y_new = cy - r * np.cos(theta_new)

        return x_new, y_new, theta_new

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
        nodeC.cost = nodeA.cost + self.euclid_distance(nodeA, nodeC)
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
                self.chooseParent(nextNode)
                self.tree.append(nextNode)
                self.rewire(nextNode)
                if self.goalCheck(nextNode):
                    return self.tracePath(nextNode)
        return None


if __name__ == "__main__":
    grid = np.zeros((100, 100), dtype=np.int8)

    grid[20:45, 40] = 1
    grid[55:80, 40] = 1
    grid[30, 50:75] = 1

    goal = Node(63, 31)
    start = Node(10, 10)
    rrt = RRT(start, goal, grid, 3, 8000, 4)

    path = rrt.plan()
    if path:
        print(f"Path found with {len(path)} nodes:")
        for node in path:
            print(f"  ({node.x:.1f}, {node.y:.1f})")
    else:
        print("No path found.")
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r")
    if path:
        print(f"Total path cost is {path[-1].cost}")
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
