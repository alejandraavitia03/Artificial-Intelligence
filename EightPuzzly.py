import copy
from gc import set_debug
import heapq as hp
import numpy as np

# Final state map: grid_element -> <x_position, y_position>
final_state = {
    1: [0, 0],
    2: [0, 1],
    3: [0, 2],
    4: [1, 0],
    5: [1, 1],
    6: [1, 2],
    7: [2, 0],
    8: [2, 1],
    0: [2, 2]
}


# My State Class


class State:
    # Graph and cost
    def __init__(self, graph_snapshot, cost, depth):
        self.graph_snapshot = np.array(graph_snapshot)
        self.cost = cost
        self.depth = depth

    # Turns grid into a one liner
    def flatten(self):
        return list(self.graph_snapshot.flatten())

    # getGraph
    def graph(self):
        return self.graph_snapshot

    # getDepth
    def getDepth(self):
        return self.depth

    # make sure we are getting ints
    def __lt__(self, other):
        if not isinstance(other, State):
            return TypeError("Please compare against an int")
        return self.cost + self.depth <= other.cost + other.depth


# To check that it is in bound
def InBound(x, y):
    if x < 0 or y < 0: return False
    if x > 2 or y > 2: return False
    return True


# This will return the Euclidian distance of a single spot
# Initial -> [x][y]
# final -> coordinates of the value we want in the correct position
def GetEuclidianDistance(initial, final) -> int:
    # We gunna make a stack with the given coordinates and add to the seen list
    stack = [initial]
    seen = {initial[0], initial[1]}
    dist = 0
    while stack:
        tmp = []  # temp array
        for [i, j] in stack:
            # if the coordinates match with the cordinates in final return distance
            if [i, j] == final: return dist

            # if we have visitied already then loop
            if (i, j) in seen: continue
            # add coordinated to seen list
            seen.add((i, j))
            for row in range(-1, 2):
                 for col in range(-1, 2):
                     if InBound(i + row, j + col) and (i + row, j + col) not in seen:
                         if [i + row, j + col] == final:
                             return dist + 1
                         else:
                             tmp.append([i + row, j + col])
            # # Left
            # if InBound(i - 1, j) and (i - 1, j) not in seen:
            #     if [i - 1, j] == final:
            #         return dist + 1
            #     else:
            #         tmp.append([i - 1, j])
            # # right
            # if InBound(i + 1, j) and (i + 1, j) not in seen:
            #     if [i + 1, j] == final:
            #         return dist + 1
            #     else:
            #         tmp.append([i + 1, j])
            # # up
            # if InBound(i, j + 1) and (i, j + 1) not in seen:
            #     if [i, j + 1] == final:
            #         return dist + 1
            #     else:
            #         tmp.append([i, j + 1])
            # # down
            # if InBound(i, j - 1) and (i, j - 1) not in seen:
            #     if [i, j - 1] == final:
            #         return dist + 1
            #     else:
            #         tmp.append([i, j - 1])
        dist += 1
        stack = tmp
    return dist

def GetManhattanDistance(initial, final) -> int:
    # We gunna make a stack with the given coordinates and add to the seen list
    stack = [initial]
    seen = {initial[0], initial[1]}
    dist = 0
    while stack:
        tmp = []  # temp array
        for [i, j] in stack:
            # if the coordinates match with the cordinates in final return distance
            if [i, j] == final: return dist

            # if we have visitied already then loop
            if (i, j) in seen: continue
            # add coordinated to seen list
            seen.add((i, j))
            # for row in range(-1, 2):
            #      for col in range(-1, 2):
            #          if InBound(i + row, j + col) and (i + row, j + col) not in seen:
            #              if [i + row, j + col] == final:
            #                  return dist + 1
            #              else:
            #                  tmp.append([i + row, j + col])
            # Left
            if InBound(i - 1, j) and (i - 1, j) not in seen:
                if [i - 1, j] == final:
                    return dist + 1
                else:
                    tmp.append([i - 1, j])
            # right
            if InBound(i + 1, j) and (i + 1, j) not in seen:
                if [i + 1, j] == final:
                    return dist + 1
                else:
                    tmp.append([i + 1, j])
            # up
            if InBound(i, j + 1) and (i, j + 1) not in seen:
                if [i, j + 1] == final:
                    return dist + 1
                else:
                    tmp.append([i, j + 1])
            # down
            if InBound(i, j - 1) and (i, j - 1) not in seen:
                if [i, j - 1] == final:
                    return dist + 1
                else:
                    tmp.append([i, j - 1])
        dist += 1
        stack = tmp
    return dist


# Calculates the whole Euclidian Cost for the state
def CalculateEuclidianCost(state) -> int:
    total_cost = 0
    for x in range(len(state)):
        for y in range(len(state[x])):
            position = [x, y]
            value = state[x][y]
            if position == final_state[value]:  # if the coordinate matches the final states coordinate at given value
                continue
            total_cost += GetEuclidianDistance(position, final_state[
                value])  # if not get the euclidean distance for that coordinate and where it should be
    return total_cost

def CalculateManhattanCost(state) -> int:
    total_cost = 0
    for x in range(len(state)):
        for y in range(len(state[x])):
            position = [x, y]
            value = state[x][y]
            if position == final_state[value]:  # if the coordinate matches the final states coordinate at given value
                continue
            total_cost += GetManhattanDistance(position, final_state[
                value])  # if not get the euclidean distance for that coordinate and where it should be
    return total_cost

# Calculated number of misplaced tiles at the current state
def CalculateTileDiffCost(state) -> int:
    total_tiles = 0
    for x in range(len(state)):
        for y in range(len(state[x])):
            position = [x, y]
            value = state[x][y]
            if position == final_state[value]:
                continue
            else:
                total_tiles += 1
    return total_tiles


def CalculateCost(heruistic, state) -> int:
    # TODO: Implement 'TileDiff' and enable 'if' check below.
    if heruistic == 'Euclidian':
        return CalculateEuclidianCost(state)
    elif heruistic == 'Misplaced':
        return CalculateTileDiffCost(state)
    else:
        return CalculateManhattanCost(state)


# Returning a list of the possible moves. I.O.W: children
def GetPossibleMoves(state, h):
    ## find zero
    pos = [0, 0]
    graph = state.graph()
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if 0 == graph[i][j]:
                pos = [i, j]
                break
    #
    retv = []
    i, j = pos
    ## Check up, left, right , down
    if InBound(i - 1, j):
        temp = copy.deepcopy(graph)
        temp[i][j], temp[i - 1][j] = temp[i - 1][j], temp[i][j]
        retv.append(State(temp, CalculateCost(h, temp), (state.getDepth() + 1)))
    if InBound(i, j - 1):
        temp = copy.deepcopy(graph)
        temp[i][j], temp[i][j - 1] = temp[i][j - 1], temp[i][j]
        retv.append(State(temp, CalculateCost(h, temp), (state.getDepth() + 1)))
    if InBound(i, j + 1):
        temp = copy.deepcopy(graph)
        temp[i][j], temp[i][j + 1] = temp[i][j + 1], temp[i][j]
        retv.append(State(temp, CalculateCost(h, temp), (state.getDepth() + 1)))
    if InBound(i + 1, j):
        temp = copy.deepcopy(graph)
        temp[i][j], temp[i + 1][j] = temp[i + 1][j], temp[i][j]
        retv.append(State(temp, CalculateCost(h, temp), (state.getDepth() + 1)))
    return retv  # list of States


def search(grid, h):
    initial_state = State(grid, CalculateCost(h, grid), 0)
    print("Expanding:\n")
    print(initial_state.graph())
    expandedNodes = 1
    tempChild = -1
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    states = [initial_state]
    hp.heapify(states)
    seen = set() #going to be a set of tuples
    total_moves = 0
    current = initial_state
    print("\t\tStarting")
    while states:
        current = hp.heappop(states) #Get the smallest and re adjust order
        if current.flatten() == goal_state: break

        if total_moves >= 0:
            print("\nThe best state to expand with g(n)= " + str(current.depth) + " and h(n) = " + str(current.cost) + "\n")
            print(current.graph())
            print("-------------------------------------------")
        total_moves += 1
        seen.add(tuple(current.flatten()))
        children = GetPossibleMoves(current, h)
        expandedNodes += 1
        tempChild = max(len(children), tempChild)
        for child_state in children:
            if set(child_state.flatten()) in seen: continue
            hp.heappush(states, child_state) #order is adjusted
            seen.add(tuple(child_state.flatten()))
    if current.flatten() != goal_state:
        print("Not possible to solve")
        return
    print("\tGOAL!\n")
    print(current.graph())
    print("-------------------------------------------")
    print("To solve this problem the search algorithm expanded a total of " + str(expandedNodes) + " nodes.")
    print("The maximum number of nodes in the queue at any one time: " + str(tempChild))
    print("The depth of the goal node was: " + str(current.depth))


# Gets Graph from difficult selected
def getGraph(diff):
    if diff == 1:
        return [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 0]]
    elif diff == 2:
        return [[1, 2, 3],
                [4, 5, 6],
                [7, 0, 8]]
    elif diff == 3:
        return [[1, 2, 0],
                [4, 5, 3],
                [7, 8, 6]]
    elif diff == 4:
        return [[0, 1, 2],
                [4, 5, 3],
                [7, 8, 6]]
    else:
        return [[8, 7, 1],
                [6, 0, 2],
                [5, 4, 3]]


def main(*args):
    print("Welcome to aavit004 8 puzzle solver,")
    defaultOrCustom = input('\nType a 1 to use a default puzzle, or 2 to enter your own puzzle')
    defaultOrCustom = int(defaultOrCustom)

    if defaultOrCustom == 1:  # Default
        difficulty = input('\nPlease choose the difficulty: \n'
                           '1. Trivial\n'
                           '2. Very Easy\n'
                           '3. Easy\n'
                           '4. Doable\n'
                           '5. Oh Boy\n'
                           ': ')
        difficulty = int(difficulty)
        grid = getGraph(difficulty)
        searchAlgo = input('\nEnter your choice of Algorithm: \n'
                           '1. Uniform Cost Search\n'
                           '2. A* with the Misplaced Tile heuristic function\n'
                           '3. A* with the Euclidian distance heuristic function\n'
                           ': ')
        searchAlgo = int(searchAlgo)

        if searchAlgo == 1:
            search(grid, "None")
        elif searchAlgo == 2:
            search(grid, "Misplaced")
        else:
            search(grid, "Euclidian")
    elif defaultOrCustom == 2: #custom
        print("\nPlease enter a single row with the numbers separated by commas. EX: 1,2,3")
        firstRow = (input("Enter the first row: ")).split(',')
        secondRow = (input("Enter the second row: ")).split(',')
        thirdRow = (input("Enter the third row: ")).split(',')

        for i in range(0,3):
            firstRow[i] = int(firstRow[i])
            secondRow[i] = int(secondRow[i])
            thirdRow[i] = int(thirdRow[i])
        
        grid = firstRow, secondRow, thirdRow

        searchAlgo = input('\nEnter your choice of Algorithm: \n'
                           '1. Uniform Cost Search\n'
                           '2. A* with the Misplaced Tile heuristic function\n'
                           '3. A* with the Euclidian distance heuristic function\n'
                           ': ')
        searchAlgo = int(searchAlgo)
        if searchAlgo == 1:
            search(grid, "None")
        elif searchAlgo == 2:
            search(grid, "Misplaced")
        else:
            search(grid, "Euclidian")
        input
    #     break;


if __name__ == '__main__':
    main()