# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        if not legalMoves[chosenIndex]:
            return Directions.STOP
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                if action == 'Stop':
                    continue
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1, float("-Inf"), float("Inf"))
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth, alpha, beta):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1,alpha,beta)
            value = None
            for action in state.getLegalActions(agentIdx):
                if action == 'Stop':
                    continue
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth,alpha,beta)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                if action == 'Stop':
                    continue
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth,alpha,beta)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                    if value >= beta:
                        return value
                    alpha = max(alpha, value)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)

        return action


#############################################
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state):
            # Hàm expectimax chính - chọn hành động tốt nhất cho Pacman (agent 0)
            bestValue, bestAction = float("-inf"), None  # Khởi tạo giá trị và hành động tốt nhất
            
            # Duyệt qua tất cả các hành động hợp lệ của Pacman
            for action in state.getLegalActions(0):
                # Lấy giá trị kỳ vọng từ hàm expectValue
                value = expectValue(state.generateSuccessor(0, action), 1, 1)
                
                # Cập nhật hành động tốt nhất nếu tìm thấy giá trị cao hơn
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    
            return bestAction if bestAction is not None else Directions.STOP

        def maxValue(state, agentIdx, depth):
            # Hàm maxValue - chọn giá trị MAX cho Pacman (agent 0)
            
            # Kiểm tra nếu đã đạt đến độ sâu giới hạn
            if depth > self.depth:
                return self.evaluationFunction(state)
            
            # Kiểm tra nếu không có hành động hợp lệ
            if len(state.getLegalActions(agentIdx)) == 0:
                return self.evaluationFunction(state)
            
            # Khởi tạo giá trị tối thiểu
            value = float("-inf")
            
            # Duyệt qua tất cả các hành động hợp lệ
            for action in state.getLegalActions(agentIdx):
                # Tính giá trị kỳ vọng của successor state
                value = max(value, expectValue(state.generateSuccessor(agentIdx, action), 
                                             agentIdx + 1, depth))
                
            return value

        def expectValue(state, agentIdx, depth):
            # Hàm expectValue - tính giá trị kỳ vọng cho ghost (agent >= 1)
            
            # Kiểm tra nếu trạng thái là win hoặc lose
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Nếu đã xét hết tất cả các ghost trong một lần lặp
            if agentIdx == state.getNumAgents():
                # Chuyển về lượt của Pacman ở độ sâu tiếp theo
                return maxValue(state, 0, depth + 1)
            
            # Lấy danh sách các hành động hợp lệ
            legalActions = state.getLegalActions(agentIdx)
            
            # Kiểm tra nếu không có hành động hợp lệ
            if not legalActions:
                return self.evaluationFunction(state)
            
            # Tính tổng giá trị kỳ vọng cho tất cả các hành động có thể
            totalValue = 0
            for action in legalActions:
                # Cộng dồn giá trị kỳ vọng từ mỗi hành động
                totalValue += expectValue(state.generateSuccessor(agentIdx, action), 
                                        agentIdx + 1, depth)
            
            # Tính giá trị trung bình (giả định xác suất đồng đều)
            # Chia tổng giá trị cho số lượng hành động để lấy giá trị kỳ vọng
            return totalValue / len(legalActions)

        # Bắt đầu tìm kiếm với hàm expectimax từ gameState hiện tại
        return expectimax(gameState)


###############################################
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule


def newEvaluationFunction(currentGameState):
    """
    Một hàm đánh giá thông minh cho Pacman: kết hợp giữa vị trí hiện tại,
    thức ăn, ghost, capsule và các thông tin trạng thái khác để ra quyết định tốt hơn.
    """

    # Lấy thông tin cơ bản
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    score = currentGameState.getScore()

    # Nếu win hoặc lose thì trả điểm tuyệt đối
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    # --- Trọng số (tùy chỉnh)
    FOOD_WEIGHT = 16.0
    FOOD_COUNT_WEIGHT = 14.0
    CAPSULE_WEIGHT = 12.0
    DANGER_GHOST_WEIGHT = 20.0
    SCARED_GHOST_WEIGHT = 64.0
    TOTAL_SCARED_TIME_WEIGHT = 2

    # --- 1. Khoảng cách đến food gần nhất
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += FOOD_WEIGHT / (1 + 2* minFoodDist)  # càng gần càng tốt
        score -= FOOD_COUNT_WEIGHT * len(foodList)  # càng ít food còn lại càng tốt
    else:
        score += 100  # thưởng nếu không còn food

    # --- 2. Ghost không sợ → tránh xa
    for ghost, scaredTime in zip(ghostStates, scaredTimes):
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if scaredTime == 0:
            if ghostDist > 0:
                score -= DANGER_GHOST_WEIGHT / ghostDist  # tránh càng xa càng tốt
            else:
                return -float('inf')  # bị bắt
        else:
            score += SCARED_GHOST_WEIGHT / (1 + 2* ghostDist)  # tiến gần để ăn ghost

    # --- 3. Capsule → nếu gần và có ghost nguy hiểm → nên ăn, khi ghost nguy hiểm đang gần thì nên ăn hơn để có thể cắn ngược lại lẹ
    if capsuleList:
        minCapsuleDist = min(manhattanDistance(pacmanPos, cap) for cap in capsuleList)
        if any(s == 0 for s in scaredTimes):  # có ghost chưa sợ
            score += CAPSULE_WEIGHT / (1 + minCapsuleDist)
        
        dangerousGhosts = [
            ghost for ghost in ghostStates 
            if ghost.scaredTimer == 0 and manhattanDistance(pacmanPos, ghost.getPosition()) <= 3
        ]

        if dangerousGhosts:  # có ghost nguy hiểm đang gần
            score += CAPSULE_WEIGHT / (1 + 4* minCapsuleDist)

    # --- 4. Tổng scared time
    score += TOTAL_SCARED_TIME_WEIGHT * sum(scaredTimes)
    return score


# Đặt hàm mới này làm hàm đánh giá mặc định
better = newEvaluationFunction

#python pacman.py -l testClassic -p ExpectimaxAgent -a depth=3,evalFn=betterEvaluationFunction -s 23520127 -n 5 --frameTime 0