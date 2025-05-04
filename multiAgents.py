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
        def alphaBeta(state):
            # Hàm alphaBeta chính - chọn hành động tốt nhất cho Pacman (agent 0)
            bestValue, bestAction = float("-inf"), None  # Khởi tạo giá trị và hành động tốt nhất
            alpha, beta = float("-inf"), float("inf")    # Khởi tạo giá trị alpha và beta
            
# Kiểm tra nếu không có hành động hợp lệ
            legalActions = state.getLegalActions(0)
            if not legalActions:
                return Directions.STOP
                
            # Duyệt qua tất cả các hành động hợp lệ của Pacman
            for action in legalActions:
                # Lấy giá trị của hành động từ hàm minValue (vì đây là ghost's turn tiếp theo)
                value = minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta)
                
                # Cập nhật hành động tốt nhất nếu tìm thấy giá trị cao hơn
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                
                # Cập nhật alpha (giá trị tốt nhất cho MAX)
                alpha = max(alpha, bestValue)

            # Đảm bảo luôn trả về một hành động hợp lệ
            if bestAction is None and legalActions:
                bestAction = legalActions[0]
                
            return bestAction  # Trả về hành động tốt nhất

        def maxValue(state, agentIdx, depth, alpha, beta):
            # Hàm maxValue - chọn giá trị MAX cho Pacman (agent 0)
            
            # Kiểm tra nếu đã đạt đến độ sâu giới hạn
            if depth > self.depth:
                return self.evaluationFunction(state)
            
            # Kiểm tra nếu không có hành động hợp lệ
            if len(state.getLegalActions(agentIdx)) == 0:
                return self.evaluationFunction(state)
            
            # Khởi tạo giá trị MIN
            value = float("-inf")
            
            # Duyệt qua tất cả các hành động hợp lệ
            for action in state.getLegalActions(agentIdx):
                # Tính giá trị của successor state bằng cách gọi minValue
                value = max(value, minValue(state.generateSuccessor(agentIdx, action), 
                                          agentIdx + 1, depth, alpha, beta))
                
                # Cắt nhánh beta: nếu value > beta, không cần xét các nhánh còn lại
                if value > beta:
                    return value
                
                # Cập nhật alpha
                alpha = max(alpha, value)
                
            return value

        def minValue(state, agentIdx, depth, alpha, beta):
            # Hàm minValue - chọn giá trị MIN cho ghost (agent >= 1)
            
            # Kiểm tra nếu trạng thái là win hoặc lose
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Nếu đã xét hết tất cả các ghost trong một lần lặp
            if agentIdx == state.getNumAgents():
                # Chuyển về lượt của Pacman ở độ sâu tiếp theo
                return maxValue(state, 0, depth + 1, alpha, beta)
            
            # Khởi tạo giá trị MIN
            value = float("inf")
            
            # Duyệt qua tất cả các hành động hợp lệ của ghost
            for action in state.getLegalActions(agentIdx):
                # Tính giá trị của successor state
                value = min(value, minValue(state.generateSuccessor(agentIdx, action), 
                                          agentIdx + 1, depth, alpha, beta))
                
                # Cắt nhánh alpha: nếu value < alpha, không cần xét các nhánh còn lại
                if value < alpha:
                    return value
                
                # Cập nhật beta
                beta = min(beta, value)
            
            # Trường hợp không có hành động hợp lệ
            if value == float("inf"):
                return self.evaluationFunction(state)
                
            return value

        # Bắt đầu tìm kiếm với hàm alphaBeta từ gameState hiện tại
        return alphaBeta(gameState)


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
                    
            return bestAction  # Trả về hành động tốt nhất

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


# def newAdvancedEvaluationFunction(currentGameState):
#     """
#     Hàm đánh giá nâng cao với chiến lược tối ưu để Pacman đạt hiệu suất cao nhất:
    
#     Chiến lược tối ưu:
#     1. Xử lý ưu tiên cao nhất cho trạng thái thắng/thua tức thì
#     2. Đánh giá food dựa trên số lượng và phân bố không gian
#     3. Phân tích chi tiết các ghost (nguy hiểm vs có thể ăn)
#     4. Sử dụng capsule có chiến lược dựa vào hoàn cảnh
#     5. Tận dụng tối đa thời gian scared của ghost
#     6. Thay đổi chiến thuật theo tiến độ trò chơi
#     """
#     # Ưu tiên cao nhất: Trạng thái kết thúc
#     if currentGameState.isWin():
#         return float('inf')
#     if currentGameState.isLose():
#         return float('-inf')
    
#     # Thu thập thông tin trạng thái
#     position = currentGameState.getPacmanPosition()
#     food = currentGameState.getFood()
#     ghostStates = currentGameState.getGhostStates()
#     capsules = currentGameState.getCapsules()
#     score = currentGameState.getScore() 
    
#     # === ĐÁNH GIÁ FOOD ===
#     foodList = food.asList()
#     foodCount = len(foodList)
    
#     # Tính toán khoảng cách đến food
#     if foodList:
#         # Khoảng cách đến tất cả các food
#         foodDistances = [manhattanDistance(position, foodPos) for foodPos in foodList]
#         # Food gần nhất
#         closestFoodDist = min(foodDistances)
        
#         # Phân cụm food - tìm 5 food gần nhất (nếu có)
#         sortedFoodDist = sorted(foodDistances)[:min(5, len(foodDistances))]
#         avgNearFoodDist = sum(sortedFoodDist) / len(sortedFoodDist)
        
#         # Phân tích phân bố food
#         stdDevFood = sum((dist - avgNearFoodDist)**2 for dist in sortedFoodDist)
#         if len(sortedFoodDist) > 1:
#             stdDevFood /= len(sortedFoodDist) - 1
#         stdDevFood = stdDevFood**0.5  # Độ lệch chuẩn của khoảng cách
#     else:
#         # Không còn food = thắng
#         closestFoodDist = 0
#         avgNearFoodDist = 0
#         stdDevFood = 0
    
#     # === ĐÁNH GIÁ GHOST ===
#     dangerousGhosts = []  # Ghost thường (nguy hiểm)
#     edibleGhosts = []     # Ghost scared (có thể ăn)
#     scaredTimers = []     # Thời gian scared còn lại
    
#     # Phân loại ghost
#     for ghost in ghostStates:
#         if ghost.scaredTimer > 0:
#             edibleGhosts.append(ghost)
#             scaredTimers.append(ghost.scaredTimer)
#         else:
#             dangerousGhosts.append(ghost)
    
#     # === ĐÁNH GIÁ GHOST NGUY HIỂM ===
#     dangerScore = 0
#     minDangerousDist = float('inf')
    
#     if dangerousGhosts:
#         # Khoảng cách đến ghost nguy hiểm
#         dangerDistances = [manhattanDistance(position, ghost.getPosition()) for ghost in dangerousGhosts]
#         minDangerousDist = min(dangerDistances)
        
#         # Độ nguy hiểm dựa trên khoảng cách
#         if minDangerousDist <= 1:  # Ghost sát bên - cực kỳ nguy hiểm
#             dangerScore = -1000
#         elif minDangerousDist <= 2:  # Ghost rất gần - nguy hiểm cao
#             dangerScore = -500
#         elif minDangerousDist < 4:  # Ghost gần - cần chú ý
#             dangerScore = -200 / minDangerousDist
#         else:  # Ghost ở xa - ít nguy hiểm
#             dangerScore = 50  # Ghost ở xa là an toàn, nên thưởng điểm
#     else:
#         # Không có ghost nguy hiểm = an toàn
#         dangerScore = 100
    
#     # === ĐÁNH GIÁ GHOST CÓ THỂ ĂN ===
#     edibleGhostScore = 0
#     canEatGhosts = False
    
#     for ghost in edibleGhosts:
#         ghostPos = ghost.getPosition()
#         dist = manhattanDistance(position, ghostPos)
#         remainingScaredTime = ghost.scaredTimer
        
#         # Kiểm tra xem có thể ăn ghost này không
#         if dist < remainingScaredTime:
#             # Cân bằng giữa lợi ích và thời gian còn lại
#             timeValue = remainingScaredTime - dist  # Lượng thời gian dư ra sau khi ăn ghost
            
#             # Điểm thưởng cho việc ăn ghost (càng gần càng tốt)
#             ghostValue = 200 - (dist * 8)
            
#             # Ghost ở xa hơn sẽ có giá trị thấp hơn theo hàm mũ
#             if dist > 3:
#                 ghostValue *= 0.8
                
#             edibleGhostScore += ghostValue
#             canEatGhosts = True
    
#     # === ĐÁNH GIÁ CAPSULE ===
#     capsuleScore = 0
#     minCapsuleDist = float('inf')
    
#     if capsules:
#         # Khoảng cách đến các capsule
#         capsuleDistances = [manhattanDistance(position, caps) for caps in capsules]
#         minCapsuleDist = min(capsuleDistances)
        
#         # Chiến lược capsule thông minh:
        
#         # 1. Nếu có ghost nguy hiểm gần, capsule trở nên quý giá
#         if minDangerousDist < 5:
#             # Độ cấp bách phụ thuộc vào khoảng cách của ghost nguy hiểm
#             urgency = max(0, (5 - minDangerousDist)) * 15
#             capsuleScore = 150 - (minCapsuleDist * urgency)
            
#             # Nếu ghost đang rất gần, capsule là ưu tiên hàng đầu
#             if minDangerousDist <= 2:
#                 capsuleScore *= 1.5
                
#         # 2. Nếu đã có nhiều ghost scared, giảm giá trị của capsule
#         elif sum(scaredTimers) > 15:
#             capsuleScore = 30 - minCapsuleDist
            
#         # 3. Trường hợp thông thường - capsule có giá trị trung bình
#         else:
#             capsuleScore = 70 - (minCapsuleDist * 3)
    
#     # === ĐÁNH GIÁ THỜI GIAN SCARED ===
    
#     # Tổng thời gian scared còn lại
#     totalScaredTime = sum(scaredTimers)
    
#     # Chiến lược sử dụng thời gian scared
#     scaredTimeScore = 0
    
#     if totalScaredTime > 0:
#         # Mỗi đơn vị thời gian scared đáng giá 8 điểm
#         scaredTimeScore = totalScaredTime * 8
        
#         # Tính mức độ tối ưu của thời gian scared
#         timeRemaining = [ghost.scaredTimer for ghost in edibleGhosts]
#         timeStdDev = 0
        
#         if len(timeRemaining) > 1:
#             avgTime = sum(timeRemaining) / len(timeRemaining)
#             timeStdDev = sum((t - avgTime)**2 for t in timeRemaining)
#             timeStdDev = (timeStdDev / (len(timeRemaining) - 1)) ** 0.5
            
#             # Thưởng nếu thời gian scared của các ghost đồng đều
#             # (nghĩa là capsule được ăn hợp lý, không chồng chéo thời gian)
#             if timeStdDev < 5:
#                 scaredTimeScore += 50
    
#     # === CHIẾN LƯỢC GIAI ĐOẠN TRÒ CHƠI ===
#     progressScore = 0
#     initialFoodCount = 30  # Ước tính số food ban đầu (có thể điều chỉnh)
    
#     # Tính tiến trình trò chơi dựa trên số food đã ăn
#     progressPercentage = 1 - (foodCount / initialFoodCount)
    
#     # Đầu game: Tập trung vào capsule và an toàn
#     if progressPercentage < 0.3:
#         progressScore = capsuleScore * 1.2
        
#     # Giữa game: Cân bằng giữa ăn ghost và food
#     elif progressPercentage < 0.7:
#         if canEatGhosts:
#             progressScore = edibleGhostScore * 1.3
#         else:
#             progressScore = -10 * closestFoodDist
            
#     # Cuối game: Ưu tiên cao nhất cho việc ăn nốt food
#     else:
#         progressScore = -25 * foodCount - 15 * closestFoodDist
    
#     # === TÍNH ĐIỂM TỔNG HỢP ===
#     finalScore = score  # Bắt đầu từ điểm hiện tại
    
#     # Cộng các thành phần điểm
#     finalScore += progressScore         # Chiến lược theo tiến trình
#     finalScore += dangerScore           # Tránh ghost nguy hiểm
#     finalScore += edibleGhostScore      # Săn ghost scared
#     finalScore += scaredTimeScore       # Tối ưu thời gian scared
#     finalScore -= 8 * foodCount         # Ít food là tốt
#     finalScore -= 2 * closestFoodDist   # Food gần là tốt
    
#     # Phạt điểm cho việc có nhiều food gần nhau mà không ăn
#     if stdDevFood < 2 and len(sortedFoodDist) > 2:
#         finalScore -= 30  # Phạt nếu có nhiều food gần nhau mà không ăn
        
#     # Điều chỉnh cuối cùng dựa trên số lượng food còn lại
#     if foodCount <= 2:  # Gần thắng
#         finalScore -= 40 * foodCount
#         finalScore -= 20 * closestFoodDist
    
#     return finalScore

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
        score += FOOD_WEIGHT / (1 + 2*minFoodDist)  # càng gần càng tốt
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
