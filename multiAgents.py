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
        newGhostPositions = successorGameState.getGhostPositions()
        #print(newFood.asList(), ",", successorGameState, ",", newPos, ",", newScaredTimes, ",", newGhostPositions)
        dist_ghost=abs(newPos[0]-newGhostPositions[0][0])+ abs(newPos[1]-newGhostPositions[0][1])
        min=1000
        for i in range (len(newFood.asList())):
            d=abs(newFood.asList()[i][0]-newPos[0])+abs(newFood.asList()[i][1]-newPos[1])
            if d<min:
                min=d
        value= 50*(1/min)+dist_ghost+newScaredTimes[0]
        #print(min)
        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        return value+100*successorGameState.getScore()

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

        totalAgents = gameState.getNumAgents()
        maxDepth = self.depth
        legalMoves = gameState.getLegalActions(0) # legal actions for Pacman
        values = []

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(0, action)
            nextAgent = 1 % totalAgents # next agent is 0 (Pacman) if there is only 1 agent, i.e. Pacman, else it is 1
            nextDepth = 1 + (1//totalAgents) # next depth is 2 if there is only 1 agent, i.e. Pacman, else it is 1
            values.append(self.minimaxValue(successorGameState, nextAgent, totalAgents, nextDepth, maxDepth))

        print(values)
        bestValue = max(values)
        bestIndices = [index for index in range(len(values)) if values[index] == bestValue]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def minimaxValue(self, gameState, agentNum, totalAgents, curDepth, maxDepth):

        if curDepth == maxDepth+1 and agentNum == 0: # reached the state obtained after action of last agent of maximum depth
            return self.evaluationFunction(gameState)

        else:

            legalMoves = gameState.getLegalActions(agentNum)

            if len(legalMoves) == 0: # a terminal node
                return self.evaluationFunction(gameState)

            bestValue = None

            for action in legalMoves:

                successorGameState = gameState.generateSuccessor(agentNum, action)
                nextAgent = (agentNum + 1) % totalAgents
                nextDepth = curDepth + ((agentNum + 1)//totalAgents)
                successorValue = self.minimaxValue(successorGameState, nextAgent, totalAgents, nextDepth, maxDepth)

                if bestValue == None:
                    bestValue = successorValue
                elif agentNum == 0: # agent is Pacman, max-node
                    bestValue = max(bestValue, successorValue)
                else: # agent is a Ghost, min-node
                    bestValue = min(bestValue, successorValue)

            return bestValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents()
        maxDepth = self.depth
        legalMoves = gameState.getLegalActions(0) # legal actions for Pacman
        values = []

        alpha = None
        beta = None
        bestValue = None
        bestIndices = []

        for i in range(len(legalMoves)):

            action = legalMoves[i]

            successorGameState = gameState.generateSuccessor(0, action)
            nextAgent = 1 % totalAgents # next agent is 0 (Pacman) if there is only 1 agent, i.e. Pacman, else it is 1
            nextDepth = 1 + (1//totalAgents) # next depth is 2 if there is only 1 agent, i.e. Pacman, else it is 1
            successorValue = self.prunedMinimaxValue(successorGameState, alpha, beta, nextAgent, totalAgents, nextDepth, maxDepth)

            if bestValue == None:
                bestValue = successorValue
                bestIndices = [i]
            else: # update bestValue so far and add corresponding indices
                if bestValue < successorValue:
                    bestValue = successorValue
                    bestIndices = [i]
                elif bestValue == successorValue:
                    bestIndices.append(i)
            
            # update alpha based on bestValue so far
            if alpha == None or alpha < bestValue: # alpha = max(alpha, bestValue)  
                alpha = bestValue

        # print(bestValue)
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def prunedMinimaxValue(self, gameState, alpha, beta, agentNum, totalAgents, curDepth, maxDepth):
        """
            alpha: best value for the first max-node reached via path to root
            beta: best value for the first min-node reached via path to root
        """

        if curDepth == maxDepth+1 and agentNum == 0: # reached the state obtained after action of last agent of maximum depth
            value = self.evaluationFunction(gameState)
            return value

        else:

            legalMoves = gameState.getLegalActions(agentNum)

            if len(legalMoves) == 0: # a terminal node
                value = self.evaluationFunction(gameState)
                return value

            bestValue = None

            for action in legalMoves:

                successorGameState = gameState.generateSuccessor(agentNum, action)
                nextAgent = (agentNum + 1) % totalAgents
                nextDepth = curDepth + ((agentNum + 1)//totalAgents)
                successorValue = self.prunedMinimaxValue(successorGameState, alpha, beta, nextAgent, totalAgents, nextDepth, maxDepth)

                if bestValue == None:
                    bestValue = successorValue
                elif agentNum == 0: # agent is Pacman, max-node
                    bestValue = max(bestValue, successorValue)
                else: # agent is a Ghost, min-node
                    bestValue = min(bestValue, successorValue)
                
                # update alpha or beta and prune based on bestValue so far
                if agentNum == 0: # max-node
                    if beta != None and bestValue > beta:
                        return bestValue
                    if alpha == None or alpha < bestValue: # alpha = max(alpha, bestValue)  
                        alpha = bestValue           
                else: # min-node
                    if alpha != None and bestValue < alpha:
                        return bestValue
                    if beta == None or beta > bestValue: # beta = min(beta, bestValue)  
                        beta = bestValue

            return bestValue  


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
        totalAgents = gameState.getNumAgents()
        maxDepth = self.depth
        legalMoves = gameState.getLegalActions(0) # legal actions for Pacman
        values = []

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(0, action)
            #nextAgent = 1 % totalAgents # next agent is 0 (Pacman) if there is only 1 agent, i.e. Pacman, else it is 1
            #nextDepth = 1 + (1//totalAgents) # next depth is 2 if there is only 1 agent, i.e. Pacman, else it is 1
            values.append(self.expectimaxValue(successorGameState, 0, 1, maxDepth, totalAgents))

        #print(values)
        bestValue = max(values)
        bestIndices = [index for index in range(len(values)) if values[index] == bestValue]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def MaxValue(self, gameState, curDepth, maxDepth):
        if curDepth==maxDepth:
            return self.evaluationFunction(gameState)
        legalMoves=gameState.getLegalActions(0)
        numOfMoves=len(legalMoves)
        if numOfMoves==0:
            return self.evaluationFunction(gameState)
        
        maxValue=float("-inf")
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(0, action)
            totalAgents = gameState.getNumAgents()
            if self.expectimaxValue(successorGameState, curDepth, 1, maxDepth, totalAgents)>maxValue:
                maxValue=self.expectimaxValue(successorGameState, curDepth,1, maxDepth, totalAgents)
        
        return maxValue


    def expectimaxValue(self, gameState, curDepth, agentNum, maxDepth, totalAgents):

        if curDepth == maxDepth+1 and agentNum == 0: # reached the state obtained after action of last agent of maximum depth
            return self.evaluationFunction(gameState)

        else:

            legalMoves = gameState.getLegalActions(agentNum)
            numOfMoves=len(legalMoves)
            if numOfMoves == 0: # a terminal node
                return self.evaluationFunction(gameState)
            
            bestValue = 0
            if agentNum==totalAgents-1:
                for action in legalMoves:
                    successorGameState = gameState.generateSuccessor(agentNum, action)
                    bestValue+=self.MaxValue(successorGameState, curDepth+1, maxDepth)
                    bestValue=bestValue/numOfMoves
            else:
                for action in legalMoves:
                    successorGameState = gameState.generateSuccessor(agentNum, action)
                    bestValue+=self.expectimaxValue(successorGameState, curDepth, agentNum +1, maxDepth, totalAgents)
                    bestValue=bestValue/numOfMoves

            return bestValue

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
