import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from collections import deque
from game import Tetris
import pygame
import random
import time
class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, state):
        return self.model(state)
class Agent():

    def __init__(self, gamma, startingEpsilon, learningRate, epsilonDecayRate, finalEpsilon, backupInterval, batchSize, memorySize):
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()

        self.gamma = gamma
        self.startingEpsilon = startingEpsilon
        self.learningRate = learningRate
        self.epsilonDecayRate = epsilonDecayRate
        self.finalEpsilon = finalEpsilon
        self.backupInterval = backupInterval
        self.batchSize = batchSize
        self.memorySize = memorySize

        self.environment = Tetris()
        self.model = DeepQNetwork()

        self.stateMemory = deque(maxlen=memorySize)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.lossFunc = nn.MSELoss()

        if torch.cuda.is_available():
            self.model.cuda()
            self.stateMemory.cuda()

    def getPredictions(self, nextStates):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(nextStates)
        self.model.train()
        return predictions

    def resetEnvironment(self):
        self.environment.__init__()
        return self.environment.getStateInfo(self.environment.visibleBoard())

    def surface(self, blockSize: int, strokeSize: int):
        board = self.environment.visibleBoard()
        dimensions = numpy.shape(board)
        surface = pygame.Surface(
            (dimensions[1] * blockSize + 2 * strokeSize, dimensions[0] * blockSize + 2 * strokeSize))
        for cols in range(dimensions[1]):
            for rows in range(dimensions[0]):
                if board[rows, cols] != 0:
                    # print(Tetris.pieceColours[int(board[rows,cols])])
                    # print(pygame.Color(Tetris.pieceColours[int(board[rows,cols])]))
                    surface.fill(Tetris.boardColours[int(board[rows, cols])], (
                        (cols * blockSize + strokeSize, rows * blockSize + strokeSize), (blockSize, blockSize)))
                else:
                    surface.fill("black", (
                        (cols * blockSize + strokeSize, rows * blockSize + strokeSize), (blockSize, blockSize)))
                if not strokeSize == 0:
                    pygame.draw.rect(surface, (49, 49, 49),
                                     ((cols * blockSize + strokeSize, rows * blockSize + strokeSize),
                                      (blockSize, blockSize)),
                                     strokeSize)
        return surface

    def learn(self):
        count = len(self.stateMemory) if self.batchSize > len(self.stateMemory) else self.batchSize

        batch = random.sample(self.stateMemory,count)
        states, rewards, nextStates, terminations = zip(*batch)

        states = torch.stack(states).reshape(-1,4)
        if torch.cuda.is_available():
            states.cuda()
        rewards = torch.from_numpy(numpy.array(rewards, dtype=numpy.float32)).reshape(-1,1)
        nextStates = torch.stack(nextStates).reshape(-1,4)

        statePredictions = self.model(states)
        nextStatePredictions = self.getPredictions(nextStates)

        actualStateValues = torch.cat(tuple(reward if gameOver else reward + self.gamma * prediction for reward, prediction, gameOver in zip(rewards, nextStatePredictions, terminations))).reshape(-1,1)
        if torch.cuda.is_available():
            actualStateValues.cuda()
            statePredictions.cuda()
        self.optimizer.zero_grad()
        loss = self.lossFunc(statePredictions, actualStateValues)
        loss.backward()
        self.optimizer.step()
    def train(self, games):
        epsilon = self.startingEpsilon
        self.environment.__init__()
        state = self.environment.getStateInfo(self.environment.visibleBoard())

        if torch.cuda.is_available():
            state = state.cuda()

        gameCount = 0
        while gameCount < games:

            epsilon = max(self.finalEpsilon, epsilon - self.epsilonDecayRate)
            possibleActionsValues = self.environment.getPossibleStateValues()
            if possibleActionsValues == {}:
                print("Game: {}/{}, Final score: {}, Cleared lines: {}, Tetrominos: {}".format(gameCount, games, self.environment.score,
                                                                               self.environment.clearedLines, self.environment.piecesDropped))
                state = self.resetEnvironment()
                gameCount += 1
                self.learn()
                continue
            possibleActions, possibleStates = zip(*possibleActionsValues.items())
            possibleStates = torch.stack(possibleStates)
            predictions = self.getPredictions(possibleStates)

            if random.random() <= epsilon:
                actionIndex = random.randint(0, len(possibleActions) -1)
            else:
                actionIndex = torch.argmax(predictions).item()

            nextState = possibleStates[actionIndex, :]
            action = possibleActions[actionIndex]

            reward, gameOver = self.environment.nextState(action)

            if self.environment.piecesDropped % 20 == 0:
                gameSurface = self.surface(20,1)
                self.screen.blit(gameSurface, (50,50))
                pygame.display.update()
                self.clock.tick(10)

            self.stateMemory.append([state, reward, nextState, gameOver])

            if gameOver:
                print("Game: {}/{}, Final score: {}, Cleared lines: {}, Tetrominos: {}".format(gameCount, games,
                                                                                               round(self.environment.score, 1),
                                                                                               self.environment.clearedLines,
                                                                                               self.environment.piecesDropped))
                if self.environment.clearedLines > 500 or self.environment.piecesDropped > 750 or self.environment.score > 6000:
                    print("save model?")
                    ans = input()
                    if ans == "y":
                        print("give it a name rn")
                        name = input()
                        torch.save(self.model, "/models/{}".format(name))
                state = self.resetEnvironment()
            else:
                state = nextState
                continue
            gameCount += 1
            self.learn()