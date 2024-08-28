import torch as torch
import torch.nn as nn
import numpy as np
from collections import deque
from game import Tetris
import random
class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3= nn.Linear(4,1)

    def forward(self, state):
        x = nn.ReLU(self.fc1(state))
        x = nn.ReLU(self.fc2(x))
        predictions = self.fc3(x)

        return predictions
class Agent():

    def __init__(self, gamma, startingEpsilon, learningRate, epsilonDecayRate, finalEpsilon, backupInterval, batchSize, memorySize):
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
        self.lossFunc = nn.HuberLoss()

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
        return self.environment.getStateInfo()

    def learn(self):
        count = len(self.stateMemory) if self.batchSize > len(self.stateMemory) else self.batchSize

        batch = random.sample(self.stateMemory,count)
        states, rewards, nextStates, terminations = zip(*batch)

        states = torch.stack(states).reshape(-1,1)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).reshape(-1,1)
        nextStates = torch.stack(nextStates).reshape(-1,1)

        statePredictions = self.model(states)
        nextStatePredictions = self.getPredictions(nextStates)

        actualStateValues = torch.tensor(tuple(reward if gameOver
                                               else reward + self.gamma * prediction
                                               for reward, prediction, gameOver
                                               in zip(rewards, nextStatePredictions, terminations))).reshape(-1,1)

        self.optimizer.zero_grad()
        loss = self.lossFunc(statePredictions, actualStateValues)
        loss.backwards()
        self.optimizer.step()
    def train(self, games):
        epsilon = self.startingEpsilon
        self.environment.__init__()
        state = self.environment.getStateInfo()

        if torch.cuda.is_available():
            state = state.cuda()

        gameCount = 0
        while gameCount < games:
            epsilon = max(self.finalEpsilon, epsilon - self.epsilonDecayRate)

            possibleActionsValues = self.environment.getPossibleStateValues()
            currentActions, currentActionValues = zip(*possibleActionsValues.items())

            predictions = self.getPredictions(currentActions)

            if random.random() <= epsilon:
                actionIndex = random.randint(0, len(currentActions) -1)
            else:
                actionIndex = torch.argmax(predictions)

            nextState = currentActionValues[actionIndex, :]
            action = currentActions[actionIndex]

            reward, gameOver = self.environment.nextState(action)

            self.stateMemory.append([state, action, nextState, gameOver])

            if gameOver:
                print("Game: {}/{}, Final score: {}, Cleared lines: {}", gameCount, games, self.environment.score, self.environment.clearedLines)
                state = self.resetEnvironment()
            else:
                state = nextState
                continue
            gameCount += 1
            self.learn()