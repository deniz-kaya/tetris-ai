from intelligence import Agent
games = 5000
gamma = 0.97
epsilon = 1
learningRate = 1e-3
epsilonDecayRate = 1e-3
finalEpsilon = 0.03
backupInterval = 500
batchSize = 32
memorySize = 10000
agent = Agent(gamma, epsilon, learningRate, epsilonDecayRate, finalEpsilon, backupInterval, batchSize, memorySize)

agent.train(games)