from intelligence import Agent
games = 100
gamma = 0.99
epsilon = 1
learningRate = 5e-4
epsilonDecayRate = 1e-2
finalEpsilon = 0.03
backupInterval = 500
batchSize = 128
memorySize = 30000
agent = Agent(gamma, epsilon, learningRate, epsilonDecayRate, finalEpsilon, backupInterval, batchSize, memorySize)

agent.train(games)


