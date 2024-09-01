from intelligence import Agent
games = 2000
gamma = 0.99
epsilon = 1
learningRate = 5e-4
epsilonDecayRate = 5e-4
finalEpsilon = 0.03
backupInterval = 500
batchSize = 256
memorySize = 30000
agent = Agent(gamma, epsilon, learningRate, epsilonDecayRate, finalEpsilon, backupInterval, batchSize, memorySize)

agent.play(1000, 'C:\\Users\\blind\PycharmProjects\\tetris-ai\models\\adam')
