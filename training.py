from intelligence import Agent
games = 2000
gamma = 0.99
epsilon = 0.1
learningRate = 1e-3
epsilonDecayRate = 5e-5
finalEpsilon = 0.003
backupInterval = 500
batchSize = 256
memorySize = 30000
agent = Agent(gamma, epsilon, learningRate, epsilonDecayRate, finalEpsilon, backupInterval, batchSize, memorySize)

agent.train(games,  modelPath="C:\\Users\\blind\PycharmProjects\\tetris-ai\models\\adam")

# , "C:\\Users\\blind\PycharmProjects\\tetris-ai\models\decent1_games", "C:\\Users\\blind\PycharmProjects\\tetris-ai\models\decent1"

# "C:\\Users\\blind\PycharmProjects\\tetris-ai\models\\decent1_games",