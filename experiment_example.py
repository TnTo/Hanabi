from Hanabi import Game, Action
from Experiment import NeuralNetwork, Experiment, N_PLAYERS, N_NETWORKS, N_PARALLEL_GAMES, MIRROR_PLAY, LEARN_FROM_PREVIOUS_EPISODES, LEARN_FROM_OTHERS_GAMES, LEARN_FROM_OTHERS_PLAYERS, N_EPOCHS, SHUFFLE_INPUT
from typing import Dict
import numpy as np
from keras import Model

N_PLAYERS: int = 5
N_NETWORKS: int = 1
N_PARALLEL_GAMES: int = 10
MIRROR_PLAY: bool = True
LEARN_FROM_PREVIOUS_EPISODES: bool = False
LEARN_FROM_OTHERS_GAMES: bool = True
LEARN_FROM_OTHERS_PLAYERS: bool = True
N_EPOCHS: int = 10
SHUFFLE_INPUT: bool = True

N_EPISODES = 20

#Create model
model = Model()

def epsilon(episode:int) -> float:
    if episode < 5:
        return 0.9
    elif episode < 10:
        return 0.5
    else:
        return 0.05

#define encoding
def encode(game:Game, action:Action) -> Dict[str, np.array]:
    return [np.array([0])]

NN = NeuralNetwork(model, epsilon, 0.95, encode)

#or

NNList = [NeuralNetwork(model, epsilon, 0.95, encode) for _ in range(N_NETWORKS)]

experiment = Experiment(NN)

experiment.run(N_EPISODES)

experiment.play()