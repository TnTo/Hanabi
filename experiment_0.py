# Single Dumb NN

from Experiment import NeuralNetwork, Experiment
import Experiment as ExperimentVars
from Hanabi import Game, Action, Player, Color, Number, Info, Hint, Discard, Play, handSize
from typing import Dict, List
import numpy as np
import tensorflow.keras as keras
from itertools import cycle
import cProfile
import pstats

profile = cProfile.Profile()
profile.enable()

ExperimentVars.N_EPOCHS = 5

N_EPISODES = 1

# Create model
inputs = keras.Input(shape=(51,))
hidden = keras.layers.Dense(100, activation="relu")(inputs)
Q = keras.layers.Dense(1)(hidden)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()


def epsilon(episode: int) -> float:
    if episode == 1:
        return 1
    elif episode < 5:
        return 0.9
    elif episode < 10:
        return 0.5
    else:
        return 0.05


# define encoding

def colorToInt(color: Color) -> int:
    if color == Color.BLUE:
        return 1
    elif color == Color.GREEN:
        return 2
    elif color == Color.RED:
        return 3
    elif color == Color.WHITE:
        return 4
    elif color == Color.YELLOW:
        return 5
    else:
        return 0


def infoToInt(info: Info) -> int:
    if info == Info.YES:
        return 1
    elif info == Info.UNKNOWN:
        return 0
    elif info == Info.NO:
        return -1
    else:
        return 0


def encode(game: Game, action: Action, activePlayer: Player) -> Dict[str, np.array]:
    inputs: List[float] = [game.hints, game.lifes]  # +2 = 2 Hints Lifes
    for color in Color:
        inputs.append(int(game.piles[color] or 0))  # +C = 7 Piles
    # add discarded
    players = cycle(game.players)
    player = next(players)
    while player != activePlayer:
        player = next(players)
    for tile in player.hand:  # +2H = 15 Active Hand
        if tile.numberInfo[tile.tile.number] == Info.YES:
            inputs.append(int(tile.tile.number or 0))
        else:
            inputs.append(0)
        if tile.colorInfo[tile.tile.color] == Info.YES:
            inputs.append(colorToInt(tile.tile.color))
        else:
            inputs.append(0)
    if len(player.hand) < handSize(ExperimentVars.N_PLAYERS):
        inputs.extend([0, 0] * (handSize(ExperimentVars.N_PLAYERS) - len(player.hand)))
    for _ in range(ExperimentVars.N_PLAYERS - 1):  # +2HP = 47 Others Hand
        player = next(players)
        for tile in player.hand:
            inputs.extend([int(tile.tile.number or 0), colorToInt(tile.tile.color)])
        if len(player.hand) < handSize(ExperimentVars.N_PLAYERS):
            inputs.extend([0, 0] * (handSize(ExperimentVars.N_PLAYERS) - len(player.hand)))
    if isinstance(action, Hint):  # +4 = 51
        inputs.append(1)
        inputs.append((game.players.index(action.player) - game.players.index(activePlayer)) % ExperimentVars.N_PLAYERS)
        if isinstance(action.feature, Number):
            inputs.extend([action.feature, 0])
        elif isinstance(action.feature, Color):
            inputs.extend([0, colorToInt(action.feature)])
        else:
            inputs.extend([0, 0])
    elif isinstance(action, Discard):
        inputs.append(2)
        pos = None
        for n in range(len(activePlayer.hand)):
            if activePlayer.hand[n].tile is action.tile:
                pos = n
        if pos:
            inputs.append(pos)
        else:
            inputs.append(0)
        inputs.extend([0, 0])
    elif isinstance(action, Play):
        inputs.append(2)
        pos = None
        for n in range(len(activePlayer.hand)):
            if activePlayer.hand[n].tile is action.tile:
                pos = n
        if pos:
            inputs.append(pos)
        else:
            inputs.append(0)
        inputs.extend([0, 0])
    else:
        inputs.extend([0, 0, 0, 0])
    return {'inputs': np.expand_dims(np.array(inputs), axis=0)}


NN = NeuralNetwork(model, epsilon, 0.95, encode)

experiment = Experiment(NN)

experiment.run(N_EPISODES)

experiment.play()

profile.disable()
ps = pstats.Stats(profile)
ps.sort_stats('calls', 'ncalls')
ps.print_stats(10)
ps.sort_stats('calls', 'percall')
ps.print_stats(10)
