import os
from typing import List

import numpy as np
import tensorflow.keras as keras

from Game import DGAME, DACTION, DTYPE, play_game, train, save, plot
from Game import (
    LIFES,
    HINTS,
    DISCARDED,
    PLAYERS,
    DPLAYER,
    DTILE,
    HAND_SIZE,
    DHANDTILE,
    PILES,
    LAST_ROUND,
)

NAME: str = "ex102"
MEMORY_SIZE: int = 0
N_EPISODES: int = 10
N_NEW_GAMES: int = 1000
N_OLD_GAMES: int = 0
EPSILON: float = 0.95
GAMMA: float = 0.95

INPUT: List[int] = (
    LIFES
    + HINTS
    + list(DISCARDED)
    + [
        PLAYERS[:DPLAYER][t * DHANDTILE + i]
        for t in range(HAND_SIZE)
        for i in range(DTILE)
    ]
    + list(PLAYERS[DPLAYER:])
    + list(PILES)
    + LAST_ROUND
)

if os.path.exists(NAME):
    model = keras.models.load_model(os.path.join(NAME, "nn"))
    memories = np.load(os.path.join(NAME, "memories.npy"))
    points = np.load(os.path.join(NAME, "points.npy"))
    loss = np.load(os.path.join(NAME, "loss.npy"))
else:
    inputs = keras.Input(shape=(len(INPUT),))
    hidden1 = keras.layers.Dense(10, activation="softplus")(inputs)
    hidden2 = keras.layers.Dense(10, activation="softplus")(hidden1)
    Q = keras.layers.Dense(1, activation="relu")(hidden2)
    model = keras.Model(inputs=inputs, outputs=Q)
    model.summary()
    model.compile(loss="mse", optimizer="rmsprop")

    memories = np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE)
    points = np.empty((0, 4))
    loss = np.empty((0, 1))

    memories, points = play_game(0, MEMORY_SIZE, 100, 0, memories, points, model, INPUT)
    loss = train(0, memories, loss, model, INPUT, patience=2)
    save(memories, points, loss, model, NAME)


for e in range(N_EPISODES):
    print("EPISODE ", e + 1)
    memories, points = play_game(
        EPSILON,
        MEMORY_SIZE,
        N_NEW_GAMES,
        N_OLD_GAMES,
        memories,
        points,
        model,
        INPUT,
    )
    loss = train(GAMMA, memories, loss, model, INPUT)
    save(memories, points, loss, model, NAME)

print("RESULT")
memories, points = play_game(1, 0, N_NEW_GAMES, 0, memories, points, model, INPUT)
save(memories, points, loss, model, NAME, last=True)

plot(loss, points, NAME)
