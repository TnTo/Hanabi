# import os
from typing import List
import random

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

# import matplotlib.pyplot as plt

from Game import (
    DGAME,
    DACTION,
    DTYPE,
    play_game,
    train,
    save,
    plot,
    print_game,
    print_action,
    # vscore
)
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
    HINT_COLOR,
    HINT_NUMBER,
    PLAYER,
    PLAY,
    DISCARD,
)

NAME: str = "Dino00"

SEED = 123456
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


INPUT: List[int] = (
    LIFES
    + HINTS
    + list(DISCARDED)
    + [
        PLAYERS[:DPLAYER][t * DHANDTILE + DTILE + i]
        for t in range(HAND_SIZE)
        for i in range(DHANDTILE - DTILE)
    ]
    + list(PLAYERS[DPLAYER:])
    + list(PILES)
    + LAST_ROUND
    + [i + DGAME for i in HINT_NUMBER]
    + [i + DGAME for i in HINT_COLOR]
    + [i + DGAME for i in PLAYER]
    + [i + DGAME for i in PLAY]
    + [i + DGAME for i in DISCARD]
)

inputs = keras.Input(shape=(len(INPUT),))
hidden1 = keras.layers.Dense(512, activation="sigmoid")(inputs)
hidden2 = keras.layers.Dense(512, activation="sigmoid")(hidden1)
Q = keras.layers.Dense(1, activation="relu")(hidden2)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()
model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=0.01))

memories = np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE)
points = np.empty((0, 4))
loss = np.empty((0, 1))


# CREATE MEMORIES DB
MEMORY_SIZE: int = 1
N_NEW_GAMES: int = 1
N_OLD_GAMES: int = 0
EPSILON: float = 0
GAMMA: float = 0.95


while memories.shape[0] < MEMORY_SIZE:
    print("Create memories")
    memories, _ = play_game(
        EPSILON,
        MEMORY_SIZE,
        N_NEW_GAMES,
        N_OLD_GAMES,
        memories,
        np.empty((0, 4)),
        model,
        INPUT,
        sample_memories=False,
    )

m = memories[0, :]
print(m)
print_game(m[:DGAME])
print_action(m[DGAME : DGAME + DACTION])
print(m[INPUT])

# Reset
points = np.empty((0, 4))
loss = np.empty((0, 1))

memories = np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE)

# Explore
MEMORY_SIZE = 50
N_NEW_GAMES = 5
N_OLD_GAMES = 5
N_EPISODES: int = 1
EPSILON = 0.5

for e in range(N_EPISODES):
    print("Exploring EPISODE ", e + 1, " of ", N_EPISODES)
    memories, _ = play_game(
        EPSILON,
        MEMORY_SIZE,
        N_NEW_GAMES,
        N_OLD_GAMES,
        memories,
        np.empty((0, 4)),
        model,
        INPUT,
    )
    loss = train(GAMMA, memories, loss, model, INPUT, patience=100)

    print("Testing")
    _, points = play_game(
        1,
        MEMORY_SIZE,
        N_NEW_GAMES,
        0,
        np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE),
        points,
        model,
        INPUT,
    )

save(memories, points, loss, model, NAME)
plot(loss, points, NAME)

# Exploit
N_EPISODES = 1
EPSILON = 0.9

for e in range(N_EPISODES):
    print("Exploiting EPISODE ", e + 1, " of ", N_EPISODES)
    memories, _ = play_game(
        EPSILON,
        MEMORY_SIZE,
        N_NEW_GAMES,
        N_OLD_GAMES,
        memories,
        np.empty((0, 4)),
        model,
        INPUT,
    )
    loss = train(GAMMA, memories, loss, model, INPUT, patience=100)

    print("Testing")
    _, points = play_game(
        1,
        MEMORY_SIZE,
        N_NEW_GAMES,
        0,
        np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE),
        points,
        model,
        INPUT,
    )

save(memories, points, loss, model, NAME)
plot(loss, points, NAME)

# Test
EPSILON = 1

memories = np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE)

print("Testing")
memories, points = play_game(
    EPSILON,
    MEMORY_SIZE,
    N_NEW_GAMES,
    0,
    memories,
    points,
    model,
    INPUT,
)
save(memories, points, loss, model, NAME, last=True)
plot(loss, points, NAME)
