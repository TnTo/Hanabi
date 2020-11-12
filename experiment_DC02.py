import os
from typing import List
import random

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

from GameDummy import DGAME, DACTION, DTYPE, play_game, train, save, plot, vscore
from GameDummy import (
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

NAME: str = "exDC02"

SEED = 123456
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


INPUT: List[int] = (
    LIFES + HINTS + list(DISCARDED) + list(PLAYERS) + list(PILES) + LAST_ROUND
)

if os.path.exists(NAME):
    model = keras.models.load_model(os.path.join(NAME, "nn"))
    memories = np.load(os.path.join(NAME, "memories.npy"))
    points = np.load(os.path.join(NAME, "points.npy"))
    loss = np.load(os.path.join(NAME, "loss.npy"))
else:
    inputs = keras.Input(shape=(len(INPUT),))
    hidden1 = keras.layers.Dense(1, activation="softplus")(inputs)
    Q = keras.layers.Dense(1, activation="linear")(hidden1)
    model = keras.Model(inputs=inputs, outputs=Q)
    model.summary()
    model.compile(
        loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
    )

    memories = np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE)
    points = np.empty((0, 4))
    loss = np.empty((0, 1))


# CREATE MEMORIES DB
MEMORY_SIZE: int = 10 ** 3
N_NEW_GAMES: int = 500
N_OLD_GAMES: int = 1000
EPSILON: float = 0
GAMMA: float = 0

i = 1
while memories.shape[0] < MEMORY_SIZE:
    print("Create memories, round ", i)
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
    i += 1

for j in range(i*2):
    print("Improve memories, round ", j + 1, " of ", i*2)
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

plt.hist(vscore(memories[:, -DGAME:]))
plt.show()

save(memories, points, loss, model, NAME, first=True)

# plt.hist(vscore(memories[:, -DGAME:]))
# plt.savefig(os.path.join(NAME, "score_memories_t0.png"))

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

loss = train(0, memories, loss, model, INPUT, patience=2)
loss = train(GAMMA, memories, loss, model, INPUT, patience=5)

save(memories, points, loss, model, NAME, first=True)

# Reset
points = np.empty((0, 4))
loss = np.empty((0, 1))

# Test
EPSILON = 1
N_OLD_GAMES = 0

print("Testing")
_, points = play_game(
    EPSILON,
    MEMORY_SIZE,
    N_NEW_GAMES,
    N_OLD_GAMES,
    np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE),
    points,
    model,
    INPUT,
)

plt.hist(points)
plt.savefig(os.path.join(NAME, "points_t0.png"))

# Explore
MEMORY_SIZE = 10 ** 5
N_EPISODES: int = 20
EPSILON = 0.5
N_OLD_GAMES = 1000

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
    loss = train(
        GAMMA,
        memories,
        loss,
        model,
        INPUT,
    )

    print("Testing")
    _, points = play_game(
        EPSILON,
        MEMORY_SIZE,
        N_NEW_GAMES,
        0,
        np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE),
        points,
        model,
        INPUT,
    )

plot(loss, points, NAME)
save(memories, points, loss, model, NAME)

# Exploit
N_EPISODES = 20
EPSILON = 0.90

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
    loss = train(
        GAMMA,
        memories,
        loss,
        model,
        INPUT,
    )

    print("Testing")
    _, points = play_game(
        EPSILON,
        MEMORY_SIZE,
        N_NEW_GAMES,
        0,
        np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE),
        points,
        model,
        INPUT,
    )

plot(loss, points, NAME)
save(memories, points, loss, model, NAME)

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
