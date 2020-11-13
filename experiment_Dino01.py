import os
from typing import List
import random

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

import matplotlib.pyplot as plt

from Game import (
    DGAME,
    DACTION,
    DTYPE,
    play_game,
    train,
    save,
    plot,
    vscore
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

NAME: str = "Dino01quater"

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

if os.path.exists(NAME):
    model = keras.models.load_model(os.path.join(NAME, "nn"))
    memories = np.load(os.path.join(NAME, "memories.npy"))
    points = np.load(os.path.join(NAME, "points.npy"))
    loss = np.load(os.path.join(NAME, "loss.npy"))
else:
    inputs = keras.Input(shape=(len(INPUT),))
    hidden1 = keras.layers.Dense(64, activation="sigmoid")(inputs)
    drop1 = keras.layers.Dropout(0.3, seed=SEED)(hidden1)
    hidden2 = keras.layers.Dense(64, activation="softplus")(drop1)
    drop2 = keras.layers.Dropout(0.3, seed=SEED)(hidden2)
    hidden3 = keras.layers.Dense(32, activation="softplus")(drop2)
    Q = keras.layers.Dense(1, activation="linear")(hidden3)
    model = keras.Model(inputs=inputs, outputs=Q)
    model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=0.01))
    model.summary()
    for layer in model.layers:
        print(layer.get_config())

    memories = np.empty((0, DGAME + DACTION + DGAME), dtype=DTYPE)
    points = np.empty((0, 4))
    loss = np.empty((0, 1))


# CREATE MEMORIES DB
MEMORY_SIZE: int = 10 ** 4
N_NEW_GAMES: int = 50
N_OLD_GAMES: int = 100
EPSILON: float = 0
GAMMA: float = 0.95

i = 0
while memories.shape[0] < MEMORY_SIZE:
    i += 1
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

for j in range(int(i/4)):
    print("Improve memories, round ", j + 1, " of ", i)
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

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

train(0, memories, loss, model, INPUT, name=NAME)

plt.hist(model.predict(memories[:, INPUT], batch_size=512))
plt.show()

for _ in range(int(1/GAMMA/4)):
    train(GAMMA, memories, loss, model, INPUT, name=NAME)

plt.hist(model.predict(memories[:, INPUT], batch_size=512))
plt.show()

save(memories, points, loss, model, NAME, first=True)

# Reset
points = np.empty((0, 4))
loss = np.empty((0, 1))

# Test
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

plt.hist(points)
plt.savefig(os.path.join(NAME, "points_t0.png"))

# Explore
MEMORY_SIZE = 10 ** 8
N_EPISODES: int = 20
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
    loss = train(GAMMA, memories, loss, model, INPUT, name=NAME)

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
    loss = train(GAMMA, memories, loss, model, INPUT, name=NAME)

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
