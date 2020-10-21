# Single Dumb NN

import cProfile
import os.path as path
import pstats

import tensorflow.keras as keras

from Experiment import Experiment, NeuralNetwork, load_experiment

profile = cProfile.Profile()
profile.enable()

# Create model
inputs = keras.Input(shape=(292,))
hidden1 = keras.layers.Dense(1000, activation="softplus")(inputs)
hidden2 = keras.layers.Dense(1000, activation="softplus")(hidden1)
hidden3 = keras.layers.Dense(500, activation="softplus")(hidden2)
Q = keras.layers.Dense(1)(hidden3)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()

name = "exp7"

if path.exists(name + ".dill"):
    ex = load_experiment(name)
else:
    ex = Experiment(
        NeuralNetwork(model, epsilon=0.5),
        name=name,
        n_games=50,
        n_episodes=50,
        keep_memory=False,
        n_epochs=1000,
    )

ex.run()

ex.nn.epsilon = 1
ex.run()

ex.save()

profile.disable()
with open(name + "_profile.txt", "w") as f:
    ps = pstats.Stats(profile, stream=f)
    ps.strip_dirs()
    ps.sort_stats("cumtime")
    ps.print_stats()
