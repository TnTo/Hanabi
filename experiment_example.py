# Single Dumb NN

from Experiment import NeuralNetwork, Experiment, load_experiment
import tensorflow.keras as keras
import os.path as path

import cProfile
import pstats

profile = cProfile.Profile()
profile.enable()

# Create model
inputs = keras.Input(shape=(292,))
hidden = keras.layers.Dense(1, activation="relu")(inputs)
Q = keras.layers.Dense(1)(hidden)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()

name = "hanabi"

if path.exists(name + ".dill"):
    ex = load_experiment(name)
else:
    ex = Experiment(NeuralNetwork(model), name=name, n_games=3, n_episodes=2)

ex.run()
ex.save()

profile.disable()
with open(name + "_profile.txt", "w") as f:
    ps = pstats.Stats(profile, stream=f)
    ps.strip_dirs()
    ps.sort_stats("cumtime")
    ps.print_stats()
