# Single Dumb NN

from Experiment import NeuralNetwork, Experiment, load_experiment
import tensorflow.keras as keras
import os.path as path
import dill

import cProfile
import pstats

profile = cProfile.Profile()
profile.enable()

# Create model
inputs = keras.Input(shape=(292,))
hidden1 = keras.layers.Dense(500, activation="relu")(inputs)
hidden2 = keras.layers.Dense(500, activation="relu")(hidden1)
Q = keras.layers.Dense(1)(hidden2)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()

name = "exp1"

if path.exists(name + ".dill"):
    ex = load_experiment(name)
else:
    ex = Experiment(
        NeuralNetwork(model),
        name=name,
        n_games=20,
        n_episodes=50,
        n_epochs=25,
        keep_memory=True,
    )

ex.run()
ex.save()

profile.disable()
with open(name + "_profile.txt", "w") as f:
    ps = pstats.Stats(profile, stream=f)
    ps.strip_dirs()
    ps.sort_stats("cumtime")
    ps.print_stats()
