# Single Dumb NN

from Experiment import NeuralNetwork, Experiment
import tensorflow.keras as keras

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


ex = Experiment(NeuralNetwork(model), name='exp1', n_games=20, n_episodes=50, n_epochs=25, keep_memory=True)

ex.run()
ex.save()

profile.disable()
with open("profile.txt", "wb") as f:
    ps = pstats.Stats(profile)
    ps.sort_stats('cumtime')
    ps.print_stats()