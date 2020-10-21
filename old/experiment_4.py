# Single Dumb NN

from Experiment import NeuralNetwork, Experiment, load_experiment
import tensorflow.keras as keras
import numpy as np
import os.path as path

import cProfile
import pstats

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

name = "exp4"


class NeuralNetwork4(NeuralNetwork):
    def train(self, memories, n_epochs, shuffle_input):
        x = np.empty((0, self.model.input_shape[1]))
        y = np.empty((0, 1))
        for memory in memories:
            x = np.vstack(
                (x, np.concatenate((memory.pre.toarray(), memory.action.toarray())))
            )
            y = np.vstack((y, memory.post))
        y = np.vectorize(self.Q)(y)
        return self.model.fit(
            x=x,
            y=y,
            epochs=n_epochs,
            shuffle=shuffle_input,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="loss", min_delta=0.05, patience=5
                )
            ],
        )


class Experiment4(Experiment):
    def update_nn(self):
        self.nn.epsilon = 0.95
        self.nn.gamma = 0.95


if path.exists(name + ".dill"):
    ex = load_experiment(name, NN=NeuralNetwork4, Exp=Experiment4)
else:
    ex = Experiment4(
        NeuralNetwork4(model),
        name=name,
        n_games=1000,
        n_episodes=50,
        keep_memory=False,
        n_epochs=500,
    )

ex.run()
ex.save()

profile.disable()
with open(name + "_profile.txt", "w") as f:
    ps = pstats.Stats(profile, stream=f)
    ps.strip_dirs()
    ps.sort_stats("cumtime")
    ps.print_stats()
