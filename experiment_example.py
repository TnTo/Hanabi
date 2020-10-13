# Single Dumb NN

from Experiment import NeuralNetwork, Experiment
import tensorflow.keras as keras
import os.path as path
import pickle


# Create model
inputs = keras.Input(shape=(292,))
hidden = keras.layers.Dense(1, activation="relu")(inputs)
Q = keras.layers.Dense(1)(hidden)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()

name = 'hanabi'

if path.exists(name+'.pickle'):
    ex = pickle.load(open(name+'.pickle', 'rb'))
else:
    ex = Experiment(NeuralNetwork(model), name=name, n_games=3, n_episodes=2)

ex.run()
ex.save()
