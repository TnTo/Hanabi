# Single Dumb NN

from Experiment import NeuralNetwork, Experiment
import tensorflow.keras as keras


# Create model
inputs = keras.Input(shape=(292,))
hidden = keras.layers.Dense(1, activation="relu")(inputs)
Q = keras.layers.Dense(1)(hidden)
model = keras.Model(inputs=inputs, outputs=Q)
model.summary()


ex = Experiment(NeuralNetwork(model), n_episodes=2, n_games=3)

ex.run()
ex.save()
