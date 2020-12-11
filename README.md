# Hanabi

This project fails to find optimal strategy for Hanabi using a neural network and a Deep-Q-Learning approach.

## Experiments

The notebooks in this repo highlight some problems in this implementation.

* LearningRate shows how high learning rate prevents convergence
* Inizialization shows how too big initialization values (i.e. greater than 1e-3 in absolute value) prevent convergence
* QUnlimited shows how, without a compensation inside the code, for high gamma (i.e. > 0.5) Q grows exponentially
* QFeedback shows how introducing a cutting for very high value predicted Q values saturate
* despite the name GradientDeath shows how this algorithm is unable to learn. In a previous version the output layer had dimension 1 and, if the last hidden layer was small and with relu activation, gradient saturated very fast


