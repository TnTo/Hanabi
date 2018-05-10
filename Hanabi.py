#!/usr/bin/python

from libHanabi import game
from libInput import inputsource
from libNN import NeuralNetwork

#Input = inputsource()
Input = NeuralNetwork()

Game = game(Input)

points = Game.run()

Input.update(points)
