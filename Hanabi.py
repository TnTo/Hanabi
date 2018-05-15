#!/usr/bin/python

from libHanabi import game
from libInput import inputsource
from libNN import NeuralNetwork

import matplotlib.pyplot as plt
from numpy import polyfit

#Input = inputsource()
Input = NeuralNetwork()

InputList = [Input, Input, Input, Input]

pointsmemory = []
movesmemory = []

NIter = 5000

for i in range (0, NIter):
    print("Match number " + str(i + 1) + " of " + str(NIter))

    Input.resetmovesmemory()

    Game = game(InputList)

    points = Game.run()

    pointsmemory.append(points)

    movesmemory.append(Input.getnumberofmoves())

    Input.update(points)

plt.scatter(range(0, NIter), pointsmemory)
fit = polyfit (range(0, NIter), pointsmemory, deg=1)
plt.plot(range(0, NIter), fit[0] * range(0, NIter) + fit[1], color='red')
plt.figure()

plt.scatter(range(0, NIter), movesmemory)
fit = polyfit (range(0, NIter), movesmemory, deg=1)
plt.plot(range(0, NIter), fit[0] * range(0, NIter) + fit[1], color='green')

plt.show()
