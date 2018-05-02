#!/usr/bin/python

from libHanabi import game
from libInput import inputsource

Input = inputsource()
Game = game(Input)

Game.run()
