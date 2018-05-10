# Hanabi

This project aim to find optimal strategy for Hanabi using a neural network

### Error Code

* 0 No error
* 1 Generic
* 2 Parameter out of range
* 3 No more element to pick
* 4 You should have already lost the game

###End Game Codes
* 0 No Problem
* 5 Game End
* 6 Invalid move
* 7 Last Round

### Input

The input function must return a list, in particular

* Discard [0, index] e.g. to discard the first tile [0,0]
* Play [1, index] e.g to play the third tile [1, 2]
* Hint [2, hint, player_name] e.g. [2, "Yellow", "North"]
