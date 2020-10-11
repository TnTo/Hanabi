from Hanabi import Game, Action, Hint, Discard, Play, Player, Color, Number, InputSource, N_PLAYERS
from keras import Model
from typing import NamedTuple, List, Union, Tuple, Callable, Dict, Any
import numpy as np
from random import random, choice, choices
from operator import itemgetter
from itertools import cycle

N_NETWORKS: int = 1
N_PARALLEL_GAMES: int = 1
MIRROR_PLAY: bool = True
LEARN_FROM_PREVIOUS_EPISODES: bool = False
LEARN_FROM_OTHERS_GAMES: bool = False
LEARN_FROM_OTHERS_PLAYERS: bool = False
N_EPOCHS: int = 25
SHUFFLE_INPUT: bool = False


class Memory(NamedTuple):
    episode: int
    gameId: int
    gameStatus: Game
    player: Player
    turn: int
    action: Action
    score: int
    Q: float


class NeuralNetwork(InputSource):
    def __init__(self, model: Model,
                 epsilon: Callable[[int], float], gamma: float,
                 encode: Callable[[Game, Action, Player], Dict[str, np.array]]):
        self.model = model
        self.encode = encode
        self.epsilon = epsilon
        self.gamma = gamma
        self.model.compile(optimizer="rmsprop", metrics='mse')

    def move(self, game: Game, activePlayer: Player, episode: int = 0, **kwargs) -> Action:
        actions = self.evaluateMoves(game, activePlayer)
        if self.epsilon(episode) < random():
            return choice(actions)[0]
        else:
            return max(actions, key=itemgetter(1))[0]

    def evaluateMoves(self, game: Game, activePlayer: Player) -> List[Tuple[Action, float]]:
        availableActions = game.availableActions()
        actions: List[Tuple[Action, float]] = []
        if Hint in availableActions:
            for player in game.players:
                if player is not activePlayer:
                    for color in Color:
                        actions.append((Hint(player, color), self.model.predict(self.encode(game, Hint(player, color), activePlayer))[0][0]))
                    for number in Number:
                        actions.append((Hint(player, number), self.model.predict(self.encode(game, Hint(player, number), activePlayer))[0][0]))
        if Discard in availableActions:
            for tile in activePlayer.hand:
                actions.append((Discard(tile.tile), self.model.predict(self.encode(game, Discard(tile.tile), activePlayer))[0][0]))
        if Play in availableActions:
            for tile in activePlayer.hand:
                actions.append((Play(tile.tile), self.model.predict(self.encode(game, Play(tile.tile), activePlayer))[0][0]))
        return actions

    def train(self, memories: List[Memory]):
        x = []
        y = []
        if isinstance(self.encode, Callable[Any, Dict[str, Any]]):
            pass
        for memory in memories:
            x.append(self.encode(memory.gameStatus, memory.action, memory.player))
            y.append(memory.Q)
        self.model.fit(x=x, y=y, epochs=N_EPOCHS, shuffle=SHUFFLE_INPUT)


class Experiment:
    def __init__(self, inputSources: Union[NeuralNetwork, List[NeuralNetwork]]):
        self.memory: List[Memory] = []
        self.inputSources = inputSources
        self.episode: int = 0

    def nextMove(self, game: Game, players: cycle, gameId: int, turn: int):
        initialState: Game = game.copy()
        activePlayer: Player = next(players)
        action = activePlayer.inputSource.move(game, activePlayer, episode=self.episode)
        game.move(activePlayer, action)
        memory: Memory = Memory(
            self.episode,
            gameId,
            initialState,
            activePlayer,
            turn,
            action,
            game.score(),
            game.score() + activePlayer.inputSource.gamma * max(activePlayer.inputSource.evaluateMoves(game, activePlayer), key=itemgetter(1))[1]
        )
        self.memory.append(memory)

    def playGame(self, gameId: int, inputSources: Union[NeuralNetwork, List[NeuralNetwork]]) -> int:
        game = Game(inputSources)
        print('Game started')
        players = cycle(game.players)
        turn: int = 0
        while not game.ended:
            turn += 1
            print('Turn ', turn)
            self.nextMove(game, players, gameId, turn)
            if game.lastRound:
                for _ in range(N_PLAYERS):
                    turn += 1
                    print('Turn ', turn)
                    self.nextMove(game, players, gameId, turn)
                break
        return game.score()

    def play(self):
        for gameId in range(N_PARALLEL_GAMES):
            if isinstance(self.inputSources, NeuralNetwork):
                score = self.playGame(gameId, self.inputSources)
            elif MIRROR_PLAY:
                score = self.playGame(gameId, choice(self.inputSources))
            else:
                score = self.playGame(gameId, choices(self.inputSources, N_PLAYERS))
        print("GAME ", gameId, "ended with ", score, " points")

    def train(self):
        if isinstance(self.inputSources, NeuralNetwork):
            self.inputSources.train(self.memory)
        else:
            for NN in self.inputSources:
                memory = self.memory
                if not LEARN_FROM_OTHERS_GAMES:
                    memory = [m for m in memory if NN in m.gameStatus.players]
                    if not LEARN_FROM_OTHERS_PLAYERS:
                        memory = [m for m in memory if NN == m.player]
                NN.train(memory)

    def run(self, episodes: int):
        for episode in range(episodes):
            self.episode = episode
            print("EPISODE: ", episode)
            if not LEARN_FROM_PREVIOUS_EPISODES:
                self.memory.clear()
            self.play()
            self.train()
