from Hanabi import (
    Game,
    Action,
    Hint,
    Discard,
    Play,
    Number,
    Color,
    InputSource,
    GameData,
    ActionData,
    N_PLAYERS,
)
from keras import Model
from typing import List
from random import random, choice
from dataclasses import dataclass
from pprint import pprint
import numpy as np
from os import mkdir
import pickle
import matplotlib.pyplot as plt
from statistics import mean


@dataclass(frozen=True)
class Memory:
    pre: GameData
    action: ActionData
    post: GameData


class NeuralNetwork(InputSource):
    def __init__(self, model: Model):
        self.model = model
        self.epsilon = 0.0
        self.gamma = 0.0
        self.model.compile(loss="mse", optimizer="rmsprop")

    def availableActions(self, game: Game) -> List[Action]:
        actions: List[Action] = []
        if Play in game.availableActions():
            actions += [Play(tile) for tile in game.players[0].hand]
        if Discard in game.availableActions():
            actions += [Discard(tile) for tile in game.players[0].hand]
        if Hint in game.availableActions():
            actions += [
                Hint(player, feature)
                for player in game.players[1:]
                for feature in Number
            ]
            actions += [
                Hint(player, feature)
                for player in game.players[1:]
                for feature in Color
            ]
        return actions

    def availableActionsData(self, game: GameData) -> List[ActionData]:
        actions: List[ActionData] = []
        if Play in game.availableActions():
            actions += [ActionData(played=tile) for tile in game.players[0].hand]
        if Discard in game.availableActions():
            actions += [ActionData(discarded=tile) for tile in game.players[0].hand]
        if Hint in game.availableActions():
            actions += [
                ActionData(player=player, hintNumber=feature)
                for player in range(1, N_PLAYERS)
                for feature in Number
            ]
            actions += [
                ActionData(player=player, hintColor=feature)
                for player in range(1, N_PLAYERS)
                for feature in Color
            ]
        return actions

    def evaluateAction(self, game: GameData, action: ActionData) -> float:
        return self.model.predict(
            np.concatenate((game.toarray(), action.toarray())).reshape(1, -1)
        )

    def move(self, game: Game) -> Action:
        actions = self.availableActions(game)
        if len(actions) == 0:
            raise NameError("No Actions Available")
        if self.epsilon < random():
            return choice(actions)
        else:
            max_value = -10e30
            gamedata = game.save()
            for action in actions:
                if (value := self.evaluateAction(gamedata, action.save())) > max_value:
                    max_action = action
                    max_value = value
            return max_action

    def Q(self, game: GameData) -> float:
        return game.score() + self.gamma * max(
            [
                self.evaluateAction(game, action)
                for action in self.availableActionsData(game)
            ]
        )

    def train(self, memories: List[Memory], n_epochs, shuffle_input):
        # convert to numpy
        x = np.empty((0, self.model.input_shape[1]))
        y = np.empty((0, 1))
        for memory in memories:
            x = np.vstack(
                (x, np.concatenate((memory.pre.toarray(), memory.action.toarray())))
            )
            y = np.vstack((y, np.array(self.Q(memory.post))))
        return self.model.fit(x=x, y=y, epochs=n_epochs, shuffle=shuffle_input)


class Experiment:
    def __init__(
        self,
        nn: NeuralNetwork,
        n_games: int = 10,
        keep_memory: bool = False,
        n_episodes: int = 10,
        n_epochs: int = 25,
        shuffle_input: bool = True,
        name: str = "hanabi",
    ):
        self.nn = nn
        self.memories: List[Memory] = []
        self.keep_memory = keep_memory
        self.games: List[Game]
        self.n_games = n_games
        self.episode = 0
        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.shuffle_input = shuffle_input
        self.points: List[List[int]] = []
        self.loss: List = []
        self.name = name

    def update_nn(self):
        if self.episode >= self.n_episodes / 2:
            self.nn.epsilon = 0.95
        elif self.episode >= self.n_episodes / 4:
            self.nn.epsilon = 0.5
        if self.episode > 1:
            self.nn.gamma = 0.95

    def create_episode(self):
        self.episode += 1
        if not self.keep_memory:
            self.memories.clear()
        self.update_nn()
        self.games = [Game(self.nn) for _ in range(self.n_games)]

    def play_episode(self):
        for game in self.games:
            while not game.ended:
                pre = game.save()
                game.move(game.players[0], action := self.nn.move(game))
                self.memories.append(Memory(pre, action.save(), game.save()))
                game.next_turn()
        self.points.append([game.score() for game in self.games])
        print("Points:")
        pprint(self.points[-1])

    def train(self):
        self.loss.append(
            self.nn.train(self.memories, self.n_epochs, self.shuffle_input).history[
                "loss"
            ]
        )

    def run(self):
        for i in range(self.n_episodes):
            print(f"EPISODE {i}")
            self.create_episode()
            self.play_episode()
            self.train()
        print(f"EPISODE {self.n_episodes}")
        self.create_episode()
        self.play_episode()

    def save(self):
        mkdir(self.name)
        pickle.dump(self.memories, open(self.name + "/memories.pickle", "wb"))
        np.savetxt(self.name + "/points.csv", np.array(self.points))
        np.savetxt(self.name + "/loss.csv", np.array(self.loss))

        plt.plot([mean(episode) for episode in self.points])
        plt.title("Points " + self.name)
        plt.savefig(self.name + "/points.pdf")
        
        plt.close()

        plt.plot([episode[-1] for episode in self.loss])
        plt.title("Loss " + self.name)
        plt.savefig(self.name + "/loss.pdf")
