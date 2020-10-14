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
from tensorflow.keras import Model, models
from typing import List
from random import random, choice
from dataclasses import dataclass
import numpy as np
from os import mkdir, path
import dill
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
            gamedata = game.save().toarray()
            Xs = np.array(
                list(
                    map(
                        lambda x: np.concatenate((gamedata, x.save().toarray())),
                        actions,
                    )
                )
            )
            Qs = self.model.predict(Xs)
            return actions[np.argmax(Qs)]

    def Q(self, game: GameData) -> float:
        actions = self.availableActionsData(game)
        gamedata = game.toarray()
        Xs = np.array(
            list(map(lambda x: np.concatenate((gamedata, x.toarray())), actions))
        )

        return game.score() + self.gamma * np.max(self.model.predict(Xs))

    def train(self, memories: List[Memory], n_epochs, shuffle_input):
        x = np.empty((0, self.model.input_shape[1]))
        y = np.empty((0, 1))
        for memory in memories:
            x = np.vstack(
                (x, np.concatenate((memory.pre.toarray(), memory.action.toarray())))
            )
            y = np.vstack((y, memory.post))
        y = np.vectorize(self.Q)(y)
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
                action = self.nn.move(game)
                game.move(game.players[0], action)
                self.memories.append(Memory(pre, action.save(), game.save()))
                game.next_turn()
        self.points.append([game.score() for game in self.games])
        print("Points:")
        print(self.points[-1])

    def train(self):
        self.loss.append(
            self.nn.train(self.memories, self.n_epochs, self.shuffle_input).history[
                "loss"
            ]
        )

    def run(self):
        while self.episode < self.n_episodes:
            print(f"EPISODE {self.episode}")
            self.create_episode()
            self.play_episode()
            self.train()
            if self.episode % 5 == 0:
                self.save_status()
        print(f"EPISODE {self.episode}")
        self.create_episode()
        self.play_episode()
        self.save_status()

    def save(self):
        try:
            mkdir(self.name)
        except FileExistsError:
            pass
        dill.dump(self.memories, open(path.join(self.name, "memories.dill"), "wb"))
        np.savetxt(path.join(self.name, "points.csv"), np.array(self.points))
        np.savetxt(path.join(self.name, "loss.csv"), np.array(self.loss))

        plt.plot([mean(episode) for episode in self.points])
        plt.title("Points " + self.name)
        plt.savefig(path.join(self.name, "points.pdf"))

        plt.close()

        plt.plot([episode[-1] for episode in self.loss])
        plt.title("Loss " + self.name)
        plt.savefig(path.join(self.name, "loss.pdf"))

    def save_status(self):
        self.nn.model.save(self.name + "_nn")
        dill.dump(
            {
                "memories": self.memories,
                "keep_memory": self.keep_memory,
                "n_games": self.n_games,
                "episode": self.episode,
                "n_episodes": self.n_episodes,
                "n_epochs": self.n_epochs,
                "shuffle_input": self.shuffle_input,
                "points": self.points,
                "loss": self.loss,
                "name": self.name,
                "epsilon": self.nn.epsilon,
                "gamma": self.nn.gamma,
            },
            open(self.name + ".dill", "wb"),
        )


def load_experiment(name: str):
    saved = dill.load(open(name + ".dill", "rb"))
    nn = NeuralNetwork(Model())
    nn.epsilon = saved["epsilon"]
    nn.gamma = saved["gamma"]
    nn.model = models.load_model(name + "_nn")
    ex = Experiment(
        nn,
        n_games=saved["n_games"],
        keep_memory=saved["keep_memory"],
        n_episodes=saved["n_episodes"],
        n_epochs=saved["n_epochs"],
        shuffle_input=saved["shuffle_input"],
        name=saved["name"],
    )
    ex.memories = saved["memories"]
    ex.episode = saved["episode"]
    ex.points = saved["points"]
    ex.loss = saved["loss"]
    return ex
