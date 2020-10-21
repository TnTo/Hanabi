from random import shuffle
from enum import IntEnum
from typing import List, Union, Type, Dict
from abc import ABC
from dataclasses import dataclass
import numpy as np

N_PLAYERS: int = 5


class InputSource(ABC):
    def move(self, game: "Game") -> "Action":
        return Action()


def handSize(n_players: int) -> int:
    if n_players in [2, 3]:
        return 5
    elif n_players in [4, 5]:
        return 4
    else:
        raise NameError("InvalidNPlayers")


class Color(IntEnum):
    BLUE = 1
    GREEN = 2
    RED = 3
    WHITE = 4
    YELLOW = 5


class Number(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class Info(IntEnum):
    YES = 1
    UNKNOWN = 0
    NO = -1


Feature = Union[Color, Number]


class Data(ABC):
    def toarray(self) -> List:
        pass


@dataclass(frozen=True)
class TileData(Data):
    color: Color
    number: Number
    isBlue: Info
    isGreen: Info
    isRed: Info
    isWhite: Info
    isYellow: Info
    isOne: Info
    isTwo: Info
    isThree: Info
    isFour: Info
    isFive: Info

    def toarray(self):
        return np.array(
            [
                int(self.color),
                int(self.number),
                int(self.isBlue),
                int(self.isGreen),
                int(self.isRed),
                int(self.isWhite),
                int(self.isYellow),
                int(self.isOne),
                int(self.isTwo),
                int(self.isThree),
                int(self.isFour),
                int(self.isFive),
            ]
        )


class Tile:
    def __init__(self, number: Number, color: Color):
        self.color: Color = color
        self.number: Number = number
        self.infos: Dict[Feature, Info] = {}
        for color in Color:
            self.infos[color] = Info.UNKNOWN
        for number in Number:
            self.infos[number] = Info.UNKNOWN

    def receive_hint(self, hint: "Hint"):
        if isinstance(hint.feature, Number):
            if self.number == hint.feature:
                for number in Number:
                    self.infos[number] = Info.NO
                self.infos[hint.feature] = Info.YES
        if isinstance(hint.feature, Color):
            if self.color == hint.feature:
                for color in Color:
                    self.infos[color] = Info.NO
                self.infos[hint.feature] = Info.YES

    def save(self):
        return TileData(
            color=self.color,
            number=self.number,
            isBlue=self.infos[Color.BLUE],
            isGreen=self.infos[Color.GREEN],
            isRed=self.infos[Color.RED],
            isWhite=self.infos[Color.WHITE],
            isYellow=self.infos[Color.YELLOW],
            isOne=self.infos[Number.ONE],
            isTwo=self.infos[Number.TWO],
            isThree=self.infos[Number.THREE],
            isFour=self.infos[Number.FOUR],
            isFive=self.infos[Number.FIVE],
        )


class Deck:
    def __init__(self):
        self.deck: List[Tile] = []
        for color in Color:
            for number in Number:
                if number == 1:
                    self.deck.append(Tile(number, color))
                    self.deck.append(Tile(number, color))
                    self.deck.append(Tile(number, color))
                elif number in [2, 3, 4]:
                    self.deck.append(Tile(number, color))
                    self.deck.append(Tile(number, color))
                elif number == 5:
                    self.deck.append(Tile(number, color))
        shuffle(self.deck)

    def draw(self) -> Tile:
        return self.deck.pop()

    def numberOfTiles(self) -> int:
        return len(self.deck)


@dataclass(frozen=True)
class ActionData(Data):
    hintColor: Union[Color, None] = None
    hintNumber: Union[Number, None] = None
    player: int = 0
    played: Union[TileData, None] = None
    discarded: Union[TileData, None] = None

    def toarray(self):
        arr = np.array(
            [int(self.hintColor or 0), int(self.hintNumber or 0), self.player]
        )
        if self.played is None:
            arr = np.concatenate((arr, np.zeros(12)))
        else:
            arr = np.concatenate((arr, self.played.toarray()))
        if self.discarded is None:
            arr = np.concatenate((arr, np.zeros(12)))
        else:
            arr = np.concatenate((arr, self.discarded.toarray()))
        return arr


class Action(ABC):
    def save(self, **kwargs) -> ActionData:
        pass


class Discard(Action):
    def __init__(self, discarded: Tile):
        self.tile = discarded

    def save(self, **kwargs) -> ActionData:
        return ActionData(discarded=self.tile.save())


class Play(Action):
    def __init__(self, played: Tile):
        self.tile = played

    def save(self, **kwargs) -> ActionData:
        return ActionData(played=self.tile.save())


class Hint(Action):
    def __init__(self, player: "Player", feature: Feature):
        self.player = player
        self.feature: Feature = feature

    def save(self, player=0, **kwargs) -> ActionData:
        return ActionData(
            hintColor=self.feature if isinstance(self.feature, Color) else None,
            hintNumber=self.feature if isinstance(self.feature, Number) else None,
            player=player,
        )


@dataclass(frozen=True)
class PlayerData(Data):
    hand: List[TileData]

    def toarray(self, hide=False):
        tiles = []
        if hide:
            tiles = [tile.toarray()[2:] for tile in self.hand] + [
                [0] * 10 for _ in range(handSize(N_PLAYERS) - len(self.hand))
            ]
        else:
            tiles = [tile.toarray() for tile in self.hand] + [
                [0] * 12 for _ in range(handSize(N_PLAYERS) - len(self.hand))
            ]
        return np.concatenate(tiles)


class Player:
    def __init__(self, deck: Deck, inputSource: InputSource):
        self.inputSource = inputSource
        self.hand: List[Tile] = []
        self.deck = deck
        for _ in range(handSize(N_PLAYERS)):
            self.draw()

    def draw(self):
        if self.deck.numberOfTiles() > 0:
            self.hand.append(self.deck.draw())

    def getHint(self, hint: Hint):
        for tile in self.hand:
            tile.receive_hint(hint)

    def discardTile(self, discard: Union[Discard, Play]):
        self.hand.remove(discard.tile)
        self.draw()

    def save(self):
        return PlayerData(hand=[tile.save() for tile in self.hand])


@dataclass(frozen=True)
class GameData(Data):
    turn: int
    lifes: int
    hints: int
    ended: bool
    piles: Dict[Color, Union[Number, None]]
    discarded: Dict[Color, Dict[Number, int]]
    players: List[PlayerData]

    def toarray(self):
        return np.concatenate(
            (
                np.array([self.turn, self.lifes, self.hints]),
                np.array([int(self.piles[color] or 0) for color in Color]),
                np.array(
                    [
                        self.discarded[color][number]
                        for color in Color
                        for number in Number
                    ]
                ),
                self.players[0].toarray(hide=True),
                np.concatenate([player.toarray() for player in self.players[1:]]),
            ),
            axis=0,
        )

    def score(self) -> int:
        return sum([int(self.piles[color] or 0) for color in Color])

    def availableActions(self) -> List[Type[Action]]:
        if self.hints == 0:
            return [Discard, Play]
        elif self.hints == 8:
            return [Hint, Play]
        else:
            return [Hint, Discard, Play]


class Game:
    def __init__(
        self, inputSources: InputSource
    ):  # from Union[InputSource, List[InputSource]] to InputSource
        self.lifes = 3
        self.hints = 8
        self.deck = Deck()
        self.discardedTiles: List[Tile] = []
        self.players: List[Player] = []
        self.piles: Dict[Color, Union[Number, None]] = {color: None for color in Color}
        self.lastRound: Union[int, None] = None
        self.ended = False
        self.players = [Player(self.deck, inputSources) for _ in range(N_PLAYERS)]
        self.turn = 1

    def availableActions(self) -> List[Type[Action]]:
        if self.hints == 0:
            return [Discard, Play]
        elif self.hints == 8:
            return [Hint, Play]
        else:
            return [Hint, Discard, Play]

    def move(self, player: Player, action: Action):
        if not type(action) in self.availableActions():
            raise NameError("InvalidMove")
        index = None
        if isinstance(action, Hint):
            if action.player == self.players[0]:
                raise NameError("InvalidMove")
            self.hints -= 1
            action.player.getHint(action)
            index = self.players.index(action.player)
        if isinstance(action, Discard):
            self.hints += 1
            self.discardedTiles.append(action.tile)
        if isinstance(action, Play):
            if action.tile.number == Number.ONE:
                if self.piles[action.tile.color] is None:
                    self.piles[action.tile.color] = action.tile.number
                else:
                    self.lifes -= 1
                    self.discardedTiles.append(action.tile)
            else:
                if self.piles[action.tile.color] == (action.tile.number - 1):
                    self.piles[action.tile.color] = action.tile.number
                else:
                    self.lifes -= 1
                    self.discardedTiles.append(action.tile)
        return action.save(player=index)

    def is_ended(self) -> bool:
        if set([self.piles[color] for color in Color]) == set([Number.FIVE]):
            self.ended = True
        if self.lifes == 0:
            self.ended = True
        if self.deck.numberOfTiles() == 0:
            if self.lastRound is None:
                self.lastRound = N_PLAYERS
            else:
                self.lastRound -= 1
        if self.lastRound == 0:
            self.ended = True
        return self.ended

    def next_player(self):
        self.turn += 1
        self.players.append(self.players[0])
        self.players.pop(0)

    def score(self) -> int:
        score: int = 0
        for color in self.piles:
            score += int(self.piles[color] or 0)
        return score

    def count_discarded(self):
        count = {color: {number: 0 for number in Number} for color in Color}
        for tile in self.discardedTiles:
            count[tile.color][tile.number] += 1
        return count

    def save(self):
        return GameData(
            turn=self.turn,
            lifes=self.lifes,
            hints=self.hints,
            ended=self.ended,
            piles={color: self.piles[color] for color in Color},
            discarded=self.count_discarded(),
            players=[player.save() for player in self.players],
        )
