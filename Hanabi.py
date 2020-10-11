from random import shuffle, choice
from enum import Enum, IntEnum
from typing import List, Union, Type, Dict
from abc import ABC
from copy import deepcopy

N_PLAYERS: int = 5


class InputSource(ABC):
    def move(self, game: 'Game', activePlayer: 'Player', **kwargs) -> 'Action':
        return Action()


def handSize(n_players: int) -> int:
    if n_players in [2, 3]:
        return 5
    elif n_players in [4, 5]:
        return 4
    else:
        raise NameError('InvalidNPlayers')


class Color(Enum):
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


Feature = Union[Color, Number]


class Tile:
    def __init__(self, number: Number, color: Color):
        self.color = color
        self.number = number


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


class DummyDeck(Deck):
    def __init__(self):
        self.deck: List[Tile] = []


class Info(Enum):
    YES = 1
    UNKNOWN = 0
    NO = -1


class TileInHand (Tile):
    def __init__(self, tile: Tile):
        self.tile = tile
        self.numberInfo: Dict[Number, Info] = {}
        for number in Number:
            self.numberInfo[number] = Info.UNKNOWN
        self.colorInfo: Dict[Color, Info] = {}
        for color in Color:
            self.colorInfo[color] = Info.UNKNOWN


class Action(ABC):
    pass


class Discard(Action):
    def __init__(self, discarded: Tile):
        self.tile = discarded


class Play(Action):
    def __init__(self, played: Tile):
        self.tile = played


class Hint(Action):
    def __init__(self, player: 'Player', feature: Feature):
        self.player = player
        self.feature: Union[Color, Number] = feature


class Player:
    def __init__(self, deck: Deck, inputSource: InputSource):
        self.inputSource = inputSource
        self.hand: List[TileInHand] = []
        for _ in range(handSize(N_PLAYERS)):
            self.draw(deck)

    def draw(self, deck: Deck):
        if deck.numberOfTiles() > 0:
            self.hand.append(TileInHand(deck.draw()))

    def getHint(self, hint: Hint):
        if isinstance(hint.feature, Color):
            for tile in self.hand:
                if tile.tile.color == hint.feature:
                    tile.colorInfo[hint.feature] = Info.YES
                else:
                    tile.colorInfo[hint.feature] = Info.NO
        elif isinstance(hint.feature, Number):
            for tile in self.hand:
                if tile.tile.number == hint.feature:
                    tile.numberInfo[hint.feature] = Info.YES
                else:
                    tile.numberInfo[hint.feature] = Info.NO

    def discardTile(self, discard: Union[Discard, Play], deck: Deck):
        for i in range(len(self.hand)):
            if self.hand[i].tile is discard.tile:
                self.hand.pop(i)
                self.draw(deck)
                return

    def copy(self):
        player = Player(DummyDeck(), self.inputSource)
        player.hand = deepcopy(self.hand)
        return player


class Game:
    def __init__(self, inputSources: Union[InputSource, List[InputSource]]):
        self.lifes = 3
        self.hints = 8
        self.deck = Deck()
        self.discardedTiles: List[Tile] = []
        self.players: List[Player] = []
        self.piles: Dict[Color, Union[Number, None]] = {color: None for color in Color}
        self.lastRound = False
        self.ended = False
        if isinstance(inputSources, InputSource):
            self.players = [Player(self.deck, inputSources) for _ in range(N_PLAYERS)]
        elif (isinstance(inputSources, List) and (len(inputSources) == 1)):
            self.players = [Player(self.deck, inputSources[0]) for _ in range(N_PLAYERS)]
        elif len(inputSources) == N_PLAYERS:
            self.players = [Player(self.deck, inputSource) for inputSource in inputSources]
        elif len(inputSources) == 0:
            pass
        else:
            self.players = [Player(self.deck, choice(inputSources)) for _ in range(N_PLAYERS)]

    def availableActions(self) -> List[Type[Action]]:
        if self.hints == 0:
            return [Discard, Play]
        elif self.hints == 8:
            return [Hint, Play]
        else:
            return [Hint, Discard, Play]

    def move(self, player: Player, action: Action):
        if isinstance(action, Hint):
            if self.hints == 0:
                raise NameError('InvalidMove')
            else:
                self.hints -= 1
                action.player.getHint(action)
        elif isinstance(action, Discard):
            if self.hints == 8:
                raise NameError('InvalidMove')
            else:
                self.hints += 1
                player.discardTile(action, self.deck)
                self.discardedTiles.append(action.tile)
        elif isinstance(action, Play):
            player.discardTile(action, self.deck)
            if action.tile.number == 1:
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
                if action.tile.number == Number.FIVE:
                    end = True
                    for color in Color:
                        if self.piles[color] != Number.FIVE:
                            end = False
                    if end:
                        self.ended = True
            if self.lifes == 0:
                self.ended = True
            if self.deck.numberOfTiles() == 0:
                self.lastRound = True

    def score(self) -> int:
        score: int = 0
        for color in self.piles:
            score += int(self.piles[color] or 0)
        return score

    def copy(self):
        game = Game([])
        game.players = self.players.copy()
        game.lifes = deepcopy(self.lifes)
        game.hints = deepcopy(self.hints)
        game.deck = deepcopy(self.deck)
        game.discardedTiles = deepcopy(self.discardedTiles)
        game.piles = deepcopy(self.piles)
        game.lastRound = deepcopy(self.lastRound)
        game.ended = deepcopy(self.ended)
        return game
