from random import shuffle, sample
from enum import Enum, IntEnum
from typing import TypeVar, List, NamedTuple, Union, Type, Dict
from abc import ABC
from copy import deepcopy

N_PLAYERS: int

def handSize (n_players:int) -> int:
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

Feature = TypeVar('Feature', Color, Number)

class Tile:
    def __init__(self, number: Number, color: Color):
        self.color = color
        self.number = number

class Deck:
    def __init__(self):
        self.deck: List[Tile]
        for color in Color:
            for number in Number:
                if number == 1:
                    self.deck.append(Tile(number, color))
                    self.deck.append(Tile(number, color))
                    self.deck.append(Tile(number, color))
                elif number in [2,3,4]:
                    self.deck.append(Tile(number, color))
                    self.deck.append(Tile(number, color))
                elif number == 5:
                    self.deck.append(Tile(number, color))
        shuffle(self.deck)

    def draw(self) -> Tile:
        return self.deck.pop()

    def numberOfTiles(self) -> int:
        return len(self.deck)

class InputSource(ABC):
    def move(self, game:'Game') -> 'Action':
        pass

class Info(Enum):
    YES = 1
    UNKNOWN = 0
    NO = -1

class TileInHand (NamedTuple):
    tile: Tile
    numberInfo: Info
    colorInfo: Info

class Action(ABC):
    pass

class Discard(Action):
    def __init__(self, discarded:Tile):
        self.tile = discarded

class Play(Action):
    def __init__(self, played:Tile):
        self.tile = played

class Hint(Action):
    def __init__(self, player:'Player', feature:Feature):
        self.player = player
        self.feature = feature

class Player:
    def __init__(self, deck:Deck, inputSource:InputSource):
        self.inputSource = inputSource
        self.hand: List[TileInHand]
        for _ in range(handSize(N_PLAYERS)):
            self.draw(deck)

    def draw(self, deck:Deck):
        if deck.numberOfTiles() > 0:
            self.hand.append(TileInHand(deck.draw(), Info.UNKNOWN, Info.UNKNOWN))
    
    def getHint(self, hint:Hint):
        if type(hint.feature) == Color:
            for tile in self.hand:
                if tile.tile.color == hint.feature:
                    tile.colorInfo = Info.YES
                else:
                    tile.colorInfo = Info.NO
        elif type(hint.feature) == Number:
            for tile in self.hand:
                if tile.tile.number == hint.feature:
                    tile.numberInfo = Info.YES
                else:
                    tile.numberInfo = Info.NO

    def discardTile (self, discard:Discard, deck:Deck):
        for i in range(len(self.hand)):
            if self.hand[i].tile is discard.tile:
                self.hand.pop(i)
                self.draw(deck)
                return
     
class Game:
    def __init__(self, inputSources: Union[InputSource, List[InputSource]]):
        self.lifes = 3
        self.hints = 8
        self.deck = Deck()
        self.discardedTiles: List[Tile] = []
        self.players: List[Player]
        self.piles: Dict[Color, Union[Number, None]] = {color:None for color in Color}
        self.lastRound = False
        self.ended = False
        if (type(inputSources) == InputSource) or ((type(inputSources) == List) and (len(inputSources) == 1)):
            self.players = [Player(self.deck, inputSources) for _ in N_PLAYERS]
        elif len(inputSources) == N_PLAYERS:
            self.players = [Player(self.deck, inputSource) for inputSource in inputSources]
        else:
            self.players = [Player(self.deck, *sample(inputSources, 1)) for _ in N_PLAYERS]

    def availableActions(self) -> List[Type[Action]]:
        if self.hints == 0:
            return [Discard, Play]
        elif self.hints == 8:
            return [Hint, Play]
        else:
            return [Hint, Discard, Play]
    
    def move(self, player:Player, action:Action):
        if type(action) == Hint:
            if self.hints == 0:
                raise NameError('InvalidMove')
            else:
                self.hints -= 1
                action.player.getHint(action)
        elif type(action) == Discard:
            if self.hints == 8:
                raise NameError('InvalidMove')
            else:
                self.hints += 1
                player.discardTile(action, self.deck)
                self.discardedTiles.append(action.tile)
        elif type(action) == Play:
            player.discardTile(action, self.deck)
            if action.tile.number == 1:
                if self.piles[action.tile.color] == None:
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
            if self.lifes == 0:
                self.ended = True
            if self.deck.numberOfTiles() == 0:
                self.lastRound = True
    
    def simulateMove(self, player:Player, action:Action) -> 'Game':
        return deepcopy(self).move(player, action)

    def score(self) -> int:
        score: int = 0
        for color in self.piles:
            if self.piles[color] != None:
                score += self.piles[color]
        return score 