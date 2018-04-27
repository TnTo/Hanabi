class tile:
    number = 0
    color = ""
    owner = ""
    knowncolor = False
    knownnumber = False

    def getnumber (self):
        return
    def getcolor (self):
        return
    def getowner (self):
        return
    def setowner (self):
        return
    def setknowncolor (self):
        return
    def setknownnumber (self):
        return
    def __init__ (self, num, col):
        return

class deck:

    def draw (self, player):
        return
    def __init__ (self):
        return

class board:
    hint = 8
    life = 3
    stacks = [0,0,0,0]

    def loselife(self):
        return
    def getlife(self):
        return
    def gainhint(self):
        return
    def losehint(self):
        return
    def gethint(self):
        return
    def addtile (self):
        return
    def gettile (self):
        return
    def __init__ (self):
        return

class player:
    hand = []

    def draw (self):
        return
    def discard (self):
        return
    def hint (self):
        return
    def gethand (self):
        return
    def __init__ (self):
        return

class game:
    Deck = deck()
    Board = board()
    North = player()
    South = player()
    East = player()
    West = player()

    players = [South,East,North,West]

    def move (self):
        return
    def hasended (self):
        return
    def __init__ (self):
        return
