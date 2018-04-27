class tile:
    number = 0
    color = ""
    owner = ""
    knowncolor = False
    knownnumber = False

    def getnumber (self):
        return number
    def getcolor (self):
        return color
    def getowner (self):
        return owner
    def setowner (self, player):
        owner = player.name
        return 0
    def setknowncolor (self, known):
        knowncolor = known
        return 0
    def setknownnumber (self, known):
        knownnumber = known
        return 0
    def __init__ (self, num, col):
        if num > 0 and num < 6:
            number = num
        else:
            return 2
        if col in [red, blue, yellow, green]:
            color = col
        else:
            return 2
        return 0

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
    def run (self):
        return
    def __init__ (self):
        return
