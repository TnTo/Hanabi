from random import shuffle

class tile:
    number = 0
    color = ""
    owner = ""
    knowncolor = False
    knownnumber = False

    def getnumber (self):
        return self.number
    def getcolor (self):
        return self.color
    def getowner (self):
        return self.owner
    def setowner (self, player):
        self.owner = player.name
        return 0
    def setknowncolor (self, knowncolor):
        self.knowncolor = known
        return 0
    def setknownnumber (self, knownnumber):
        self.knownnumber = known
        return 0
    def printtile (self):
        print (str(self.number) + self.color)
        return 0
    def __init__ (self, num, col):
        if num > 0 and num < 6:
            self.number = num
        else:
            return 2
        if col in ["red", "blue", "yellow", "green"]:
            self.color = col
        else:
            return 2
        return

class deck:
    tiles = []

    def draw (self, player):
        return
    def __init__ (self):
        for num in range (1,6):
            for col in ["red", "blue", "yellow", "green"]:
                if num == 1: numberoftiles = 3
                elif num in [2,3,4]: numberoftiles = 2
                elif num == 5: numberoftiles = 1
                for i in range (0, numberoftiles):
                    self.tiles.append( tile(num,col) )
        #random sorting
        shuffle (self.tiles)
        #for item in self.tiles:
        #    item.printtile()
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
    name = ""

    def draw (self):
        return
    def discard (self):
        return
    def hint (self):
        return
    def gethand (self):
        return
    def __init__ (self, name):
        return

class game:
    Deck = deck()
    Board = board()
    North = player("North")
    South = player("South")
    East = player("East")
    West = player("West")

    players = [South,East,North,West]

    def move (self):
        return
    def hasended (self):
        return
    def run (self):
        return
    def __init__ (self):
        return
