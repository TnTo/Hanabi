from random import shuffle

colors = ["red", "blue", "yellow", "green"]

class tile:
    def __init__ (self, num, col, owner):
        if num > 0 and num < 6:
            self.number = num
        else:
            self.number = 0
            return 2
        if col in colors:
            self.color = col
        else:
            self.color = ""
            return 2
        self.owner = owner
        knowncolor = False
        knownnumber = False
        return

    def getnumber (self):
        return self.number

    def getcolor (self):
        return self.color

    def getowner (self):
        return self.owner

    def setowner (self, player):
        self.owner = player.getname()
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


class deck:
    def __init__ (self):
        self.tiles = []
        for num in range (1,6):
            for col in colors:
                if num == 1: numberoftiles = 3
                elif num in [2,3,4]: numberoftiles = 2
                elif num == 5: numberoftiles = 1
                for i in range (0, numberoftiles):
                    #We could add a check: if num and col are not valid tile(num,col,"Deck") == 2
                    self.tiles.append( tile(num,col,"Deck") )
        #random sorting
        shuffle (self.tiles)
        #for item in self.tiles:
        #    item.printtile()
        return

    def howmanytile (self):
        return len(self.tiles)

    def draw (self):
        if self.howmanytile() == 0:
            return 3
        return self.tiles.pop()


class board:
    def __init__ (self):
        self.hint = 8
        self.life = 3
        self.stacks = [0,0,0,0]
        return

    def loselife(self):
        if self.life == 0:
            return 3
        self.life = self.life - 1
        return 0

    def getlife(self):
        return self.life

    def gainhint(self):
        self.hint = self.hint + 1
        return 0

    def losehint(self):
        if self.hint == 0:
            return 3
        self.hint = self.hint - 1
        return 0

    def gethint(self):
        return self.hint

    def getstacks (self):
        return self.stacks

    def addtile (self, tile):
        tilecolor = color.index(tile.getcolor())
        tilenumber = tile.getnumber()
        stacknumber = self.stacks[tilecolor]
        if tilenumber == (stacknumber + 1):
            tile.setowner("Board")
            self.stacks[tilecolor] = tilenumber
            if tilenumber == 5:
                self.gainhint()
        else:
            tile.setowner("Discard")
            #I'm not sure this is valid in python
            if (not self.loselife())
                return 4
        return 0


class player:
    hand = []
    name = ""

    def draw (self, deck):
        return
    def discard (self):
        return
    def hint (self):
        return
    def gethand (self):
        return self.hand
    def getname (self):
        return self.name
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
