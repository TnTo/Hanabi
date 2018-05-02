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

    def setowner (self, name):
        self.owner = name
        return 0

    def setknowncolor (self, known):
        self.knowncolor = known
        return 0

    def setknownnumber (self, known):
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
            return 3 #Check
        return self.tiles.pop()


class board:
    def __init__ (self):
        self.hint = 8
        self.life = 3
        self.stacks = [0,0,0,0]
        return

    def loselife(self):     #to refactor to check whether the game is lost
        if self.life == 0:
            return 3 #Check
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
            if (self.loselife() == 2):
                return 4
        return 0


class player:
    def draw (self):
        #This not check how many tile already in hand
        drawntile = self.deck.draw()
        if (drawntile == 3):
            return 4 #Check
        else:
            self.hand.append(drawntile)
            drawntile.setowner(self.name)
            return 0

    def __init__ (self, name, game):
        self.name = name
        self.hand = []
        self.deck = game.getdeck()
        self.board = game.getboard()
        while (len(self.hand) < 5):
            self.draw()
        return

    def discard (self, index):
        if (self.deck.howmanytile() == 0 or len(self.hand == 0)):
            return 3 #Check
        else:
            if (index >= 0 and index < 5):
                self.hand.pop(index).setowner("Discard")
                self.draw()
                self.board.gainhint()
                return 0
            else:
                return 2

    def hint (self, category, player):
        if category in colors:
            #Hint is about color
            if (player != self.name and player in game.getplayers()):
                for tile in player.gethand():
                    if tile.getcolor == category:
                        tile.setknowncolor(True)
                        return 0
            else:
                return 2
        elif category in range(1,6):
            if (player != self.name and player in game.getplayers()):
                for tile in player.gethand():
                    if tile.getnumber == category:
                        tile.setknownnumber(True)
                        return 0
            else:
                return 2
        else:
            return 2

    def playtile (self, index):
        if (self.deck.howmanytile() == 0 or len(self.hand == 0)):
            return 3 #Check
        else:
            if (index >= 0 and index < 5):
                self.board.addtile(self.hand.pop(index))
                return 0 #check
            else:
                return 2

    def gethand (self):
        return self.hand

    def getname (self):
        return self.name


class game:
    def __init__ (self):
        self.Deck = deck()
        self.Board = board()
        self.North = player("North", self)
        self.South = player("South", self)
        self.West = player("West", self)
        self.East = player("East", self)
        self.players = [self.South, self.East, self.North, self.West]
        return

    def getdeck(self):
        return self.Deck

    def getboard (self):
        return self.Board

    def getplayers (self):
        return self.players

    def move (self):
        return
    def hasended (self):
        return
    def run (self):
        return
