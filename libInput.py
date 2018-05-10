colors = ["Red", "Blue", "Yellow", "Green"]

class inputsource:

    def getinput(self, status):
        if status["last"]:
            self.printstatus(status)
        else:
            print("Your move isn't allowed, try a different one")
        ret = []
        print("Choose your move")
        print("0 to discard a tile")
        print("1 to play a tile")
        print("2 to give an hint")
        while True:
            move = int(input("-> "))
            if move in range(0,3):
                ret.append(move)
                break
            else:
                print("Invalid selection")
        if move in [0,1]:
            print("Choose the tile")
            for h in range (0,4):
                num = status["players"][status["activeplayer"]]["hand"][h][0]
                col = status["players"][status["activeplayer"]]["hand"][h][1]
                if num == None:
                    num = "*"
                else:
                    num = str(num)
                if col == None:
                    col = "*"
                else:
                    col = col[0]
                print(str(h) + " " + num + col)
            while True:
                tile = int(input("-> "))
                if tile in range (0,4):
                    ret.append(tile)
                    break
                else:
                    print("Invalid selection")
        else:
            print("Choose the player")
            players = list(status["players"].keys())
            i = players.index(status["activeplayer"])
            for p in range(1,4):
                print(str(p-1) + " " + players[(p + i) % 4])
            while True:
                player = int(input("-> "))
                if player in range (0,3):
                    ret.append(players[(player + 1 + i) % 4])
                    break
                else:
                    print("Invalid selection")
            print("Choose the hint")
            for n in range(1,6):
                print(str(n) + " to reveal the " + str(n) + " tiles")
            for c in range(6,10):
                print(str(c) + " to reveal the " + colors[c-6] + " tiles")
            while True:
                category = int(input("-> "))
                if category in range (1,10):
                    if category in range(1,6):
                        ret.append(category)
                    else:
                        ret.append(colors[category-6])
                    break
                else:
                    print("Invalid selection")
        return ret

    def printstatus(self, status):
        print()
        print(status["activeplayer"] + "'s turn")
        print()
        print("Hint: " + str(status["hint"]))
        print("Life: " + str(status["life"]))
        print("Tiles in the deck: " + str(status["deck"]))
        print("Stacks:")
        for i in range (0,4):
            print(" " + str(status["stack"][i]) + " " + colors[i])
        print("Discarded tiles:")
        discarded = ""
        for t in status["discard"]:
            discarded = discarded + str(t[0]) + t[1][0] + " "
        print (discarded)
        print ()
        hand = ""
        print("Your hand:")
        for h in range (0,4):
            num = status["players"][status["activeplayer"]]["hand"][h][0]
            col = status["players"][status["activeplayer"]]["hand"][h][1]
            if num == None:
                num = "*"
            else:
                num = str(num)
            if col == None:
                col = "*"
            else:
                col = col[0]
            hand = hand + " " + num + col
        print(hand + "\n")
        players = list(status["players"].keys())
        i = players.index(status["activeplayer"])
        for p in range(1,4):
            pp = (p + i) % 4
            print(players[pp] + "'s hand:")
            hand = ""
            khand = ""
            for h in range(0,4):
                num = str(status["players"][players[pp]]["hand"][h][0])
                col = status["players"][players[pp]]["hand"][h][1][0]
                knum = status["players"][players[pp]]["hand"][h][2]
                kcol = status["players"][players[pp]]["hand"][h][3]
                if knum == True:
                    knum = num
                else:
                    knum = "*"
                if kcol == True:
                    kcol = col
                else:
                    kcol = "*"
                hand = hand + " " + num + col
                khand = khand + " " + knum + kcol
            print(hand)
            print(khand)
