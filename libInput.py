colors = ["Red", "Blue", "Yellow", "Green"]

class inputsource:

    def getinput(self, status):
        self.printstatus(status)
        return [0]

    def printstatus(self, status):
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
        print(hand)
        players = list(status["players"].keys())
        i = players.index(status["activeplayer"])
        for p in range(1,4):
            pp = p + i % 4
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
