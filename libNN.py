from numpy import array, matrix, zeros, empty, exp, array_equal, outer, dot, transpose
from numpy.random import randn
from random import uniform as randu

#Hyperparameters
NInput = 83
NOutput = 35

NHiddenLayers = 2
NeuronsHiddenLayer = 90

alpha = 1.0
eta = 0.05

#Global Variables
colors = ["Red", "Blue", "Yellow", "Green"]

class NeuralNetwork:
    def __init__ (self):
        #matrix of weight from input to first layer
        self.w = [] #list of weight matrix
        self.b = [] #list of bias vector
        #Need i/o from file to store improvements
        for i in range (0, (NHiddenLayers + 1)):
            rows = NeuronsHiddenLayer
            columns = NeuronsHiddenLayer
            if i == 0:
                rows = NInput
            if i == NHiddenLayers:
                columns = NOutput
            self.w.append(zeros((rows, columns))) #Random Gaussian(0,1)
            self.b.append(zeros(columns)) #Zeros
        self.moves = []
        self.lastmove = []
        self.players = []
        self.activeplayer = ""

    def parsestatus (self, status):
        input = empty(NInput)
        input[0] = status["life"]
        input[1] = status["hint"]
        input[2] = status["deck"]
        #Discarded tiles
        for i in range(0,4):
            for j in range(0,5):
                input[3 + i*5 + j] = status["discard"].count([(j+1), colors[i]])
        #Filled till input[22] included

        #Opponents' hand
        #-1 means None
        #Colors are [0,4] instead of [1,5]
        self.players = list(status["players"].keys())
        ind = self.players.index(status["activeplayer"])
        self.activeplayer = status["activeplayer"]
        for i in range (0,3):
            for j in range (0,4):
                try:
                    a = status["players"][self.players[(ind + 1 + i) % 4]]["hand"][j][0]
                except:
                    a = None
                try:
                    b = status["players"][self.players[(ind + 1 + i) % 4]]["hand"][j][1]
                except:
                    b = None
                try:
                    c = status["players"][self.players[(ind + 1 + i) % 4]]["hand"][j][2]
                except:
                    c = None
                try:
                    d = status["players"][self.players[(ind + 1 + i) % 4]]["hand"][j][3]
                except:
                    d = None

                first = 23 + i*16 + j*4
                if a == None:
                    input[first] = -1
                else:
                    input[first] = a - 1
                if b == None:
                    input[first + 1] = -1
                else:
                    input[first + 1] = colors.index(b)
                if c == None:
                    input[first + 2] = -1
                elif c == True:
                    input[first + 2] = 1
                else:
                    input[first + 2] = 0
                if d == None:
                    input[first + 3] = -1
                elif d == True:
                    input[first + 3] = 1
                else:
                    input[first + 3] = 0
        #End opponents' has
        #Fill till input[70] included
        #Active player's hand
        hand = status["players"][status["activeplayer"]]["hand"]
        for i in range (0,4):
            a, b = hand [i]
            if a == None:
                input[71 + 2*i] = -1
            elif a == True:
                input[71 + 2*i] = 1
            else:
                input[71 + 2*i] = 0
            if b == None:
                input[72 + 2*i] = -1
            elif b == True:
                input[72 + 2*i] = 1
            else:
                input[72 + 2*i] = 0
        #Fill till input[78] included
        for i in range(0,4):
            input[79 + i] = status["stack"][i]
        #Parsind Completed
        return input

    def parseoutput (self, output):
        if output in range (0,4):
            return [0, output]
        elif output in range (4,8):
            return [1, output - 4]
        elif output in range (8,35):
            player = ((output - 8) // 9) + 1
            if ((output - 8) % 9) in range(0,5):
                return [2, (((output - 8) % 9) + 1), \
                    self.players[(self.players.index(self.activeplayer) + player) % 4]]
            elif ((output - 8) % 9) in range(5,9):
                return [2, colors[(((output - 8) % 9) - 5)], \
                    self.players[(self.players.index(self.activeplayer) + player) % 4]]

    def sigma (self, x):
        return 1/(1 + exp(-x))

    def softmax (self, x):
        y = exp (x)
        return y/sum(y)

    def goforward(self, input):
        input = input #Maybe unnecessary
        for i in range(0, NHiddenLayers + 1):
            if i == NHiddenLayers:
                output = self.softmax((input @ self.w[i]) + self.b[i])
            else:
                output = self.sigma((input @ self.w[i]) + self.b[i])
            input = output
        return output

    def getinput (self, status):
        if status["last"] and self.lastmove != []:
            self.moves.append(self.lastmove)
            self.lastmove = []
        input = self.parsestatus(status)
        outputprob = self.goforward(input)
        r = randu(0.0,1.0)
        print (str(r))
        output = 0
        while ((sum(outputprob[:(output + 1)])) < r):
            output = output + 1
        self.lastmove = [input, output, outputprob]
        print (str(output))
        return self.parseoutput(output)

    def cost (self, x, n, points):
        return (abs(x - (points / 20.0)))**(alpha*n)

    def dcost (self, x, n, points):
        if x != (points/20.0):
            return alpha * n * self.cost (x, n, points) / (x - (points / 20.0))
        else:
            return 0

    def update (self, points):
        if not (array_equal(self.lastmove, self.moves[-1])):
            self.moves.append(self.lastmove)

        #I aim to generate a great number of corrections (n) and sum up all
        #while still using old weights and bias for computing
        #I don't know if it's methodologically correct

        wupd = [] #list of updated weight matrix
        bupd = [] #list of updated bias vector
        for i in range (0, (NHiddenLayers + 1)):
            # rows = NeuronsHiddenLayer
            # columns = NeuronsHiddenLayer
            # if i == 0:
            #     rows = NInput
            # if i == NHiddenLayers:
            #     columns = NOutput
            wupd = self.w
            bupd = self.b

        for n in range(1, len(self.moves) + 1):
            print(n)

            input, output, outputprobvect = self.moves[-n]
            outputprob = outputprobvect[output]

            dC = zeros(NOutput)
            dC[output] = self.dcost (outputprob, n, points)

            #If dC is really small the correction will be smaller (and not significant)
            #In order to preserve some machine-time corrections will be calculated only for not small dC
            #if dC[output] > 10**(-20):
            if True:

                z = []
                h = []
                tmpinput = input
                for i in range (0, NHiddenLayers + 1):
                    z.append((tmpinput @ self.w[i]) + self.b[i])
                    if i == NHiddenLayers:
                        h.append(self.softmax(z[i]))
                    else:
                        h.append(self.sigma(z[i]))
                    tmpinput = h[i]
                    #print (z[i].shape)
                    #print (h[i].shape)

                root = dC
                for l in range (1, NHiddenLayers + 2):
                    if l == 1:
                        bupd[-l] = root * h[-l] * (1 - h[-l])
                    else:
                        bupd[-l] = dot(root, (transpose(self.w[-l + 1]) * (h[-l] * (1 - h[-l]))))
                    if l == NHiddenLayers + 1:
                        wupd[-l] = outer(input, bupd[-l])
                    else:
                        wupd[-l] = outer(h[-(l + 1)], bupd[-l])
                    root = bupd[-l]

                self.w = wupd
                self.b = bupd
