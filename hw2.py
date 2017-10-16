## Kristina Kolibab
## Homework 2

'''
This is a neural network with hidden layers m = 2
'''

import numpy as np

# Correctly reads in the training set based on it's formatting
def readTrainingFile( file ):

    f = open(file, "r")
    trainList = []
    trueAnswer = []
    for line in f:
        trainList.append(line.split(' ')[0:-1]) # read in each point 
        trueAnswer.append(line.split(' ')[-1]) # read in each 'Y'/'N' answer
    f.close()
    trainList = np.array(trainList, dtype = float) # convert to numpy array
    trueAnswer = np.array(trueAnswer, dtype = float)
    trainList = trainList/np.linalg.norm(trainList)
    np.array(trueAnswer)
    return trueAnswer, trainList

# Correctly reads in the training set based on it's formatting
def readTestFile( file ):

    f = open(file, "r")
    vecTesters = []
    for line in f:
        vecTesters.append(line.split(' ')[0:-1])
    f.close()
    vecTesters = np.array(vecTesters, dtype = float)
    vecTesters = vecTesters/np.linalg.norm(vecTesters)
    return vecTesters

# Converts my binary to ascii character
def bin_ascii( str ):
    int(str, 2)
    str = chr(int(str, 2))
    return str

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sig_deriv(x):
    return x * (1 - x)

# A class to store all my units
class Unit:

    def __init__(self, trainList, trueAnswer):
        self.tL = trainList
        self.tA = trueAnswer
        self.w = np.zeros(1000) # hidden units
        self.w2 = np.zeros(2) # output unit
        self.Z = np.zeros(900) # points after calculating dot product
        self.Yz = np.zeros(900) # taking derivative of points 
        self.Zo = np.zeros((900, 2)) # np.zeros(1000) # output unit calculating points w/ dot product
        self.Yzo = np.zeros((900, 2)) # np.zeros(1000) # output unit taking derivative of points
        self.Error = np.zeros(2) # Error 
        # Should I keep track of all errors, or maybe the 
        # prior error, to track if I'm making any progress?

# Forward propagation from hidden units output
def forward_propagate():
    trueAnswer, trainList = readTrainingFile("train.txt")
    hu1 = Unit(trainList, trueAnswer) # hidden unit 1
    hu2 = Unit(trainList, trueAnswer) # hidden unit 2
    ho = Unit(trainList, trueAnswer) # output unit

    # this is done one at a time for EVERY row
    i = 0
    for row in trainList: 
        # input to hidden layers
        hu1.Z[i] = np.dot(hu1.w, row)
        hu2.Z[i] = np.dot(hu2.w, row)
        hu1.Yz[i] = sigmoid(hu1.Z[i])
        hu2.Yz[i] = sigmoid(hu2.Z[i])
    
        # hidden layers to output layer
        ho.Zo[i][0] = np.dot(ho.w2[0], hu1.Yz[i]) # does the output need a weight for each hidden unit?
        ho.Zo[i][1] = np.dot(ho.w2[1], hu2.Yz[i])
        ho.Yzo[i][0] = sigmoid(ho.Zo[i][0])
        ho.Yzo[i][1] = sigmoid(ho.Zo[i][1])
        
        # Want error as close to 0 as you can,
        # how do you know when you are close enough?
        ho.Error[0] = (ho.Yzo[i][0] - hu1.tA[i])**2
        ho.Error[1] = (ho.Yzo[i][1] - hu2.tA[i])**2
        i += 1

# Backward propagation from output to hidden units
def backward_propagate():
          
    
# M A I N
def main():
    forward_propagate()
    
if __name__ == "__main__":
    main()


