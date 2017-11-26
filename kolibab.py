## Kristina Kolibab
## Homework 2

'''
This is a neural network with hidden layers m = 2
'''

import numpy as np

# Correctly reads in the training set based on it's formatting
def readTrainingFile( filename ):
    f = open(filename, "r")
    trainList = []
    trueAnswer = []
    lines = f.read()
    lines = lines.split('\n')
    if lines[-1] == "":
        lines = lines[:-1]
    f.close()
    for line in lines:
        curr_line = line.split(' ')
        if curr_line[-1] == '\n':
            curr_line = curr_line[:-1]
        vec = curr_line[:-1]
        label = float(curr_line[-1])
        vec = [float(x) for x in vec]
        #print "Vec = ", vec
        #print "Label = ", label
        trainList.append(vec) # read in each point 
        trueAnswer.append(label) # read in each -1/+1 answer
    trainList = np.array(trainList, dtype = float) # convert to numpy array
    trueAnswer = np.array(trueAnswer, dtype = float)
    #trainList = trainList/np.linalg.norm(trainList)
    return trueAnswer, trainList

# Correctly reads in the training set based on it's formatting
def readTestFile( filename ):
    f = open(filename, "r")
    vecTesters = []
    for line in f:
        vecTesters.append(line.split(' ')[0:-1])
    f.close()
    vecTesters = np.array(vecTesters, dtype = float)
    vecTesters = vecTesters/np.linalg.norm(vecTesters)
    return vecTesters

def sign_threshold(x):
	if x >= 0.0:
		return 1.0
	else:
		return -1.0

def tanh_d(x):
    return 1 - np.tanh(x)**2

# A class to store all my units
class Unit:
    def __init__(self, trainList, trueAnswer):
        self.input = trainList
        self.y = trueAnswer
        self.w = np.zeros(0) # hidden units
        self.output = np.zeros(0)
        self.b = 0.5

# Initialize Unit objects outside of class and functions
trueAnswer, trainList = readTrainingFile("train.txt") 

#Weights = np.loadtxt("weight-1.txt")

H1 = Unit(trainList, trueAnswer) # hidden unit 1
H1.w = np.random.rand(1000) # np.zeros(1000)
#H1.w = Weights[0]
H1.y = trueAnswer

H2 = Unit(trainList, trueAnswer) # hidden uint 2
H2.w = np.random.rand(1000) # np.zeros(1000)
#H2.w = Weights[1]
H2.y = trueAnswer

OU = Unit(trainList, trueAnswer) # output unit
OU.input = np.zeros(2)
OU.w = np.random.rand(2) # np.zeros(2) 
#OU.w = np.loadtxt("weight-0.txt")
OU.y = trueAnswer


def eval_forward():
        learning_rate = 0.05
        Error = True
        err = 0
        epoch = 0
        boolean_vals = []
        while(Error != False):
            Error = False
            err = 0
            for i in range(len(trainList)):
                    vec = trainList[i]
                    vec = vec/np.linalg.norm(vec)
                    val = trueAnswer[i]

                    # forwards --------------------
                    # input to hidden layers 
                    H1.activation = np.dot(H1.w, vec) + H1.b
                    H1.output = np.tanh(H1.activation) 
                    H2.activation = np.dot(H2.w, vec) + H2.b
                    H2.output = np.tanh(H2.activation) 

                    # hidden layers to output 
                    OU.input = [H1.output, H2.output]
                    OU.input = np.array(OU.input)
                    OU.activation = np.dot(OU.w, OU.input) + OU.b
                    OU.output = np.tanh(OU.activation)
                    pred = sign_threshold(OU.output)
                    boolean_vals.append(pred) # this goes to an output file
                    OU.Error = 0.5*abs(val - pred) # measure 0/1 mistakes

                    if OU.Error == 1.0:
                        Error = True
                        err += 1
                
                    # backwards --------------------
                    OU.delta0 = -2 * (val - pred) * tanh_d(OU.output) 

                    OU.delta1 = OU.delta0 * OU.w[0]* tanh_d(H1.activation)
                    OU.delta2 = OU.delta0 * OU.w[1] * tanh_d(H2.activation)
                    
                    H1.w = H1.w - (learning_rate * vec * OU.delta1)
                    H2.w = H2.w - (learning_rate * vec * OU.delta2)   
                    OU.w = OU.w - (learning_rate * OU.input * OU.delta0)
                    H1.b = H1.b - (learning_rate * OU.delta1)
                    H2.b = H2.b - (learning_rate * OU.delta2)

            epoch += 1
            print("Error: ", err/float(len(trainList)))
        print("Epoch: ", epoch)

        # file stuff to submit
        f = open("boolean_values.txt", "w")
        f.write(str(boolean_vals))
        f.close() 
    
        f_ = open("final_weights.txt", "w")
        f_.write(str(2))
        f_.write('\n')
        f_.write(str(OU.w))
        f_.write('\n')
        A = np.core.defchararray.replace(str(H1.w), '\n', '')
        f_.write(str(A))
        f_.write('\n')
        B = np.core.defchararray.replace(str(H2.w), '\n', '')
        f_.write(str(B))
        f_.close()

# M A I N
def main():
    eval_forward()

if __name__ == "__main__":
    main()

















