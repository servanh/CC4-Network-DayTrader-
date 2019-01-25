import numpy as np

class CC4:

    #number of symbols that are different from the zero-symbol of the
    #language used.

    def _hammingWeight(self, vector):
        hammingWeight = 0
        for i in range(0, len(vector)):
            if vector[i] == 1:
                hammingWeight += 1  # "For each training vector presented to the network if an input neuron recieves a 1
        return hammingWeight       # its weight to the hidden neuron corresponding to this training vector is set to 1."

    def feedForward(self, input):
        input = np.append(input, [1])
        return self.start(np.matmul(self.start(np.matmul(input, self.inWeights)), self.outWeights))

    def __init__(self, sampleIn, trainOut, radius):
        hiddenSize = len(sampleIn)
        inputSize = len(sampleIn[0])
        trainOutLen = len(trainOut)

        if hiddenSize != trainOutLen:
            raise ValueError('Output vectors and input vectors length should be the same.')

        outputSize = len(trainOut[0])
        # "...the last node of the input layer is set to one to act as bias to the hidden layer."
        self.inWeights = np.empty([inputSize+1, hiddenSize])
        self.outWeights = np.empty([hiddenSize, outputSize])
        for i in range(0, hiddenSize):
            # "If s is the number of 1's in the training vector, excluding the
            # bias input , and the desired radius of generalization is r, then the weight between bias
            # neuron and the hidden neuron corresponding to this training vector is r(radius) -s +1."
            s = self._hammingWeight(sampleIn[i])
            self.inWeights[inputSize, i] = radius - s + 1
            for j in range(0, inputSize):
                if sampleIn[i][j] == 1:
                    self.inWeights[j, i] = 1
                else:
                    self.inWeights[j, i] = -1
            for a in range(0, outputSize):
                # "The weights in the output layer are equal to 1 if the output value is 1
                #  and -1 if the output value is 0"
                if trainOut[i][a] == 1:
                    self.outWeights[i, a] = 1
                else:
                    self.outWeights[i, a] = -1

        #"The output of the activation function is 1 if summation is positive
        #and zero otherwise"
        self.start = np.vectorize(lambda x: 1 if x>0 else 0)


