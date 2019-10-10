import numpy as np

class Perceptron:
    def __init__(self, input_dim, err_threshold=0):
        self.INPUT_DIM = input_dim
        self.LEARNING_RATE = 0.5
        self.weights = np.random.random_sample((input_dim,1)).flatten()
        self.bias = np.random.rand()
        self.ERR_THRESHOLD = err_threshold

    def getWeights(self):
        return((self.weights, self.bias))

    def train(self, data, labels):
        data = np.asarray(data)
        labels = np.asarray(labels)
        bias = self.bias*np.ones((data.shape[0]))

        # initial forward pass
        Y = np.dot(data, self.weights)
        Y = Y.flatten() + self.bias*np.ones((data.shape[0]))
        Y[Y > 0] = 1
        Y[Y <= 0] = 0

        while(MeanSquareError(Y, labels) > self.ERR_THRESHOLD):
            # forward pass to get perceptron output
            Y = np.dot(data, self.weights)
            Y = Y.flatten() + self.bias*np.ones((data.shape[0]))
            Y[Y > 0] = 1
            Y[Y <= 0] = 0

            # calculate new weights based on forward pass
            new_weights = self.weights + self.LEARNING_RATE*(np.dot((labels - Y), data))
            self.weights = new_weights

            # calculate new bias based on forward pass
            new_bias = self.bias + np.sum(self.LEARNING_RATE*(labels.transpose() - Y))
            self.bias = new_bias

            print("Mean Squared Error: ", MeanSquareError(Y, labels), "\n")


def MeanSquareError(Y, labels):
    num_samples = labels.shape[0]
    err = np.sum((labels.transpose() - Y)**2)/num_samples
    return err



if __name__ == '__main__':
    p = Perceptron(2)
    # data and labels for AND gate
    DATA = [[0,0],[0,1],[1,0],[1,1]]
    LABELS = [0,0,0,1]
    p.train(data=DATA, labels=LABELS)
    print(p.getWeights())
