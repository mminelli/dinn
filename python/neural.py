import numpy as np
import numpy.random as rnd
import time
import pickle
import os
import sys

def m_sign(x):
    if x < -0.2:
        return -1
    elif x > 0.2:
        return 1
    else:
        return 0
m_sign = np.vectorize(m_sign)  # in order to apply the function to numpy arrays


def m_sign2(x):
    if x < 0:
        return -1
    return 1
m_sign2 = np.vectorize(m_sign2)


def binarize(x):
    return m_sign2(x)
binarize = np.vectorize(binarize)


def ternarize(x):
    return m_sign(x)
ternarize = np.vectorize(ternarize)


tanh_steep = 1  # coefficient for controlling the steepness of the tanh function
def activation_function(x):
    # return (1.0/(1.0 + np.exp(-x)))  # sigmoid
    return np.tanh(tanh_steep*x)  # tanh
    # return np.arctan(x)  # arctan
    # if x <= 0:
    #   return -1
    # return 1
activation_function = np.vectorize(activation_function)


def activation_derivative(x):
    act_x = activation_function(x)
    # return (act_x * (1.0 - act_x))  # deriv of sigmoid
    return (tanh_steep * (1 - act_x * act_x))  # deriv of tanh
    # return 1/(1+x*x)  # deriv of arctan
activation_derivative = np.vectorize(activation_derivative)


def from_lbl_to_vector(label):
    out = np.full(10, 0.0)
    out[label] = 1.0
    return out


def H(x):
    if abs(x) <= 1:
        return 1
    return 0
H = np.vectorize(H)


def softmax(x):  # x is a matrix
    out = np.zeros(x.shape)
    for i in range(x.shape[0]):
        e_x = np.exp(x[i])
        denominator = np.sum(e_x)
        for j in range(x.shape[1]):
            out[i][j] = e_x[j] / denominator
    return out


class NeuralNetwork(object):

    def __init__(self, topology, learn_rate, regularization_factor, tick_size, learning_rate_decay=False, minibatch_size=100):
        super(NeuralNetwork, self).__init__()

        self.topology = topology
        self.L = len(topology)
        self.learn_rate = learn_rate
        self.regularization_factor = regularization_factor
        self.minibatch_size = minibatch_size
        self.tick_size = tick_size
        self.learning_rate_decay = learning_rate_decay

        self.weights = []
        self.biases = []
        self.weights.append(np.zeros(1))  # nothing entering the 1st layer
        self.biases.append(np.zeros(1))  # no bias for 1st layer
        for layer in range(1, self.L):
            self.weights.append(-2+4*rnd.rand(topology[layer-1], topology[layer]))
            self.biases.append(-2+4*rnd.rand(1, topology[layer]))

        self.weighted_sums = []
        self.activated = []
        self.bin_activated = []
        for layer in range(0, self.L):
            self.weighted_sums.append(np.zeros(topology[layer]))
            self.activated.append(np.zeros(topology[layer]))
            self.bin_activated.append(np.zeros(topology[layer]))


    def print_topology(self):
        output = "("
        for i in range(len(self.topology)):
            output += str(self.topology[i])
            if i < len(self.topology) - 1:
                output += " - "
        output += ")"
        return output


    def print_network_info(self):
        output = self.print_topology()
        output += " - learning rate: " + str(self.learn_rate)\
         + " - regularization factor: " + str(self.regularization_factor)\
         + " - tick size: " + str(self.tick_size)\
         + " - minibatch size: " + str(self.minibatch_size)
        return output


    def process_weights(self, x):
        return np.int((np.round(x/self.tick_size)))
    process_weights = np.vectorize(process_weights)


    def apply(self, x):
        predictions = []
        layer_input = x
        for layer in range(1, self.L):
            layer_input = np.dot(layer_input, self.process_weights(self, self.weights[layer])) + self.process_weights(self, self.biases[layer])
            if layer < self.L-1:
                layer_input = binarize(activation_function(layer_input*self.tick_size))
        for index in range(x.shape[0]):
            predictions.append(np.argmax(layer_input[index]))
        return predictions


    def train(self, X, Y, epochs=10000, evaluate=True, evaluate_after_each_minibatch=False):

        num_training_samples = len(Y)
        correct = 0
        correct = self.evaluate(Xtest, Ytest)
        max_accuracy = correct
        epoch_max_accuracy = 0
        print('{}\nBefore training  -  Accuracy: {}/{}'.format(self.print_network_info(), correct, len(Ytest)))

        for epoch in range(epochs):
            print('Epoch {}...'.format(epoch+1))
            # shuffle X and Y in the same way (to maintain the correspondence)
            rnd_state = rnd.get_state()
            rnd.shuffle(X)
            rnd.set_state(rnd_state)
            rnd.shuffle(Y)

            # decrease the learning rate
            if self.learning_rate_decay:
                if epoch > 0 and epoch % 5 == 0:
                    self.learn_rate *= 0.95

            # divide the training set in minibatches of the given size
            num_minibatches = int(np.ceil(num_training_samples/self.minibatch_size))
            minibatches_img = []  # list of minibatches for images
            minibatches_lbl = []  # list of minibatches for labels
            for i in range(0, num_minibatches):
                minibatches_img.append(X[i*self.minibatch_size : (i+1)*self.minibatch_size])
                minibatches_lbl.append(Y[i*self.minibatch_size : (i+1)*self.minibatch_size])

            for batch in range(0, num_minibatches):
                if batch % 200 == 0:
                    print('Epoch {} - Minibatch {}/{}'.format(epoch+1, batch+1, num_minibatches))
                self.train_minibatch(minibatches_img[batch], minibatches_lbl[batch])

            # at the end of each epoch, evaluate the accuracy of the NN
            if evaluate == True:
                correct = self.evaluate(Xtest, Ytest)
                # correct_no_process_weights = self.evaluate_no_process_weights(Xtest, Ytest)
                print('{}\nEpoch {}  -  Accuracy: {} ({})/{}'.format(self.print_network_info(), epoch+1, correct, correct_no_process_weights, len(Ytest)))

                with open('all_accuracies_' + '_'.join([str(x) for x in topology]) + '.txt', 'at') as acc_file:
                    acc_file.write(str(correct) + '\n')

                if correct > max_accuracy:
                    max_accuracy = correct
                    epoch_max_accuracy = epoch + 1
                    # save the best network
                    with open('best_biases_' + '_'.join([str(x) for x in topology]) + '.dat', 'wb') as out_biases:
                        pickle.dump(self.biases, out_biases)
                    with open('best_weights_' + '_'.join([str(x) for x in topology]) + '.dat', 'wb') as out_weights:
                        pickle.dump(self.weights, out_weights)
                    with open('accuracy_' + '_'.join([str(x) for x in topology]) + '.txt', 'wt') as acc_file:
                    	acc_file.write(str(correct) + '\n')
                print('Best accuracy: {}/{} at epoch {}'.format(max_accuracy, len(Ytest), epoch_max_accuracy))


    def train_minibatch(self, X, Y):
        nabla_b, nabla_w = self.backprop_minibatch(X, Y)
        # update weights and biases
        for k in range(1, self.L):
            self.biases[k] = self.biases[k] - self.learn_rate * (nabla_b[k])
            self.weights[k] = self.weights[k] - self.learn_rate * (nabla_w[k] + self.regularization_factor*self.weights[k])


    def backprop_minibatch(self, X, Y):
        num_samples = len(Y)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        self.activated[0] = X

        
        # ========    FEEDFORWARD    ========
        
        for layer in range(1, self.L):
            self.weighted_sums[layer] = self.tick_size*np.dot(self.activated[layer-1], self.process_weights(self, self.weights[layer])) + self.tick_size*self.process_weights(self, self.biases[layer])
            if layer < self.L-1:
                self.activated[layer] = activation_function(self.weighted_sums[layer])
                self.bin_activated[layer] = binarize(self.activated[layer])
            else:
                self.activated[layer] = softmax(self.weighted_sums[layer])


        # ========    FEEDBACK    ========

        output = self.activated[-1]  # network output: last activations
        Ymat = np.zeros((num_samples, 10))
        for ii in range(num_samples):
            Ymat[ii] = from_lbl_to_vector(Y[ii])
        
        delta = output - Ymat
        nabla_b[-1] = np.sum(delta, 0)
        nabla_w[-1] = np.dot(self.bin_activated[-2].transpose(), delta)

        # keep going backward
        for layer in range(2, self.L):
            sp = activation_derivative(self.weighted_sums[-layer])
            delta = np.dot(delta, self.weights[-layer+1].transpose()) * H(self.activated[-layer]) * sp
            nabla_b[-layer] = np.sum(delta, 0)
            nabla_w[-layer] = np.dot(self.activated[-layer-1].transpose(), delta)
        return (nabla_b, nabla_w)


    def evaluate(self, Xtest, Ytest):
        num_samples = len(Ytest)
        count_correct = 0
        predictions = self.apply(Xtest)
        for index in range(num_samples):
            if predictions[index] == Ytest[index]:
                count_correct += 1
            # else:
            #     print(index)
        return count_correct



#####  MAIN  #####


rnd.seed(0)

print('Loading input data...')

# load dataset
with open('img_train.dat', 'rb') as inImg:
    Xtrain = pickle.load(inImg)
with open('lbl_train.dat', 'rb') as inLbl:
    Ytrain = pickle.load(inLbl)
with open('img_test.dat', 'rb') as inImg:
    Xtest = pickle.load(inImg)
with open('lbl_test.dat', 'rb') as inLbl:
    Ytest = pickle.load(inLbl)

print('Input data loaded.')

image_size = 16
nInput = image_size * image_size
nOutput = 10

topology = [nInput]
tick_size = float(sys.argv[1])
for ll in sys.argv[2:]:
	topology.append(int(ll))
topology.append(nOutput)

nn = NeuralNetwork(topology, learn_rate=0.006, regularization_factor=0.001, tick_size=tick_size, minibatch_size=10, learning_rate_decay=True)

load_network_from_file = False
# load the best network and try some predictions
if load_network_from_file:
    with open('start_biases.dat', 'rb') as biases:
      start_biases = pickle.load(biases)
    with open('start_weights.dat', 'rb') as weights:
      start_weights = pickle.load(weights)
    nn.weights = start_weights
    nn.biases = start_biases


start = time.time()

nn.train(Xtrain, Ytrain)

end = time.time()
elapsed = end - start
print('Time: {} s'.format(elapsed))
