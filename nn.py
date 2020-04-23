#!/usr/bin/python3
import mnist
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, train_images, train_labels, hidden_layer_size=12,
                 learning_rate=0.0001):
        print('init neuralnetwork')
        self.train_images = train_images
        self.train_labels = train_labels
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

        """
        28 * 28 is the size of the images in the dataset so its the size of the
        input layer
        we generate weights from -1 to 1 so the sigmoid function would
        output values ranging from 0 to 1
        """
        self.in_to_hidden_weights = np.random.uniform(-1, 1,
                (28 * 28, self.hidden_layer_size))
        self.hidden_to_out_weights = np.random.uniform(-1, 1,
                (self.hidden_layer_size, 10))
        self.hidden_layer = np.zeros(self.hidden_layer_size)
    
    def loss_func(self, actual_y, desired_y):
        return (desired_y - actual_y) ** 2

    # takes in an image, returns the output layer
    def feedforward(self, image):
        flattened_image = image.flatten()
        self.hidden_layer = np.zeros(self.hidden_layer_size)
        self.in_layer = flattened_image
        out_layer = np.zeros(10)
        for pixel_idx, pixel in enumerate(self.in_layer):
            for hidden_neuron in range(self.hidden_layer_size):
                self.hidden_layer[hidden_neuron] += pixel *\
                        self.in_to_hidden_weights[pixel_idx][hidden_neuron]
        for hidden_neuron in range(self.hidden_layer_size):
            self.hidden_layer[hidden_neuron] = sigmoid(self.hidden_layer[hidden_neuron])
        for hidden_neuron in range(self.hidden_layer_size):
            hidden_neuron_activation = self.hidden_layer[hidden_neuron]
            for i in range(out_layer.shape[0]):
                out_layer[i] += hidden_neuron_activation *\
                        self.hidden_to_out_weights[hidden_neuron][i]
        for out_neuron in range(out_layer.shape[0]):
            out_layer[out_neuron] = sigmoid(out_layer[out_neuron])
        self.out_layer = out_layer

    def backpropagate(self, desired_label):
        desired_out = np.zeros(10)
        desired_out[desired_label] = 1.
        for in_neuron in range(len(self.in_layer)):
            for hidden_neuron in range(len(self.hidden_layer)):
                changes_in_weight = np.array([])
                for desired_out_val in desired_out:
                    weights = np.array([in_neuron_weights[hidden_neuron] for 
                        in_neuron_weights in self.in_to_hidden_weights])
                    slope = 2. * (self.hidden_layer[hidden_neuron] - desired_out_val) *\
                        sigmoid_derivative((weights * self.in_layer).sum()) *\
                        self.in_layer[in_neuron]
                    if slope < 0:
                        change_in_weight = abs(slope) * self.learning_rate
                        changes_in_weight = np.append(changes_in_weight, change_in_weight)
                    if slope > 0:
                        change_in_weight = -1 * abs(slope) * self.learning_rate
                        changes_in_weight = np.append(changes_in_weight, change_in_weight)
                if changes_in_weight.size:
                    avg_change_in_weight = np.average(changes_in_weight)
                    self.in_to_hidden_weights[in_neuron][hidden_neuron] += avg_change_in_weight

    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for idx, image in enumerate(self.train_images):
                self.feedforward(image)
                desired_label = self.train_labels[idx]
                desired_out_layer = np.zeros(10)
                desired_out_layer[desired_label] = 1.
                loss = ((self.out_layer - desired_out_layer) ** 2).sum()
                total_loss += total_loss
                print(loss)
                #desired_label_activation = out_layer[desired_label]
                #loss = self.loss_func(desired_label_activation, 1.)
                #total_loss += loss
                #print(total_loss)
                self.backpropagate(desired_label)
            print('loss: {}'.format(total_loss))

if __name__ == '__main__':
    print('loading MNIST dataset...')
    train_images = mnist.train_images() / 255.
    train_labels = mnist.train_labels()
    test_images = mnist.test_images() / 255.
    test_labels = mnist.test_labels()
    print('data loaded')

    network = NeuralNetwork(train_images, train_labels)
    network.train(100)
    print('eyyy')
