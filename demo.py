from numpy import exp, array, random, dot, mean, abs


class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
        self.in_out_dim = [[3, 4], [4, 1]]
        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        # self.synaptic_weights = 2 * random.random((3, 1)) - 1
        self.synaptic_weights = []
        self.num_layers = 2
        for iteration in xrange(self.num_layers):
            self.synaptic_weights.append(
                2 * random.random((self.in_out_dim[iteration][0], self.in_out_dim[iteration][1])) - 1)

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single
            # neuron).
            layer_0 = training_set_inputs
            layer_1 = self.__sigmoid(dot(layer_0, self.synaptic_weights[0]))
            layer_2 = self.__sigmoid(dot(layer_1, self.synaptic_weights[1]))
            # layer_2 = dot(layer_1, self.synaptic_weights[1])
            # output = self.think(training_set_inputs)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error_2 = training_set_outputs - layer_2
            # Print to see the decrease in the error
            if (iteration % 10000) == 0:
                print "Error:" + str(mean(abs(error_2)))
            # Back Propagation
            delta_2 = error_2 * \
                self.__sigmoid_derivative(layer_2)
            error_1 = dot(delta_2, self.synaptic_weights[1].T)
            delta_1 = dot(
                error_1 * self.__sigmoid_derivative(layer_1), learning_rate)

            # # Adjust the weights.
            self.synaptic_weights[0] += dot(layer_0.T, delta_1)
            self.synaptic_weights[1] += dot(layer_1.T, delta_2)

    # # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        layer_0 = inputs
        layer_1 = self.__sigmoid(dot(layer_0, self.synaptic_weights[0]))
        layer_2 = self.__sigmoid(dot(layer_1, self.synaptic_weights[1]))
        return layer_2


if __name__ == "__main__":

    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs,
                         training_set_outputs, 100000, 0.01)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))
