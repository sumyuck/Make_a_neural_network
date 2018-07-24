from numpy import exp, array, random, dot, mean, abs


class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
        self.in_out_dim = [[3, 4], [4, 1]]
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
        for tr_iter in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            l = self.num_layers
            layers = []
            errors = [None] * (l + 1)
            delta = [None] * (l + 1)
            layers.append(training_set_inputs)
            for iteration in range(l):
                layers.append(self.__sigmoid(
                    dot(layers[iteration], self.synaptic_weights[iteration])))

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            errors[l] = training_set_outputs - layers[l]
            delta[l] = dot(errors[l] *
                           self.__sigmoid_derivative(layers[l]), learning_rate)

            # Print to see the decrease in the error
            if (tr_iter % 10000) == 0:
                print "Error:" + str(mean(abs(errors[l])))

            # Back Propagation
            for iteration in xrange(l - 1, 0, -1):
                errors[iteration] = dot(
                    delta[iteration + 1], self.synaptic_weights[iteration].T)
                delta[iteration] = dot(
                    errors[iteration] * self.__sigmoid_derivative(layers[iteration]), learning_rate)

            # # Adjust the weights.
            for iteration in xrange(l):
                self.synaptic_weights[
                    iteration] += dot(layers[iteration].T, delta[iteration + 1])

    # # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        data = inputs
        for iteration in xrange(self.num_layers):
            data = self.__sigmoid(dot(data, self.synaptic_weights[iteration]))
        return data


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
                         training_set_outputs, 200000, 0.01)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))
