from functions import *


def neuron_values(neurons: list) -> list:
    output = []

    for neuron in neurons:
        output.append(neuron.value)
    return output


class NeuralNetwork:
    neurons = []  # list of lists of Neuron objects
    learn_rate = 0.1  # value multiplies by derivatives of some functions

    def __init__(self, number_of_layers: int, number_of_neurons: list):
        # default constructor create a neuron network with 4 layers: 784, 16, 16, 10 neurons in layer
        for i in range(0, number_of_layers):  # create layers for exampe 4 empty list
            self.neurons.append([])

        def layer_init(index, num_of_neurons, number_of_previous_layer_neurons):
            for i in range(0, num_of_neurons):  # create layer of neurons with random biases and random weights
                self.neurons[index].append(Neuron(uniform(0, 0.1),
                                                  create_random_list(number_of_previous_layer_neurons, -1, 1)))

        for i in range(0, number_of_neurons[0]):  # create first layer, every neuron in this layer shouldn't have weights
            self.neurons[0].append(Neuron(0, []))  # and don't need bias because it's input layer

        for i in range(1, number_of_layers):
            layer_init(i, number_of_neurons[i], number_of_neurons[i-1])  # create hidden and output layers

    def fill_first_layer(self, data):
        for neuron, value in zip(self.neurons[0], data):
            neuron.value = value

    def clear_network(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.clear()

    def run(self):
        first = True
        prev = self.neurons[0]

        for layer in self.neurons:
            if first:
                first = False  # first layer doesn't need calculate because it's an input layer
                continue

            for neuron in layer:
                neuron.calculate(prev)
            prev = layer

    def show_output(self):  # print value of each neuron in the last layer
        for i, neuron in enumerate(self.neurons[-1]):
            print(str(i) + ": " + str(neuron.value))

    def cost(self, desired_output: list) -> float:  # function which gives the rate how good is the network
        output = 0
        for desired_value, output_neuron in zip(desired_output, self.neurons[-1]):
            output += (desired_value - output_neuron.value) * (desired_value - output_neuron.value)

        return output

    def derivative_of_cost(self, desired_output: float, neuron) -> float:  # function needed to calculate weights etc
        # in backprop
        return 2*(desired_output - neuron.value)

    def backpropagation(self, desired_output: list) -> None:
        for index, layer in reversed(list(enumerate(self.neurons))):  # we start from the end and go to the first layer
            if index == 0:  # first layer ends the backpropagation algorithm
                break

            next_desired_output = neuron_values(self.neurons[index - 1])  # at the beginning set next desired values
            # list

            for num, neuron in enumerate(layer):  # for each neurons in this layer change bias and every
                # weights and value of neurons in previous layer

                cost = self.derivative_of_cost(desired_output[num], neuron)

                neuron.bias += 1 * derivative_of_ReLU(neuron.z) * cost * self.learn_rate
                # we don't care about good order because we don't change values of neurons
                # and we don't change value of some weight twice so output stay the same

                for i, (weight, previous_neuron) in enumerate(zip(neuron.weights, self.neurons[index-1])):
                    weight += previous_neuron.value * cost * derivative_of_ReLU(previous_neuron.z) * self.learn_rate

                    if index != 1:  # second layer can't change value of previous layer
                        next_desired_output[i] += weight * cost * derivative_of_ReLU(previous_neuron.z) * self.learn_rate

            desired_output = next_desired_output  # at the end of the layer swap this two list now we take care of the
            # next layer


class Neuron:
    value = 0
    z = 0  # value of sum of every neurons from previous layer multiples by weights (before ReLU)
    bias = 0
    weights = []  # list of numbers represents weights connected to this neuron (from the left)

    def __init__(self, bias, weights):
        self.value = 0
        self.bias = bias
        self.weights = weights
        self.z = 0

    def clear(self):
        self.value = 0
        self.z = 0

    def calculate(self, previous_layer):
        for element, weight in zip(previous_layer, self.weights):
            self.z += element.value * weight

        self.value = dying_ReLU(self.z - self.bias)
