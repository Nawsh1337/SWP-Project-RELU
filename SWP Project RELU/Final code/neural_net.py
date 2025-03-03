#File runs when Draw Architecture button is clicked and it displays the model architecture
from matplotlib import pyplot
from math import cos, sin, atan
from palettable.tableau import Tableau_10
from time import localtime, strftime
import numpy as np

class Neuron():
    def __init__(self, x, y, bias=None):
        self.x = x
        self.y = y
        self.bias = bias  # Store the bias value

    def draw(self, neuron_radius, id=-1):
        # Draw the neuron circle
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        # pyplot.gca().text(self.x, self.y - 0.15, str(id), size=10, ha='center')

        # If bias is provided, draw the bias inside the neuron
        if self.bias is not None:
            pyplot.gca().text(self.x, self.y, "{:.2f}".format(self.bias), size=10, ha='center', va='center')


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, biases=None):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, biases)

    def __intialise_neurons(self, number_of_neurons, biases):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            # If biases are provided, use them, otherwise set to None
            bias = biases[iteration] if biases is not None else None
            neuron = Neuron(x, self.y, bias)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight=0.4, textoverlaphandler=None):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        # assign colors to lines depending on the sign of the weight
        color = Tableau_10.mpl_colors[7]
        
        if abs(weight) > 0.2:
            color = Tableau_10.mpl_colors[3]

        # assign different linewidths to lines depending on the size of the weight
        abs_weight = abs(weight)
        linewidth = 2

        # draw the weights and adjust the labels of weights to avoid overlapping
        if abs_weight > 0.0:
            # while loop to determine the optimal locaton for text labels to avoid overlapping
            index_step = 2
            num_segments = 10
            txt_x_pos = neuron1.x - x_adjustment + index_step * (neuron2.x - neuron1.x + 2 * x_adjustment) / num_segments
            txt_y_pos = neuron1.y - y_adjustment + index_step * (neuron2.y - neuron1.y + 2 * y_adjustment) / num_segments
            while ((not textoverlaphandler.getspace([txt_x_pos - 0.5, txt_y_pos - 0.5, txt_x_pos + 0.5, txt_y_pos + 0.5])) and index_step < num_segments):
                index_step = index_step + 1
                txt_x_pos = neuron1.x - x_adjustment + index_step * (neuron2.x - neuron1.x + 2 * x_adjustment) / num_segments
                txt_y_pos = neuron1.y - y_adjustment + index_step * (neuron2.y - neuron1.y + 2 * y_adjustment) / num_segments

            a = pyplot.gca().text(txt_x_pos, txt_y_pos, "{:3.2f}".format(weight), size=8, ha='center')
            a.set_bbox(dict(facecolor='white', alpha=0))

        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=linewidth, color=color)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0, weights=None, textoverlaphandler=None):
        j = 0  # index for neurons in this layer
        for neuron in self.neurons:
            i = 0  # index for neurons in previous layer
            neuron.draw(self.neuron_radius, id=j + 1)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weights[i][j], textoverlaphandler)
                    i = i + 1
            j = j + 1

        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer ' + str(layerType), fontsize=12)


class TextOverlappingHandler():
    def __init__(self, width, height, grid_size=0.2):
        self.grid_size = grid_size
        self.cells = np.ones((int(np.ceil(width / grid_size)), int(np.ceil(height / grid_size))), dtype=bool)

    def getspace(self, test_coordinates):
        x_left_pos = int(np.floor(test_coordinates[0] / self.grid_size))
        y_botttom_pos = int(np.floor(test_coordinates[1] / self.grid_size))
        x_right_pos = int(np.floor(test_coordinates[2] / self.grid_size))
        y_top_pos = int(np.floor(test_coordinates[3] / self.grid_size))
        if self.cells[x_left_pos, y_botttom_pos] and self.cells[x_left_pos, y_top_pos] \
                and self.cells[x_right_pos, y_top_pos] and self.cells[x_right_pos, y_botttom_pos]:
            for i in range(x_left_pos, x_right_pos):
                for j in range(y_botttom_pos, y_top_pos):
                    self.cells[i, j] = False
            return True
        else:
            return False


class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, biases=None):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0
        self.biases = biases or {}

    def add_layer(self, number_of_neurons, biases=None):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, biases)
        self.layers.append(layer)

    def draw(self, weights_list=None):
        vertical_distance_between_layers = 6
        horizontal_distance_between_neurons = 2
        overlaphandler = TextOverlappingHandler(
            self.number_of_neurons_in_widest_layer * horizontal_distance_between_neurons,
            len(self.layers) * vertical_distance_between_layers, grid_size=0.2)

        pyplot.figure(figsize=(12, 9))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.draw(layerType=0)
            elif i == len(self.layers) - 1:
                layer.draw(layerType=-1, weights=weights_list[i - 1], textoverlaphandler=overlaphandler)
            else:
                layer.draw(layerType=i, weights=weights_list[i - 1], textoverlaphandler=overlaphandler)

        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        pyplot.show()


class DrawNN():
    def __init__(self, neural_network, weights_list=None, biases=None):
        self.neural_network = neural_network
        self.weights_list = weights_list
        self.biases = biases

        # If weights_list is none, create a default one
        if weights_list is None:
            weights_list = []
            for first, second in zip(neural_network, neural_network[1:]):
                tempArr = np.ones((first, second)) * 0.4
                weights_list.append(tempArr)
            self.weights_list = weights_list

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer, self.biases)
        print(self.biases)
        if self.neural_network == [3,3,1]:
            for i,l in enumerate(self.neural_network):
                if i==0:
                    network.add_layer(l, [None,None,None])
                else:
                    network.add_layer(l, self.biases.get(l, None))
        elif self.neural_network == [3,1,1]:
            for i,l in enumerate(self.neural_network):
                if i==0:
                    network.add_layer(l, self.biases.get(l, None))
                elif i == 1:
                    network.add_layer(l, self.biases.get(l, None))
                elif i == 2:
                    network.add_layer(l, self.biases.get('ob', None))
        else:
            for l in self.neural_network:
                network.add_layer(l, self.biases.get(l, None))
        network.draw(self.weights_list)


