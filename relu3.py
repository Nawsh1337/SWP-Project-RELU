import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D

def relu(x):
    return np.maximum(0, x)

def relu_network(x, weights1, biases1, weights2, biases2):
    hidden_input = np.dot(x, weights1) + biases1
    hidden_output = relu(hidden_input)
    final_input = np.dot(hidden_output, weights2) + biases2
    return relu(final_input)

# Generate input data
x1 = np.arange(-10, 11, 1)
x2 = np.arange(-10, 11, 1)
x1_grid, x2_grid = np.meshgrid(x1, x2)
inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Initialize weights and biases for the layers
initial_weights1 = np.zeros((2, 4))
initial_biases1 = np.zeros(4)
initial_weights2 = np.zeros((4, 1))
initial_biases2 = np.zeros(1)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Update plot based on weights and biases
def update_plot(weights1, biases1, weights2, biases2):
    weights1 = np.array(weights1).reshape(2, 4)
    biases1 = np.array(biases1).reshape(4)
    weights2 = np.array(weights2).reshape(4, 1)
    biases2 = np.array(biases2).reshape(1)

    output = relu_network(inputs, weights1, biases1, weights2, biases2).reshape(x1_grid.shape)
    ax.clear()
    ax.plot_surface(x1_grid, x2_grid, output, cmap="viridis", alpha=0.8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Output")
    ax.set_zlim(0, 20)
    plt.draw()

def parse_textbox_values():
    # Parse values from text boxes for each weight and bias
    weights1 = [float(w1_tb[i].text) for i in range(8)]
    biases1 = [float(b1_tb[i].text) for i in range(4)]
    weights2 = [float(w2_tb[i].text) for i in range(4)]
    biases2 = [float(b2_tb.text)]
    update_plot(weights1, biases1, weights2, biases2)

# Initial plot update
update_plot(initial_weights1.flatten(), initial_biases1, initial_weights2.flatten(), initial_biases2)

# Define positions for text boxes
textbox_positions = {
    "w1": [(0.05 + 0.1 * i, 0.9, 0.05, 0.03) for i in range(8)],
    "b1": [(0.05 + 0.1 * i, 0.85, 0.05, 0.03) for i in range(4)],
    "w2": [(0.05 + 0.1 * i, 0.8, 0.05, 0.03) for i in range(4)],
    "b2": [(0.05, 0.75, 0.05, 0.03)],
}

# Create text boxes for weights and biases
w1_tb = [TextBox(plt.axes(pos), f"w1_{i}", initial=str(0)) for i, pos in enumerate(textbox_positions["w1"])]
b1_tb = [TextBox(plt.axes(pos), f"b1_{i}", initial=str(0)) for i, pos in enumerate(textbox_positions["b1"])]
w2_tb = [TextBox(plt.axes(pos), f"w2_{i}", initial=str(0)) for i, pos in enumerate(textbox_positions["w2"])]
b2_tb = TextBox(plt.axes(textbox_positions["b2"][0]), "b2", initial=str(0))

# Connect text boxes to the update function
for tb in w1_tb + b1_tb + w2_tb + [b2_tb]:
    tb.on_submit(lambda _: parse_textbox_values())

plt.show()
