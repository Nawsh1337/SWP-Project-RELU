import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons

def relu(x):
    return np.maximum(0, x)
def relu_network(x, weights1, biases1, weights2, biases2):#calculate output
    hidden_input = np.dot(x, weights1) + biases1
    hidden_output = relu(hidden_input)
    final_input = np.dot(hidden_output, weights2) + biases2
    return relu(final_input)
x1 = np.arange(-10, 11, 1)
x2 = np.arange(-10, 11, 1)
x1_grid, x2_grid = np.meshgrid(x1, x2)
inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]

weights1 = np.zeros((2, 4))
biases1 = np.zeros(4)
weights2 = np.zeros((4, 1))
biases2 = np.zeros(1)

print(biases2)


parameter_names = [f"IW{i+1}" for i in range(8)] + [f"IB{i+1}" for i in range(4)] + \
                  [f"HW{i+1}" for i in range(4)] + ["OB"]

print(parameter_names)
#IW1,IW2....,IW8,IB1,IB2...,IB4,HW1,HW2.....,HW4,OB
selected_param = None
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
def update_plot():#updates the plot
    global weights1, biases1, weights2, biases2
    output = relu_network(inputs, weights1, biases1, weights2, biases2).reshape(x1_grid.shape)
    ax.clear()
    ax.plot_surface(x1_grid, x2_grid, output, cmap="viridis", alpha=0.8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Output")
    ax.set_zlim(0, 20)
    update_values_display()
    plt.draw()
def apply_change(event):#when apply is clicked
    global weights1, biases1, weights2, biases2
    if selected_param is None:
        print("No parameter selected.")
        return
    try:
        new_value = float(textbox_value.text)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return
    if selected_param.startswith("IW"):
        index = int(selected_param[2:]) - 1
        weights1.flat[index] = new_value
    elif selected_param.startswith("IB"):
        index = int(selected_param[2:]) - 1
        biases1[index] = new_value
    elif selected_param.startswith("HW"):
        index = int(selected_param[2:]) - 1
        weights2.flat[index] = new_value
    elif selected_param == "OB":
        biases2[0] = new_value
    update_plot()

def radio_callback(label):#gets value of selected radio button
    global selected_param
    selected_param = label

def update_values_display():#for the box on the right
    global weights1, biases1, weights2, biases2
    values = []
    values += [f"{name}: {value:.2f}" for name, value in zip(parameter_names[:8], weights1.flatten())]
    values += [f"{name}: {value:.2f}" for name, value in zip(parameter_names[8:12], biases1)]
    values += [f"{name}: {value:.2f}" for name, value in zip(parameter_names[12:16], weights2.flatten())]
    values.append(f"{parameter_names[16]}: {biases2[0]:.2f}")
    values_display.set_val("\n".join(values))

radio_ax = plt.axes([0.02, 0.2, 0.2, 0.7], facecolor='lightgrey')
radio_buttons = RadioButtons(radio_ax, parameter_names)
radio_buttons.on_clicked(radio_callback)

textbox_ax = plt.axes([0.3, 0.9, 0.2, 0.05])
textbox_value = TextBox(textbox_ax, "Value", initial="0")

button_ax = plt.axes([0.55, 0.9, 0.1, 0.05])
apply_button = Button(button_ax, "Apply")
apply_button.on_clicked(apply_change)

values_display_ax = plt.axes([0.8, 0.2, 0.18, 0.7], facecolor='white')
values_display = TextBox(values_display_ax, "Weights & Biases", initial="")
values_display.set_active(False)

update_plot()
plt.show()
