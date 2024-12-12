import tkinter as tk
from tkinter import ttk # enables drop-down lists
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)#for adding plt in tkinter
import matplotlib.pyplot as plt
import numpy as np
import neural_net as nn
from tkinter import *
import customtkinter

def draw_architecture():
  network = nn.DrawNN(layer_sizes,weights_list=weights_list, biases=biases)
  network.draw()

root = tk.Tk()
root.geometry("1920x1080")
root.title("Relu Visualizer")
frame = tk.Frame(root)
# label = tk.Label(text="Result               +       Network Structure!")
# label.config(font=("Courier", 32))
# label.pack()
frame.pack()

layer_sizes = [3,5,1]

plt.ion()
fig = plt.Figure()
results = fig.add_subplot(121, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)#allows using matplotlib plot in tkinter
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# Visualize Architecure Button
vizb = tk.Button(root, text='Draw Architecture', bd='5',command=draw_architecture)
vizb.place(relx = 0.901, rely = 0.88)

#param dropdown
params = [f"IW{i+1}" for i in range(layer_sizes[0]*layer_sizes[1])] + [f"IB{i+1}" for i in range(layer_sizes[1])] + \
                  [f"HW{i+1}" for i in range(layer_sizes[1])] + ["OB"]
paramdd = ttk.Combobox(state="readonly", values=params)
paramdd.set(params[0])
paramdd.place(relx = 0.1, rely = 0.89)
param_select_label = tk.Label(root, text="Select Weight/Bias to Edit")
param_select_label.place(relx = 0.1, rely = 0.87)

#slider
def slider_event(value):
    slider_val_label.configure(text = "{:.1f}".format(value))

slider_val_label = tk.Label(root, text="")
slider_val_label.place(relx = 0.25, rely = 0.89)    

slider = customtkinter.CTkSlider(master=root, from_=-10, to=10,command = slider_event,fg_color = 'blue')
slider.set(0)
slider.place(relx=0.2, rely=0.88)







def construct_weights_from_values(weight_values, input_nodes=layer_sizes[0], hidden_nodes=layer_sizes[1], output_nodes=layer_sizes[2]):
    num_input_to_hidden = input_nodes * hidden_nodes 
    num_hidden_to_output = hidden_nodes * output_nodes 
    input_to_hidden_weights = np.array(weight_values[:num_input_to_hidden]).reshape((input_nodes, hidden_nodes))
    hidden_to_output_weights = np.array(weight_values[num_input_to_hidden:num_input_to_hidden + num_hidden_to_output]).reshape((hidden_nodes, output_nodes))
    weights_list = [input_to_hidden_weights, hidden_to_output_weights]
    return weights_list

weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16,17,18,19,20]
weights_list = construct_weights_from_values(weights)
biases = {
    layer_sizes[1]: [0.3, -0.1, 0.4, 0.2,0.5],  # Biases for the hidden layer (4 neurons)
    layer_sizes[2]: [0.1]  # Bias for the output layer (1 neuron)
}

















if __name__ == '__main__':
  root.mainloop()
  print("\nEnded Successfully!")
  exit(0)
