import tkinter as tk
from tkinter import ttk#enables drop-down lists
from tkinter import messagebox 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)#for adding plt in tkinter
import matplotlib.pyplot as plt
import numpy as np
import neural_net as nn
from tkinter import *
import customtkinter

layer_sizes = [3,4,1]
params = [f"IW{i+1}" for i in range(layer_sizes[0]*layer_sizes[1])] + [f"IB{i+1}" for i in range(layer_sizes[1])] + \
[f"HW{i+1}" for i in range(layer_sizes[1])] + ["OB"]

def draw_architecture():
  # print(layer_sizes)
  # print(weights)
  # print(weights_list)
  global layer_sizes,weights_list
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



plt.ion()
fig = plt.Figure()

gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])  # Adjust width ratios

results = fig.add_subplot(121, projection='3d')

canvas = FigureCanvasTkAgg(fig, master=root)#allows using matplotlib plot in tkinter
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)

def display_weights():
  global fig,params,weights_list,weights
  params_without_biases = [param for param in params if not param.startswith('IB') and not param == 'OB']
  if len(fig.axes) > 1:
      fig.delaxes(fig.axes[1])
  bar = fig.add_subplot(gs[1])
  y_pos = np.arange(len(params_without_biases))
  bar.barh(y_pos, np.array(weights), align='center')
  bar.set_yticks(y_pos, labels=params_without_biases)
  bar.invert_yaxis()
  bar.set_xlim(-10, 10)
  bar.axvline(x=0, color='red', linestyle='--', linewidth=1)
  bar.set_xlabel('Weights')
  bar.set_title('Parameter Weights Visualization')
  fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
  canvas.draw()

def display_biases():
  global fig,params,flattened_biases
  biasparams = [param for param in params if param.startswith('IB') or param == 'OB']
  if len(fig.axes) > 1:
      fig.delaxes(fig.axes[1])
  bar = fig.add_subplot(gs[1])
  y_pos = np.arange(len(biasparams))
  # print(y_pos)
  # print(np.array(flattened_biases))
  bar.barh(y_pos, np.array(flattened_biases), align='center')
  bar.set_yticks(y_pos, labels=biasparams)
  bar.invert_yaxis()
  bar.set_xlim(-10, 10)
  bar.axvline(x=0, color='red', linestyle='--', linewidth=1)
  bar.set_xlabel('Biases')
  bar.set_title('Parameter Biases Visualization')
  fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
  canvas.draw()


def update_hidden_layer_neurons():
  new_hidden_neurons = hiddendd.get()
  global layer_sizes,weights,weights_list,biases,flattened_biases,params
  layer_sizes[1] = int(new_hidden_neurons)
  weights = np.zeros(layer_sizes[0]*layer_sizes[1]+layer_sizes[1])
  print(weights_list)
  weights_list = construct_weights_from_values(weights,layer_sizes[0],layer_sizes[1],layer_sizes[2])
  print(weights_list)
  biases = {
    layer_sizes[1]: np.zeros(layer_sizes[1]),  # Biases for the hidden layer
    layer_sizes[2]: np.zeros(1)  # Bias for the output layer
  }
  flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2]])
  params = [f"IW{i+1}" for i in range(layer_sizes[0]*layer_sizes[1])] + [f"IB{i+1}" for i in range(layer_sizes[1])] + \
[f"HW{i+1}" for i in range(layer_sizes[1])] + ["OB"]
  paramdd.configure(values=params)
  display_weights()
  update_output()
  messagebox.showinfo("Information", "All Weights and Biases have been reset to 0 for the new architecture.") 


def on_enter(e):
  e.widget['background'] = 'white'

def on_leave(e):
  global orig_color
  e.widget['background'] = orig_color

#Hidden Layer Neurons Selection

hiddendd = ttk.Combobox(state="readonly", values=[x+1 for x in range(10)])
hiddendd.set(layer_sizes[1])
hiddendd.place(relx = 0.45, rely = 0.89)
hiddendd_label = tk.Label(root, text="Hidden Layer Neurons",background='yellow')
hiddendd_label.place(relx = 0.45, rely = 0.87)
# Hidden neurons set Button
hiddenset_button = tk.Button(root, text='Apply Changes', bd='5',command=update_hidden_layer_neurons)
hiddenset_button.place(relx = 0.55, rely = 0.88)
hiddenset_button.bind('<Enter>',on_enter)
hiddenset_button.bind('<Leave>',on_leave)



# Visualize Architecure Button
vizb = tk.Button(root, text='Draw Architecture', bd='5',command=draw_architecture,highlightbackground='black',highlightthickness=2)
vizb.place(relx = 0.1, rely = 0.1)
orig_color = vizb.cget("background")
vizb.bind('<Enter>',on_enter)
vizb.bind('<Leave>',on_leave)
#param dropdown

paramdd = ttk.Combobox(state="readonly", values=params)
paramdd.set(params[0])
paramdd.place(relx = 0.1, rely = 0.89)
param_select_label = tk.Label(root, text="Select Weight/Bias to Edit",background = 'yellow')
param_select_label.place(relx = 0.1, rely = 0.87)

def update_params():
    global weights,biases,flattened_biases,weights_list,layer_sizes
    selected_param = paramdd.get()
    new_value = slider.get()
    if selected_param.startswith('IW'):
      weights[int(selected_param[2:])-1] = new_value
      weights_list = construct_weights_from_values(weights,layer_sizes[0],layer_sizes[1],layer_sizes[2])
      display_weights()

    elif selected_param.startswith('HW'):
      weights[int(selected_param[2:]) + layer_sizes[0]*layer_sizes[1]-1] = new_value  
      weights_list = construct_weights_from_values(weights,layer_sizes[0],layer_sizes[1],layer_sizes[2])
      display_weights()

    elif selected_param.startswith('IB'):
      biases[layer_sizes[1]][int(selected_param[2:])-1] = new_value
      flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2]])
      display_biases()

    else:
      biases[layer_sizes[2]][0] = new_value 
      flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2]])
      display_biases()
    update_output()

#slider
def slider_event(value):
    slider_val_label.configure(text = "{:.1f}".format(value))

def update_slider_value(event):
    global weights, biases, flattened_biases, layer_sizes
    selected_param = paramdd.get()
    if selected_param.startswith('IW'):
        new_value = weights[int(selected_param[2:]) - 1]
    elif selected_param.startswith('HW'):
        new_value = weights[int(selected_param[2:]) + layer_sizes[0] * layer_sizes[1] - 1]
    elif selected_param.startswith('IB'):
        new_value = biases[layer_sizes[1]][int(selected_param[2:]) - 1]
    else:  
        new_value = biases[layer_sizes[2]][0]

    slider.set(new_value)
    slider_val_label.config(text=f"{new_value:.1f}")

paramdd.bind("<<ComboboxSelected>>", update_slider_value)


slider = customtkinter.CTkSlider(master=root, from_=-10, to=10,command = slider_event,fg_color = 'blue')
slider.set(0)
slider.place(relx=0.2, rely=0.88)
sliderconfirm_button = tk.Button(root, text='Update Weight/Bias Value', bd='5',command=update_params)
sliderconfirm_button.place(relx = 0.32, rely = 0.88)

sliderconfirm_button.bind('<Enter>',on_enter)
sliderconfirm_button.bind('<Leave>',on_leave)

slider_val_label = tk.Label(root, text=slider.get())
slider_val_label.place(relx = 0.25, rely = 0.9)    


def construct_weights_from_values(weight_values, input_nodes=layer_sizes[0], hidden_nodes=layer_sizes[1], output_nodes=layer_sizes[2]):
    num_input_to_hidden = input_nodes * hidden_nodes 
    num_hidden_to_output = hidden_nodes * output_nodes 
    input_to_hidden_weights = np.array(weight_values[:num_input_to_hidden]).reshape((input_nodes, hidden_nodes))
    hidden_to_output_weights = np.array(weight_values[num_input_to_hidden:num_input_to_hidden + num_hidden_to_output]).reshape((hidden_nodes, output_nodes))
    weights_list = [input_to_hidden_weights, hidden_to_output_weights]
    return weights_list

weights = np.zeros(layer_sizes[0]*layer_sizes[1]+layer_sizes[1])
weights_list = construct_weights_from_values(weights)
biases = {
    layer_sizes[1]: np.zeros(layer_sizes[1]),  # Biases for the hidden layer
    layer_sizes[2]: np.zeros(1)  # Bias for the output layer
}
flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2]])
# print(flattened_biases)




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
# inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]
x3_fixed = 0
inputs = np.c_[x1_grid.ravel(), x2_grid.ravel(), np.full_like(x1_grid.ravel(), x3_fixed)]


def update_output():
  if len(fig.axes) > 1:
      fig.delaxes(fig.axes[0])
  results = fig.add_subplot(121, projection='3d')
  global layer_sizes
  weights1 = np.zeros((layer_sizes[0], layer_sizes[1]))
  biases1 = np.zeros(layer_sizes[1])
  weights2 = np.zeros((layer_sizes[1], 1))
  biases2 = np.zeros(1)

  for x in range(layer_sizes[0]*layer_sizes[1]):#0,1,2,3,4,5,6,7
    print(x,layer_sizes[0],layer_sizes[1])
    #  weights1[int(x/layer_sizes[1])][x if x < layer_sizes[1] else x-layer_sizes[1]] = weights[x]
    row, col = divmod(x, layer_sizes[1])
    weights1[row][col] = weights[x]

  
  for x in range(layer_sizes[1]):#8,9,10,11
     weights2[x][0] = weights[x+layer_sizes[0]*layer_sizes[1]]
  
  for x in range(layer_sizes[1]):
     biases1[x] = biases[layer_sizes[1]][x]
  
  biases2[0] = biases[layer_sizes[2]][0]
  output = relu_network(inputs, weights1, biases1, weights2, biases2).reshape(x1_grid.shape)
  results.plot_surface(x1_grid, x2_grid, output, cmap="viridis", alpha=0.8)
  results.set_xlabel("x1")
  results.set_ylabel("x2")
  results.set_zlabel("Output")
  results.set_zlim(0, 20)
  canvas.draw()

x3_values = [0, 5, 10, 15, 20]
x3_var = tk.DoubleVar(value=x3_values[0])

def update_x3_value(*args):
    global inputs, x1_grid, x2_grid
    x3_fixed = x3_var.get()
    inputs = np.c_[x1_grid.ravel(), x2_grid.ravel(), np.full_like(x1_grid.ravel(), x3_fixed)]
    update_output()

x3_menu = tk.OptionMenu(root, x3_var, *x3_values, command=lambda _: update_x3_value())
x3_menu.place(relx=0.63, rely=0.91)
x3_menu.config(width=10)

x3_menu_label = tk.Label(root, text="Value of x3",background='yellow')
x3_menu_label.place(relx = 0.63, rely = 0.88)

if __name__ == '__main__':
  root.mainloop()
  print("\nEnded Successfully!")
  exit(0)
