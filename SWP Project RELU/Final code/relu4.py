import tkinter as tk
from tkinter import ttk#enables drop-down lists
from tkinter import messagebox,filedialog 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)#for adding plt in tkinter
import matplotlib.pyplot as plt
import numpy as np
import neural_net as nn
from tkinter import *
import customtkinter
import csv
import model_preprocess as mp
from tkinter import Toplevel#for new window for models
import re
#capture x and y where user clicked


arbitrary_inputs = [0,0,0]#x1,x2,x3 for prediction
layer_sizes = [3,4,1]
# params = [f"IW{i+1}" for i in range(layer_sizes[0]*layer_sizes[1])] + [f"IB{i+1}" for i in range(layer_sizes[1])] + \
# [f"HW{i+1}" for i in range(layer_sizes[1])] + ["OB"]


params = (
    [f"IW{i+1:01d}{j+1:01d}" for i in range(layer_sizes[0]) for j in range(layer_sizes[1])] +
    [f"IB{j+1:01d}" for j in range(layer_sizes[1])] +
    [f"HW{j+1:01d}" for j in range(layer_sizes[1])] +
    ["OB"])


def draw_architecture():
  global info_label
  info_label.config(text="Info: N/A")
  # print(layer_sizes)
  # print(weights)
  # print(weights_list)
  global layer_sizes,weights_list
  print(weights_list)
  print(biases)
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

gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])  #adjust width ratios

results = fig.add_subplot(121, projection='3d')

#extract position of where the user clicks
def on_click(event):
  global callback_id
  if event.inaxes:
    coord_text = results.format_coord(event.xdata, event.ydata)
      
    #extract numeric values from coord_text using regex
    match = re.findall(r"[-+]?\d*\.\d+|\d+", coord_text)
      
    if len(match) >= 2:
      x_raw = float(match[0])
      y_raw = float(match[1])

      # Apply scaling
      x_scaled = (x_raw * 20) - 10
      y_scaled = (y_raw * 20) - 10
      x_scaled *= 1.1
      y_scaled *= 1.1
      print('x: ',x_scaled, '   y: ',y_scaled)
      if x_scaled < 0 or y_scaled < 0:
        arbitrary_inputs[0] = 0 if x_scaled < 0 else x_scaled
        arbitrary_inputs[1] = 0 if y_scaled < 0 else y_scaled
        info_label.config(text="Info: X1 and/or X2 was set to 0.")
      else:
        arbitrary_inputs[0] = x_scaled
        arbitrary_inputs[1] = y_scaled
        info_label.config(text="Info: Valid X1 and X2 provided.")
      arbitrary_inputs[2] = x3_fixed
      print(x3_fixed)
      label_x1.config(text=f"X1: {arbitrary_inputs[0]:.1f}")
      label_x2.config(text=f"X2: {arbitrary_inputs[1]:.1f}")
      print(arbitrary_inputs)
      forward(arbitrary_inputs)

canvas = FigureCanvasTkAgg(fig, master=root)#allows using matplotlib plot in tkinter
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)
callback_id = canvas.mpl_connect("button_press_event",on_click)

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
  bar.set_xlim(-1*np.max(np.abs(weights)), np.max(np.abs(weights)))
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
  bar.set_xlim(-1*np.max(np.abs(flattened_biases)), np.max(np.abs(flattened_biases)))
  bar.axvline(x=0, color='red', linestyle='--', linewidth=1)
  bar.set_xlabel('Biases')
  bar.set_title('Parameter Biases Visualization')
  fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
  canvas.draw()


def update_hidden_layer_neurons():
  info_label.config(text="Info: N/A")
  new_hidden_neurons = hiddendd.get()
  global layer_sizes,weights,weights_list,biases,flattened_biases,params
  layer_sizes[1] = int(new_hidden_neurons)
  weights = np.zeros(layer_sizes[0]*layer_sizes[1]+layer_sizes[1])
  print(weights_list)
  weights_list = construct_weights_from_values(weights,layer_sizes[0],layer_sizes[1],layer_sizes[2])
  print(weights_list)
  biases = {
    layer_sizes[1]: np.zeros(layer_sizes[1]),  #Biases for the hidden layer
    layer_sizes[2] if layer_sizes[1]>1 else 'ob': np.zeros(1)  #Bias for the output layer
  }
  flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'])

  params = (
    [f"IW{i+1:01d}{j+1:01d}" for i in range(layer_sizes[0]) for j in range(layer_sizes[1])] +
    [f"IB{j+1:01d}" for j in range(layer_sizes[1])] +
    [f"HW{j+1:01d}" for j in range(layer_sizes[1])] +
    ["OB"])

#   params = [f"IW{i+1}" for i in range(layer_sizes[0]*layer_sizes[1])] + [f"IB{i+1}" for i in range(layer_sizes[1])] + \
# [f"HW{i+1}" for i in range(layer_sizes[1])] + ["OB"]
  paramdd.configure(values=params)
  paramdd.set(params[0])
  paramdd.event_generate("<<ComboboxSelected>>")
  paramdd.configure(values=params)
  display_weights()
  update_output()
  forward()
  messagebox.showinfo("Information", "All Weights and Biases have been reset to 0 for the new architecture.") 


def importer(model_name = None):
    info_label.config(text="Info: N/A")
    if model_name == None:
      file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
      
      if not file_path:
          messagebox.showerror("Error", "No file selected.")
          return

    global layer_sizes, weights, weights_list, biases, flattened_biases, params

    try:
            if model_name == None:
              with open(file_path, 'r') as file:
                reader = csv.reader(file)
                new_hidden_neurons = int(next(reader)[0])
                layer_sizes[1] = new_hidden_neurons
                values = list(map(float, next(reader)))
            else:#if loaded model is being displayed
               new_hidden_neurons,values = mp.model_weights_loader(model_name)
               layer_sizes[1] = new_hidden_neurons
            # print(new_hidden_neurons,values)

            hidden_weights_count = layer_sizes[0] * layer_sizes[1]
            hidden_bias_count = layer_sizes[1]
            output_weights_count = layer_sizes[1] * layer_sizes[2]

            expected_total = hidden_weights_count + hidden_bias_count + output_weights_count + 1
            if len(values) != expected_total:
                raise ValueError(f"Expected {expected_total} values, but got {len(values)}.")

            hidden_weights = values[:hidden_weights_count]
            hidden_biases = values[hidden_weights_count:hidden_weights_count + hidden_bias_count]
            output_weights = values[hidden_weights_count + hidden_bias_count:hidden_weights_count + hidden_bias_count + output_weights_count]
            output_bias = values[-1]

            weights_list = [
                np.array(hidden_weights).reshape(layer_sizes[0], layer_sizes[1]),
                np.array(output_weights).reshape(layer_sizes[1], layer_sizes[2])
            ]

            global weights,biases
            weights = np.concatenate([weights_list[0].flatten(), weights_list[1].flatten()])
            print(weights)
            biases = {
                layer_sizes[1]: np.array(hidden_biases),#Hidden layer biases
                layer_sizes[2] if layer_sizes[1]>1 else 'ob': np.array([output_bias])#Output layer bias
            }

            flattened_biases = np.append(biases[layer_sizes[1]], biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'])

            params = (
            [f"IW{i+1:01d}{j+1:01d}" for i in range(layer_sizes[0]) for j in range(layer_sizes[1])] +
            [f"IB{j+1:01d}" for j in range(layer_sizes[1])] +
            [f"HW{j+1:01d}" for j in range(layer_sizes[1])] +
            ["OB"])

            paramdd.configure(values=params)
            paramdd.set(params[0])
            paramdd.event_generate("<<ComboboxSelected>>")
            display_weights()
            update_output()
            forward()
            if model_name == None:
               messagebox.showinfo("Success", "Weights and Biases have been updated from the CSV file.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while importing: {str(e)}")


import_button = tk.Button(root, text='Import', bd='5',command= importer)
import_button.place(relx = 0.05, rely = 0.2)

##### Opens new window for selecting the model

def open_model_window():
    # Create a new top-level window
    model_window = Toplevel(root)
    model_window.title("Choose a Model to Display.")
    model_window.geometry("300x200")

    def option_selected(option):
        importer(option)
        model_window.destroy()

    btn1 = tk.Button(model_window, text="Max. Model", command=lambda: option_selected("max"))
    btn1.pack(pady=10)

    btn2 = tk.Button(model_window, text="Min. Model", command=lambda: option_selected("min"))
    btn2.pack(pady=10)

    btn3 = tk.Button(model_window, text="Avg. Model", command=lambda: option_selected("avg"))
    btn3.pack(pady=10)

    btn4 = tk.Button(model_window, text="Select New Model Weights File.", command=lambda: option_selected("other"))
    btn4.pack(pady=10)

pretrained_model_button = tk.Button(root, text="Display Model", bd=5, command=open_model_window)
pretrained_model_button.place(relx=0.05, rely=0.4)

# ##################For prediction on chosen model
# def open_model_predict_window():
#     # Create a new top-level window
#     model_window = Toplevel(root)
#     model_window.title("Choose a Model to Display.")
#     model_window.geometry("300x200")
    
#     def option_selected(option):#values to be sent to model_preprocess for prediction
#         def process_values():
#           x1 = entry_x1.get()
#           x2 = entry_x2.get()
#           x3 = entry_x3.get()
#           if x1 == '':
#              x1 = 0
#           if x2 == '':
#              x2 = 0
#           if x3 == '':
#              x3 = 0
#           x1 = float(x1)
#           x2 = float(x2)
#           x3 = float(x3)
#           res  = mp.predict(option,[x1,x2,x3])
#           text = "Success, "+ "The output of the " + option + ' model for the values ' +  str([x1,x2,x3]) + ' is ' + str(res)
#           messagebox.showinfo('Output',text)

#           model_window.destroy()

#         for widget in model_window.winfo_children():
#           widget.destroy()#remove the buttons
#         tk.Label(model_window, text="X1:").pack()
#         entry_x1 = tk.Entry(model_window)
#         entry_x1.pack(pady=5)

#         tk.Label(model_window, text="X2:").pack()
#         entry_x2 = tk.Entry(model_window)
#         entry_x2.pack(pady=5)

#         tk.Label(model_window, text="X3:").pack()
#         entry_x3 = tk.Entry(model_window)
#         entry_x3.pack(pady=5)
        
        
#         process_btn = tk.Button(model_window, text="Process Values", command=process_values)
#         process_btn.pack(pady=10)

        

#     btn1 = tk.Button(model_window, text="Max. Model", command=lambda: option_selected("max"))
#     btn1.pack(pady=10)

#     btn2 = tk.Button(model_window, text="Min. Model", command=lambda: option_selected("min"))
#     btn2.pack(pady=10)

#     btn3 = tk.Button(model_window, text="Avg. Model", command=lambda: option_selected("avg"))
#     btn3.pack(pady=10)

# pretrained_model_predict_button = tk.Button(root, text="Test Model", bd=5, command=open_model_predict_window)
# pretrained_model_predict_button.place(relx=0.05, rely=0.5)


def exporter():
  info_label.config(text="Info: N/A")
  file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
  if not file_path:
      messagebox.showerror("Error", "No file selected.")
      return
  global layer_sizes, weights_list, biases
  hidden_weights = weights_list[0].flatten()
  output_weights = weights_list[1].flatten()
  hidden_biases = biases[layer_sizes[1]]
  output_bias = biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'][0]

  all_values = list(hidden_weights) + list(hidden_biases) + list(output_weights) + [output_bias]

  with open(file_path, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([layer_sizes[1]])
      writer.writerow(all_values)

export_button = tk.Button(root, text='Export', bd='5',command= exporter)
export_button.place(relx = 0.05, rely = 0.3)


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
    info_label.config(text="Info: N/A")
    global weights,biases,flattened_biases,weights_list,layer_sizes
    selected_param = paramdd.get()
    hidden_idx = None
    new_value = slider.get()
    if selected_param.startswith('IW'):
      input_idx = int(selected_param[2:3])-1#changed
      if len(selected_param) == 5:#special case for 10 hidden neurons e.g IW310, input 3 hidden 10 makes length 5 instead of 4
        hidden_idx = 9#changed
      else:
         hidden_idx = int(selected_param[-1])-1
      flat_index = input_idx * layer_sizes[1] + hidden_idx
      weights[flat_index] = new_value
      # weights[int(selected_param[2:])-1] = new_value
      weights_list = construct_weights_from_values(weights,layer_sizes[0],layer_sizes[1],layer_sizes[2])
      display_weights()

    elif selected_param.startswith('HW'):
      weights[int(selected_param[2:])-1 + layer_sizes[0]*layer_sizes[1]] = new_value  
      weights_list = construct_weights_from_values(weights,layer_sizes[0],layer_sizes[1],layer_sizes[2])
      display_weights()

    elif selected_param.startswith('IB'):
      biases[layer_sizes[1]][int(selected_param[2:])-1] = new_value
      flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'])
      display_biases()

    else:
      biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'][0] = new_value 
      flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'])
      display_biases()
    update_output()
    forward()
#slider
def slider_event(value):
    slider_val_label.configure(text = "{:.1f}".format(value))

def update_slider_value(event):
    global weights, biases, flattened_biases, layer_sizes
    selected_param = paramdd.get()
    hiddden_idx = None
    if selected_param.startswith('IW'):
      input_idx = int(selected_param[2:3])-1#changed
      if len(selected_param) == 5:#special case for 10 hidden neurons e.g IW310, input 3 hidden 10 makes length 5 instead of 4
        hidden_idx = 9#changed
      else:
         hidden_idx = int(selected_param[-1])-1
      flat_index = input_idx * layer_sizes[1] + hidden_idx
      new_value = weights[flat_index]
        # new_value = weights[int(selected_param[2:]) - 1]
    elif selected_param.startswith('HW'):
        new_value = weights[int(selected_param[2:])-1 + layer_sizes[0] * layer_sizes[1]]
    elif selected_param.startswith('IB'):
        new_value = biases[layer_sizes[1]][int(selected_param[2:])-1]
    else:  
        new_value = biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'][0]

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

#added comment to test
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
    layer_sizes[2] if layer_sizes[1]>1 else 'ob': np.zeros(1)  # Bias for the output layer
}
flattened_biases = np.append(biases[layer_sizes[1]],biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'])
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


#forward pass / prediction function
def forward(ab = None):
  x1,x2,x3 = None,None,None
  if ab:
    x1 = ab[0]
    x2 = ab[1]
    x3 = ab[2]
  else:
     x1 = arbitrary_inputs[0]
     x2 = arbitrary_inputs[1]
     x3 = arbitrary_inputs[2]
  
  inputs = np.array([[x1, x2, x3]])
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
  
  print("IN FORWARD")
  print(x1,x2,x3)
  print(weights1,weights2)
  print(biases1,biases2)
  biases2[0] = biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'][0]
  output = relu_network(inputs, weights1, biases1, weights2, biases2)
  output_label.config(text=f"Output: {output[0][0]:.4f}")

label_x1 = tk.Label(root, text="X1",background='yellow')
label_x1.place(relx=0.05, rely=0.5)
label_x1.config(text=f"X1: {arbitrary_inputs[0]:.1f}")

label_x2 = tk.Label(root, text="X2",background='yellow')
label_x2.place(relx=0.05, rely=0.54)
label_x2.config(text=f"X2: {arbitrary_inputs[1]:.1f}")

output_label = tk.Label(root, text="Output: ",background='yellow')
output_label.place(relx=0.05, rely=0.75)

info_label = tk.Label(root, text="Info: N/A",background='yellow')
info_label.place(relx=0.05, rely=0.8)



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
  
  biases2[0] = biases[layer_sizes[2] if layer_sizes[1]>1 else 'ob'][0]
  output = relu_network(inputs, weights1, biases1, weights2, biases2).reshape(x1_grid.shape)
  results.plot_surface(x1_grid, x2_grid, output, cmap="viridis", alpha=0.8)
  results.set_xlabel("x1")
  results.set_ylabel("x2")
  results.set_zlabel("Output")
  results.set_zlim(0, 20)
  canvas.draw()

x3_var = tk.DoubleVar(value=0)

def update_x3_value(value=None):
    global inputs, x1_grid, x2_grid,x3_fixed
    if value is not None:
        x3_fixed = float(value)
        x3_value_label.config(text=f"x3: {x3_fixed:.1f}")#Update the x3 value label
    else:
        x3_fixed = x3_var.get()
    arbitrary_inputs[2] = x3_fixed
    inputs = np.c_[x1_grid.ravel(), x2_grid.ravel(), np.full_like(x1_grid.ravel(), x3_fixed)]
    display_weights()
    update_output()
    forward()
x3_slider = customtkinter.CTkSlider(master=root, from_=-10, to=10, command=lambda value: update_x3_value(value))
x3_slider.set(0)
x3_slider.place(relx=0.05, rely=0.66)

x3_value_label = tk.Label(root, text="x3: 0.0", background='yellow')
x3_value_label.place(relx=0.05, rely=0.62)



update_params()
if __name__ == '__main__':
  root.mainloop()
  print("\nEnded Successfully!")
  exit(0)
