#a little bit of logic for loading pretrained models
import os
import numpy as np
import torch
import torch.nn as nn
from tkinter import filedialog,messagebox
class ReLU_MaxNN2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReLU_MaxNN2, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden_out = self.relu(self.hidden(x))
        output = self.output(hidden_out)
        return output

input_size = 3
hidden_size = None

path = 'SWP-Project-RELU\SWP Project RELU\Final code\Functions\Models'

# print(os.listdir(path))

#user will select the model and it will be loaded.

params  = None
model_text = ''
def model_weights_loader(model = 'max'):
    if model == 'max':
        params = np.load(os.path.join(path, 'training_weights_max.npy'))
        hidden = int((len(params) - 1)/5)
        model_text = 'Max. Model'
    elif model == 'avg':
        params = np.load(os.path.join(path, 'training_weights_avg.npy'))
        hidden = int((len(params) - 1)/5)
        model_text = 'Avg. Model'
    elif model == 'other':
        file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
        if not file_path:
            messagebox.showerror("Error", "No file selected.")
            return
        #3*x + x + x = 15
#         [3,1,1]

# iw = 3*1 = 3
# ib = 1
# hw = 1
# ob = 1
# total = 6

# [3,x,1]
# iw = 3*x = 6/x
# ib = x
# hw = 2
# ob = 1
# total = 11

# [3,3,1]
# iw = 3*3 = 9
# ib= 3,x
# hw = 3,y
# ob = 1
# total = 16

# iw + ib+hw = 15
# 3*x + x + x = 15
#thats how we came up witht he solution its basically just param length - 1 where 1 is ob, and 3x is iw x is ib and x is hw so iw+hw+ib = param length - 1
# so 5x = param length - 1 
        params = np.load(file_path)
        hidden = int((len(params) - 1)/5)
        model_text = 'Custom Model'
    else:#min
        params = np.load(os.path.join(path, 'training_weights_min.npy'))
        hidden = int((len(params) - 1)/5)
        model_text = 'Min. Model'
    return model_text,hidden, params
# model_weights_loader('avg')
