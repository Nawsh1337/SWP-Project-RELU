import os
import numpy as np
import torch
import torch.nn as nn

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


path = 'SWP Project RELU\Final code\Functions\Models'

# print(os.listdir(path))

#user will select the model and it will be loaded.

params  = None

def model_weights_loader(model = 'max'):
    if model == 'max':
        hidden = 3
        params = np.load(os.path.join(path, 'max_3_weights.npy'))
    elif model == 'avg':
        hidden = 3
        params = np.load(os.path.join(path, 'avg_3_weights.npy'))
    else:#min
        hidden = 5
        params = np.load(os.path.join(path, 'min_5_weights.npy'))
    return hidden, params
# model_weights_loader('avg')

def predict(model_name,values):
    model = None
    if model_name == 'max':
        hidden_size = 3
        model = ReLU_MaxNN2(input_size, hidden_size) 
        model.load_state_dict(torch.load("SWP Project RELU/Final code/Functions/Models/max_3.pth"))
        
    elif model_name == 'avg':
        hidden_size = 3
        model = ReLU_MaxNN2(input_size, hidden_size) 
        model.load_state_dict(torch.load("SWP Project RELU/Final code/Functions/Models/avg_3.pth"))

    else:#min
        hidden_size = 5
        model = ReLU_MaxNN2(input_size, hidden_size) 
        model.load_state_dict(torch.load("SWP Project RELU/Final code/Functions/Models/min_5.pth"))

    model.eval()
    input_tensor = torch.tensor([values], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor).item()
    return output