import os
import numpy as np
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
model_weights_loader('avg')

