#first file
import numpy as np
import plotly.graph_objects as go

def relu(x):
    return np.maximum(0, x)

def relu_network(x, weights, biases):
    z = np.dot(x, weights) + biases
    return relu(z)

x1 = np.arange(-10, 11, 1)
x2 = np.arange(-10, 11, 1)
x1_grid, x2_grid = np.meshgrid(x1, x2)
inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]
print(inputs)

initial_w1 = 0
initial_w2 = 0
initial_bias = 0
weights = np.array([[initial_w1], [initial_w2]])
biases = np.array([initial_bias])
output_grid = relu_network(inputs, weights, biases).reshape(x1_grid.shape)

fig = go.Figure(data=[go.Surface(z=output_grid, x=x1, y=x2)])

sliders = []

slider_weight_1 = {
    "active": 5,
    "currentvalue": {"prefix": "Weight 1: "},
    "pad": {"t": 50},
    "steps": []
}
for w1 in np.round(np.arange(-5, 5.1, 0.1), 1):
    step = {
        "label": f"{w1:.1f}",
        "method": "update",
        "args": [
            {"z": [relu_network(inputs, np.array([[w1], [initial_w2]]), biases).reshape(x1_grid.shape)]}
        ]
    }
    slider_weight_1["steps"].append(step)
sliders.append(slider_weight_1)

slider_weight_2 = {
    "active": 5,
    "currentvalue": {"prefix": "Weight 2: "},
    "pad": {"t": 130},
    "steps": []
}
for w2 in np.round(np.arange(-5, 5.1, 0.1), 1):
    step = {
        "label": f"{w2:.1f}",
        "method": "update",
        "args": [
            {"z": [relu_network(inputs, np.array([[initial_w1], [w2]]), biases).reshape(x1_grid.shape)]}
        ]
    }
    slider_weight_2["steps"].append(step)
sliders.append(slider_weight_2)

slider_bias = {
    "active": 5,
    "currentvalue": {"prefix": "Bias: "},
    "pad": {"t": 200},
    "steps": []
}
for b in np.round(np.arange(-10, 10.1, 0.1), 1):
    step = {
        "label": f"{b:.1f}",
        "method": "update",
        "args": [
            {"z": [relu_network(inputs, weights, np.array([b])).reshape(x1_grid.shape)]}
        ]
    }
    slider_bias["steps"].append(step)
sliders.append(slider_bias)

fig.update_layout(
    print('changed'),
    sliders=sliders,
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="Output",
        xaxis=dict(nticks=21),
        yaxis=dict(nticks=21)
    )
)
fig.update_layout(
    scene=dict(
        zaxis=dict(range=[0, 20]),#need to fix this
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="Output"
    )
)


fig.show()
