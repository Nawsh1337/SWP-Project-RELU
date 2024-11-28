import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

def relu(x):
    return np.maximum(0, x)

def relu_network(x, weights, biases):
    z = np.dot(x, weights) + biases
    return relu(z)

x1 = np.arange(-10, 11, 1)
x2 = np.arange(-10, 11, 1)
x1_grid, x2_grid = np.meshgrid(x1, x2)
inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]

initial_w1 = 0
initial_w2 = 0
initial_bias = 0

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

def update_plot(w1, w2, b):
    weights = np.array([[w1], [w2]])
    biases = np.array([b])
    output = relu_network(inputs, weights, biases).reshape(x1_grid.shape)
    ax.clear()
    ax.plot_surface(x1_grid, x2_grid, output, cmap="viridis", alpha=0.8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Output")
    ax.set_zlim(0, 20)
    plt.draw()

update_plot(initial_w1, initial_w2, initial_bias)

ax_w1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgrey")
ax_w2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor="lightgrey")
ax_bias = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgrey")

slider_w1 = Slider(ax_w1, "Weight 1", -5, 5, valinit=initial_w1, valstep=0.1)
slider_w2 = Slider(ax_w2, "Weight 2", -5, 5, valinit=initial_w2, valstep=0.1)
slider_bias = Slider(ax_bias, "Bias", -10, 10, valinit=initial_bias, valstep=0.1)

def sliders_on_change(val):
    w1 = slider_w1.val
    w2 = slider_w2.val
    bias = slider_bias.val
    update_plot(w1, w2, bias)

slider_w1.on_changed(sliders_on_change)
slider_w2.on_changed(sliders_on_change)
slider_bias.on_changed(sliders_on_change)

def on_click(event):
    if event.inaxes != ax:
        return
    
    w1 = slider_w1.val
    w2 = slider_w2.val
    bias = slider_bias.val
    weights = np.array([[w1], [w2]])
    biases = np.array([bias])
    output = relu_network(inputs, weights, biases).reshape(x1_grid.shape)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x1_norm = (event.x - ax.bbox.x0) / ax.bbox.width
    x2_norm = (event.y - ax.bbox.y0) / ax.bbox.height
    x1_val = xlim[0] + x1_norm * (xlim[1] - xlim[0])
    x2_val = ylim[0] + x2_norm * (ylim[1] - ylim[0])

    idx = np.unravel_index(
        np.argmin((x1_grid - x1_val) ** 2 + (x2_grid - x2_val) ** 2),
        x1_grid.shape
    )
    x1_val, x2_val = x1_grid[idx], x2_grid[idx]
    clicked_output = output[idx]

    equation = f"f(x1={x1_val}, x2={x2_val}, w1={w1}, w2={w2}, bias={bias}) = {clicked_output}"
    print(equation)

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
