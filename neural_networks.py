import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        
        #input to hidden weights
        self.iweights = np.random.uniform(-0.5, 0.5, (input_dim, hidden_dim))
        
        #hidden to output weights
        self.hweights = np.random.uniform(-0.5, 0.5, (hidden_dim, output_dim))
        
        #initialize bias for each layer
        self.ibias = np.random.uniform(-0.5, 0.5, hidden_dim)
        self.hbias = np.random.uniform(-0.5, 0.5, output_dim)

    #helper functions to handle different activation functions
    def activate(self, x):
        """Applies the selected activation function."""
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def activate_derivative(self, x):
        """Computes the derivative of the activation function."""
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")
        

    def forward(self, X, y=None):
        # TODO: forward pass, apply layers to input X
        self.z_hidden = np.dot(X, self.iweights) + self.ibias
        self.a_hidden = self.activate(self.z_hidden)
        self.z_output = np.dot(self.a_hidden, self.hweights) + self.hbias
        self.output = np.tanh(self.z_output)
        out = self.output
        if y is not None:
            self.loss = 0.5 * np.mean((self.output - y) ** 2)
        return out

    def backward(self, X, y):
        m = X.shape[0]
        # Compute derivative of loss w.r.t output activation
        d_output = (self.output - y) / m
        # Derivative of tanh activation function at output layer
        dz_output = d_output * (1 - self.output ** 2)
        # Gradients for hidden to output weights and biases
        dw_hweights = np.dot(self.a_hidden.T, dz_output)
        db_hbias = np.sum(dz_output, axis=0)
        # Backpropagate to hidden layer
        da_hidden = np.dot(dz_output, self.hweights.T)
        dz_hidden = da_hidden * self.activate_derivative(self.z_hidden)
        # Gradients for input to hidden weights and biases
        dw_iweights = np.dot(X.T, dz_hidden)
        db_ibias = np.sum(dz_hidden, axis=0)
        # Update weights and biases
        self.hweights -= self.lr * dw_hweights
        self.hbias -= self.lr * db_hbias
        self.iweights -= self.lr * dw_iweights
        self.ibias -= self.lr * db_ibias
        
def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

#helper function to display the nodes as seen in the example
def plot_gradient_graph(ax_gradient, mlp, frame):
    ax_gradient.clear()

    # Input layer nodes
    input_nodes = [(0, 0), (0, 1)]  # x1, x2 positions
    # Hidden layer nodes
    hidden_nodes = [(0.4, 0.2), (0.4, 0.6), (0.4, 1.0)]  # h1, h2, h3 positions
    # Output layer node
    output_node = [(1, 0.5)]  # y position

    # Combine all node positions
    nodes = input_nodes + hidden_nodes + output_node

    # Add node labels
    labels = ["x1", "x2", "h1", "h2", "h3", "y"]
    for (x, y), label in zip(nodes, labels):
        ax_gradient.text(x, y, label, fontsize=12, ha='center', va='center')

    # Plot nodes
    for x, y in nodes:
        ax_gradient.plot(x, y, 'o', markersize=30, color='blue', alpha=0.8)

    # Plot edges with gradient magnitudes
    i_to_h = mlp.iweights  # Input-to-hidden weights
    h_to_o = mlp.hweights  # Hidden-to-output weights

    # Scale gradients for visualization
    i_to_h_scaled = np.abs(i_to_h) / np.max(np.abs(i_to_h))
    h_to_o_scaled = np.abs(h_to_o) / np.max(np.abs(h_to_o))

    # Input to hidden edges
    for i, (x1, y1) in enumerate(input_nodes):
        for j, (x2, y2) in enumerate(hidden_nodes):
            weight_thickness = i_to_h_scaled[i, j] * 5  # Scale for thickness
            ax_gradient.plot([x1, x2], [y1, y2], linewidth=weight_thickness, color='purple', alpha=0.8)

    # Hidden to output edges
    for j, (x1, y1) in enumerate(hidden_nodes):
        x2, y2 = output_node[0]
        weight_thickness = h_to_o_scaled[j, 0] * 5  # Scale for thickness
        ax_gradient.plot([x1, x2], [y1, y2], linewidth=weight_thickness, color='purple', alpha=0.8)

    # Set limits and remove axes
    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)
    ax_gradient.axis('off')
    ax_gradient.set_title(f"Gradients at Frame {frame}")


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    print(f"Processing step: {frame}")
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    #check the log loss to see if the model is learning
    for _ in range(10):
        mlp.forward(X, y)
        mlp.backward(X, y)
        print(f"Loss after step {frame}: {mlp.loss}")

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.a_hidden 
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f"Hidden Space at Frame {frame}")

    # TODO: Hyperplane visualization in the hidden space
    x_hidden = np.linspace(-2, 2, 50)
    y_hidden = np.linspace(-2, 2, 50)
    X_hidden, Y_hidden = np.meshgrid(x_hidden, y_hidden)
    
    hweights_flat = mlp.hweights.flatten()
    hbias_flat = mlp.hbias.flatten()
    Z_hidden = -(hweights_flat[0] * X_hidden + hweights_flat[1] * Y_hidden + hbias_flat[0]) / hweights_flat[2]
    ax_hidden.plot_surface(X_hidden, Y_hidden, Z_hidden, alpha=0.3, color='gray')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # TODO: Distorted input space transformed by the hidden layer
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title(f"Input Space at Frame {frame}")
    
    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    mlp.forward(grid)
    preds = mlp.output.reshape(xx.shape)  
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], cmap='bwr', alpha=0.3)
    
    # TODO: Visualize features and gradients as circles and edges 
    plot_gradient_graph(ax_gradient, mlp, frame)
    
    assert hidden_features is not None and hidden_features.size > 0, "Hidden features are empty!"
    assert X is not None and X.size > 0, "Input data is empty!"
    # The edge thickness visually represents the magnitude of the gradient


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    frames=step_num//10
    print(f"Total frames: {frames}")
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)