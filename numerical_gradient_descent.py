import random
import matplotlib.pyplot as plt

from utils import mse_loss, generate_linear_toy_data
from models import MultiNeuronLayer


if __name__ == "__main__":
    random.seed(42)

    # Hyperparameters
    num_samples = 100
    d_in = 64
    d_out = 16
    noise_std = 0.05

    lr = 0.5         # Learning rate (eta)
    h = 1e-5         # Numerical epsilon
    num_epochs = 100  # Number of gradient update epochs

    # Toy-data
    inputs, targets, true_weights, true_bias = generate_linear_toy_data(num_samples, d_in, d_out, noise_std)

    model = MultiNeuronLayer(d_in, d_out)
    loss_history = []

    for ep in range(num_epochs):

        # Current loss
        y_pred = [model.forward(x) for x in inputs]
        loss = sum(mse_loss(t, p) for t, p in zip(targets, y_pred)) / len(targets)
        loss_history.append(loss)

        # Compute 2D Gradient Matrix for weights dL/dw (shape: d_out x d_in)
        w_grad = [[0.0] * d_in for _ in range(d_out)]

        for i in range(d_out):
            for j in range(d_in):
                model.neuron_layer[i].weight[j] += h    # perturb single weight w[i][j]

                y_pred_w_grad = [model.forward(x) for x in inputs]
                loss_w_grad = sum(mse_loss(t, p) for t, p in zip(targets, y_pred_w_grad)) / len(targets)

                # partial derivative w.r.t w[i][j]
                w_grad[i][j] = (loss_w_grad - loss) / h

                model.neuron_layer[i].weight[j] -= h    # backtrack

        # Compute 1D Gradient Vector for bias dL/db (shape: d_out)
        b_grad = [0.0] * d_out

        for i in range(d_out):
            model.neuron_layer[i].bias += h      # perturb single bias b[i]

            y_pred_b_grad = [model.forward(x) for x in inputs]
            loss_b_grad = sum(mse_loss(t, p) for t, p in zip(targets, y_pred_b_grad)) / len(targets)

            # partial derivative w.r.t b[i]
            b_grad[i] = (loss_b_grad - loss) / h

            model.neuron_layer[i].bias -= h     # backtrack

        # Optimizer step
        for i in range(d_out):
            for j in range(d_in):
                model.neuron_layer[i].weight[j] -= lr * w_grad[i][j]
            model.neuron_layer[i].bias -= lr * b_grad[i]
        
        if ep % 10 == 0 or ep == num_epochs - 1:
            print(f"Epoch {ep:03d}/{num_epochs} | MSE Loss: {loss:.6f}")

    # Plot training loss progression
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='dodgerblue', linewidth=2)
    plt.title("Numerical Gradient Descent Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("num_gd_loss.png")
    plt.show()