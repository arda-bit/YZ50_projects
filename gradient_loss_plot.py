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

    # Toy-data
    inputs, targets, true_weights, true_bias = generate_linear_toy_data(num_samples, d_in, d_out, noise_std)

    errors = [-1.0]
    while errors[-1] <= 1.0:
        errors.append(errors[-1] + 0.05)
    
    d_in, d_out = len(true_weights[0]), len(true_weights)

    print(f'd_in: {d_in}, d_out: {d_out}')

    weight_loss_list = []
    bias_loss_list = []
    
    for error in errors:
        
        obs_weight = [[true_weights[i][j] + error for j in range(d_in)] for i in range(d_out)]
        obs_bias = [true_bias[i] + error for i in range(d_out)]

        model = MultiNeuronLayer(d_in, d_out, manual_weights=obs_weight, manual_bias=true_bias)
        preds = [model.forward(x) for x in inputs]
        weight_loss_list.append(sum(mse_loss(t, p) for t, p in zip(targets, preds)) / len(targets))

        model = MultiNeuronLayer(d_in, d_out, manual_weights=true_weights, manual_bias=obs_bias)
        preds = [model.forward(x) for x in inputs]
        bias_loss_list.append(sum(mse_loss(t, p) for t, p in zip(targets, preds)) / len(targets))

    plt.plot(errors, weight_loss_list, label="Weight Error Loss")
    plt.plot(errors, bias_loss_list, label="Bias Error Loss")
    plt.xlabel("Error Offset")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()

