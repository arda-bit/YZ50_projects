import math
import random

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def gaussian_dist(x: float, mu: float = 0.0, variance: float = 1.0) -> float:
    sigma = math.sqrt(variance)
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coefficient * exponent


def gaussian_init(param: list[float]) -> list[float]:
    return [gaussian_dist(elem, 0, 1) for elem in param]


def mse_loss(y_target: list[float], y_pred: list[float]) -> list[float]:
    return sum((y_t - y_p)**2 for y_t, y_p in zip(y_target, y_pred)) / len(y_target)


def generate_linear_toy_data(num_samples: int, d_in: int, d_out: int, noise_std: float = 0.1):

    true_weights = [[random.uniform(-1.0, 1.0) for _ in range(d_in)] for _ in range(d_out)]
    true_bias = [random.uniform(-0.5, 0.5) for _ in range(d_out)]

    inputs = []
    targets = []

    for _ in range(num_samples):
        x = [random.uniform(-2.0, 2.0) for _ in range(d_in)]
        
        y_obs = []
        for j in range(d_out):
            linear_output = sum(xi * wij for xi, wij in zip(x, true_weights[j])) + true_bias[j]
            y_target_j = sigmoid(linear_output)
            
            # Add Gaussian noise for unreducable error
            eps = random.gauss(0.0, noise_std)
            y_obs.append(y_target_j + eps)

        inputs.append(x)
        targets.append(y_obs)

    return inputs, targets, true_weights, true_bias




    
