from utils import sigmoid, gaussian_dist, gaussian_init
from typing import Callable, List, Optional

class SingleNeuron:
    def __init__(self, d_in: int,
        manual_weights: Optional[List[float]] = None,
        manual_bias: Optional[float] = None,
    ) -> None:
        self.d_in = d_in
        
        self.weight = manual_weights if manual_weights else gaussian_init([0.0] * d_in)
        self.bias = manual_bias if manual_bias else gaussian_dist(0.0)

        # Internal guard-rails
        assert len(self.weight) == self.d_in
        assert type(self.bias) == float

    def forward(self, x: List[float], activation: Callable = sigmoid) -> float:
        out = sum(xi * wi for xi, wi in zip(x, self.weight)) + self.bias
        return activation(out)


class MultiNeuronLayer:
    def __init__(self, d_in: int, d_out: int,
        manual_weights: Optional[List[List[float]]] = None,
        manual_bias: Optional[List[float]] = None,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out

        self.neuron_layer = [SingleNeuron(d_in, 
                manual_weights=manual_weights[i] if manual_weights else None,
                manual_bias=manual_bias[i] if manual_bias else None)
            for i in range(d_out)]

        # Internal guard-rails
        assert len(self.neuron_layer) == self.d_out

    def forward(self, x: List[float], activation: Callable = sigmoid) -> List[float]:
        out = []
        for i in range(self.d_out):
            out.append(self.neuron_layer[i].forward(x, activation=activation))
        return out



        
