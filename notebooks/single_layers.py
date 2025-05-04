from typing import List, Optional, Tuple, cast
from typeguard import typechecked
import math

@typechecked
class Neuron:
    def __init__(self, weights: List[float], inputs: List[float], bias: float = 0.0) -> None:
        self._inputs = inputs
        self._weights = weights
        self._bias = bias

    @property
    def inputs(self) -> List[float]:
        return self._inputs

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def bias(self) -> float:
        return self._bias


# Single Layer, Single Neuron
n = Neuron([1.0, 2.0, 3.0], [0.2, 0.8, -0.5], 2)
output = n.inputs[0] * n.weights[0] + n.inputs[1] * n.weights[1] + n.inputs[2] * n.weights[2] + n.bias
print(output)

# Single Layer, Multiple Neurons
n1 = Neuron([0.2, 0.8, -0.5, 1], [1.0, 2.0, 3.0, 2.5], 2)
n2 = Neuron([0.5, -0.91, 0.26, -0.5], [1.0, 2.0, 3.0, 2.5], 3)
n3 = Neuron([-0.26, -0.27, 0.17, 0.87], [1.0, 2.0, 3.0, 2.5], 0.5)

values: List[float] = []
result: Optional[Tuple[float, float, float]] = None # The result should be exactly 3 values, one for each neuron
weights_combined: List[List[float]] = [n.weights for n in [n1, n2, n3]]
biases_combined: List[float] = [n.bias for n in [n1, n2, n3]]
inputs: List[float] = n1.inputs # or n2 or n3, all the same inputs

for neuronX_weights, neuronX_bias in zip(weights_combined, biases_combined):
    neuronX_output: float = 0.0
    for inputX, weightX in zip(inputs, neuronX_weights):
        neuronX_output += inputX * weightX
    neuronX_output += neuronX_bias
    values.append(neuronX_output)

if len(values) == 3:
    result = cast(Tuple[float, float, float], tuple(values))
else:
    result = None
print(result)
