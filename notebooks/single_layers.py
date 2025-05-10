from typing import List, Optional, Tuple, cast
from typeguard import typechecked
import matplotlib.pyplot as plt
import math
import csv

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
#################################################

def activation(z: float) -> int:
    return 1 if z >= 0 else -1

def init_parameters() -> Tuple[List[float], float, float, int]:
    weights = [0.0, 0.0]
    bias = 0.0
    learning_rate = 0.2
    epochs = 50
    return weights, bias, learning_rate, epochs

def load_csv(path: str) -> Tuple[List[List[int]], List[int]]:
    X: List[List[int]] = []
    y: List[int] = []
    with open(path, 'r') as file:
        reader: csv._reader = csv.reader(file)
        data: List[List[str]] = [[c.replace('\ufeff', '') for c in row] for row in reader]
        for row in data:
            x1: int = int(row[0])
            x2: int = int(row[1])
            label: int = int(row[2])
            X.append([x1, x2])
            y.append(label)
    return X, y

def plot(name: str, X: List[List[int]], Y: List[int], weights: List[float], bias: float):
    # This for is for the decision boundarie
    for i in range(len(X)):
        x1, x2 = X[i]
        if y[i] == 1:
            plt.scatter(x1, x2, color='blue', marker='x', label='1' if i == 1 else "")
        else:
            plt.scatter(x1, x2, color='red', marker='o', label='-1' if i == 0 else "")

    x_vals: List[float] = [-2 + i * 0.1 for i in range(41)]

    if weights[1] != 0:
        y_vals: List[float] = [-(weights[0] * x + bias) / weights[1] for x in x_vals]
        plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')
    else:
        x0: float = -bias / weights[0]
        plt.axvline(x=x0, color='k', linestyle='--', label='Decision boundary')

    plt.title(f"Perceptron — Logical Gate {name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.savefig(f"{name}_perceptron.png")

def train(X: List[List[int]], y: List[int], epoch: int, learning_rate: float, weights: List[float], bias: float) -> Tuple[List[List[int]], List[int], List[float], float]:
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for i in range(len(X)):
            x1: int = X[i][0]
            x2: int = X[i][1]
            target: int = y[i]

            z: float = weights[0] * x1 + weights[1] * x2 + bias
            prediction: int = activation(z)
            error: int = target - prediction

            weights[0] += learning_rate * error * x1
            weights[1] += learning_rate * error * x2
            bias += learning_rate * error

            print(f"Input: {X[i]}, Target: {target}, Prediction: {prediction}, Error: {error}")
            print(f"  Weights: {weights}, Bias: {bias}")
        print("---")
    return X, y, weights, bias

#################################################
# Logic Gate OR classification, with Percepetron
X, y = load_csv("../data/portas-logicas/problemOR.csv")
weights, bias, learning_rate, epochs = init_parameters()
X, y, weights, bias = train(X, y, epochs, learning_rate, weights, bias)
plot('or', X, y, weights, bias)


#################################################
# Logic Gate AND classification, with Percepetron

X, y = load_csv("../data/portas-logicas/problemAND.csv")
weights, bias, learning_rate, epochs = init_parameters()
X, y, weights, bias = train(X, y, epochs, learning_rate, weights, bias)
plt.clf()
plot('and', X, y, weights, bias)