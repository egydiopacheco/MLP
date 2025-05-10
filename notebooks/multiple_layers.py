from typing import List, Optional, Tuple, cast
from typeguard import typechecked
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import csv

# Since XOR can't be solved with a single layer perceptron due to non-linearity,
# it will be solved here using a multi layer one, using backpropagation

def tanh(x: float) -> float:
    """Hyperbolic tangent activation function."""
    # I have seen that for small problems like these, tanh is better than using ReLU or other common activation functions
    return math.tanh(x)

def dtanh(x: float) -> float:
    """Derivative of the tanh activation function."""
    return 1 - math.tanh(x) ** 2

def dot(a: List[float], b: List[float]) -> float:
    """Calculates the dot product between two vectors."""
    return sum(i * j for i, j in zip(a, b))

def init_weights(rows: int, cols: int) -> List[List[float]]:
    """Initializes a 2D list of weights with random values between -1 and 1."""
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

def load_csv(path: str) -> Tuple[List[List[float]], List[float]]:
    """Loads inputs and targets from a CSV file."""
    X, y = [], []
    with open(path, 'r') as f:
        for row in csv.reader(f):
            x1, x2, label = [float(val.replace('\ufeff', '')) for val in row]
            X.append([x1, x2])
            y.append(label)
    return X, y

def add_bias(x: List[float]) -> List[float]:
    """Appends a bias term (1.0) to the input vector."""
    return x + [1.0]

def forward(x: List[float], w1: List[List[float]], w2: List[float]) -> Tuple[List[float], float]:
    """Performs a forward pass through the MLP and returns hidden activations and output."""
    h_in = [dot(w, x) for w in w1]
    h_out = [tanh(h) for h in h_in]
    y_pred = tanh(dot(w2, h_out + [1.0]))
    return h_out, y_pred

def backprop(x: List[float], target: float, h: List[float], y_pred: float,
             w1: List[List[float]], w2: List[float], lr: float) -> None:
    """Updates weights using backpropagation and gradient descent."""
    d_out = (target - y_pred) * dtanh(dot(w2, h + [1.0]))
    for i in range(len(w2) - 1):
        w2[i] += lr * d_out * h[i]
    w2[-1] += lr * d_out * 1.0
    for i in range(len(w1)):
        d_hidden = d_out * w2[i] * dtanh(dot(w1[i], x))
        for j in range(len(w1[i])):
            w1[i][j] += lr * d_hidden * x[j]

def train(X: List[List[float]], y: List[float], epochs: int, lr: float) -> Tuple:
    """Trains the MLP using the given dataset, learning rate, and epochs."""
    w1 = init_weights(2, 3)
    w2 = [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(epochs):
        for xi, target in zip(X, y):
            x_biased = add_bias(xi)
            h, y_pred = forward(x_biased, w1, w2)
            backprop(x_biased, target, h, y_pred, w1, w2, lr)
    return w1, w2

def predict(x: List[float], w1: List[List[float]], w2: List[float]) -> float:
    """Predicts the output of the MLP given an input and trained weights."""
    _, y_pred = forward(add_bias(x), w1, w2)
    return 1 if y_pred >= 0 else -1

def compute_accuracy_and_confusion_matrix(
    X: List[List[float]], y: List[int], w1: List[List[float]], w2: List[float]
) -> Tuple[float, List[List[int]]]:
    """Computes accuracy and confusion matrix."""
    tp = tn = fp = fn = 0
    for xi, yi in zip(X, y):
        pred = predict(xi, w1, w2)
        if pred == 1 and yi == 1:
            tp += 1
        elif pred == -1 and yi == -1:
            tn += 1
        elif pred == 1 and yi == -1:
            fp += 1
        elif pred == -1 and yi == 1:
            fn += 1
    acc = (tp + tn) / len(X)
    cm = [[tp, fn], [fp, tn]]
    return acc, cm

def plot_decision_boundary(X: List[List[float]], y: List[float], w1: List[List[float]], w2: List[float]) -> None:
    """Plots the decision boundary and data points for visualization."""
    for i in range(len(X)):
        x1, x2 = X[i]
        color = 'blue' if y[i] == 1 else 'red'
        marker = 'x' if y[i] == 1 else 'o'
        plt.scatter(x1, x2, color=color, marker=marker)

    grid = [(-1 + i * 0.1, -1 + j * 0.1) for i in range(21) for j in range(21)]
    for gx, gy in grid:
        pred = predict([gx, gy], w1, w2)
        c = 'cyan' if pred > 0 else 'orange'
        plt.plot(gx, gy, '.', color=c, alpha=0.2)

    plt.title("MLP — XOR")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.savefig("xor_mlp.png")

def plot_confusion_matrix(cm: List[List[int]]) -> None:
    """
    Plots the confusion matrix using seaborn heatmap.
    cm: Confusion matrix as a 2x2 list [[TP, FN], [FP, TN]]
    """
    labels = ["Predicted 1", "Predicted -1"]
    ax_labels = ["Actual 1", "Actual -1"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=ax_labels)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix_xor.png")

X, y = load_csv("../data/portas-logicas/problemXOR.csv")
w1, w2 = train(X, y, epochs=100, lr=0.5)
for xi, target in zip(X, y):
    pred = predict(xi, w1, w2)
    print(f"Input: {xi}, Target: {target}, Predicted: {round(pred, 2)}")
plot_decision_boundary(X, y, w1, w2)

accuracy, confusion_matrix = compute_accuracy_and_confusion_matrix(X, y, w1, w2)
print(f"Accuracy: {accuracy}")
plot_confusion_matrix(confusion_matrix)