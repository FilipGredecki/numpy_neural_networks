import numpy as np


inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

outputs = np.array([0,0,0,1]) 

weights = np.random.uniform(-1, 1, size=2)
bias = np.random.uniform(-1, 1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def neuron_output(x,weights,bias):
    z = np.dot(x,weights) + bias
    return sigmoid(z)

def compute_gradient(x, y, weights,bias):
    y_pred = neuron_output(x, weights, bias)
    error = 2 * (y_pred - y)
    d_sigmoid = y_pred * (1 - y_pred)
    gradient = error * d_sigmoid
    return gradient

epochs = 1000
lr = 0.1

for epoch in range(epochs):
    for x, y in zip(inputs,outputs):
        gradient = compute_gradient(x, y, weights,bias)
        weights -= lr * gradient * x
        bias -= lr * gradient


for x in inputs:
    y_pred = neuron_output(x, weights, bias)
    print(f"Wejście: {x}, wyjście neuronu: {y_pred:.3f}")