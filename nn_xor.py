import numpy as np

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

outputs = np.array([0,1,1,0]) 

hidden_weights = np.random.uniform(-1,1,size=(2,2))
hidden_bias = np.random.uniform(-1,1,size=2)
output_weights = np.random.uniform(-1,1,size=2)
output_bias = np.random.uniform(-1,1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def neuron_output(x,weights,bias):
    z = np.dot(weights,x) + bias
    return sigmoid(z)

def forward(x, hidden_weights, hidden_bias, output_weights, output_bias):
    z_hidden = np.dot(x, hidden_weights) + hidden_bias
    sigmoid_hidden = sigmoid(z_hidden)
    z_output = np.dot(sigmoid_hidden, output_weights) + output_bias
    sigmoid_output = sigmoid(z_output)
    return sigmoid_output


def backword_pass(x,y,hidden_weights,hidden_bias,output_weights,output_bias):
    hidden_neurons = np.dot(x,hidden_weights) + hidden_bias  # z_hidden [x, y]
    sigmoid_hidden = sigmoid(hidden_neurons)
    output_neurons = np.dot(sigmoid_hidden, output_weights) + output_bias
    sigmoid_output = sigmoid(output_neurons)

    error_output = sigmoid_output - y

    d_output_sigmoid = sigmoid_output * (1 - sigmoid_output)
    delta_output = error_output * d_output_sigmoid
    gradient_output_weights = delta_output * sigmoid_hidden  
    gradient_output_bias = delta_output

    d_hidden_sigmoid = sigmoid_hidden * (1 - sigmoid_hidden)
    delta_hidden = delta_output * output_weights * d_hidden_sigmoid
    gradient_hidden_weights = np.outer(x,delta_hidden)
    gradient_hidden_bias = delta_hidden

    return gradient_output_weights,gradient_output_bias,gradient_hidden_weights,gradient_hidden_bias

epochs = 10000
lr = 0.2

for epoch in range(epochs):
    for x,y in zip(inputs,outputs):
        gradient_output_weights,gradient_output_bias,gradient_hidden_weights,gradient_hidden_bias = backword_pass(x,y,hidden_weights,hidden_bias,output_weights,output_bias)
        hidden_weights -= lr * gradient_hidden_weights
        hidden_bias -= lr * gradient_hidden_bias
        output_weights -= lr * gradient_output_weights
        output_bias -= lr * gradient_output_bias
    
for x in inputs:
    y_pred = forward(x, hidden_weights, hidden_bias, output_weights, output_bias)
    label = 1 if y_pred >= 0.5 else 0 
    print(f"input: {x}, predicted output: {label}")
