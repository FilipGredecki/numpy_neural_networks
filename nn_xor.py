import numpy as np

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

outputs = np.array([0,0,0,1]) 

hidden_weights = np.random.uniform(-1,1,size=(2,2))
hidden_bias = np.random.uniform(-1,1,size=2)
output_weights = np.random.uniform(-1,1,size=2)
output_bias = np.random.uniform(-1,1)