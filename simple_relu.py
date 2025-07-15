import numpy as np

x = np.load('x.npy')
y = np.load('y.npy')

hidden_weights = np.random.uniform(-0.1,0.1,size=(10,2))
hidden_bias = np.random.uniform(-0.1,0.1,size=2)
output_weights = np.random.uniform(-0.1,0.1,size=2)
output_bias = np.random.uniform(-0.1,0.1)