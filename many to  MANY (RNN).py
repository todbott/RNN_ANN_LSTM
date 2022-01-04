import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset

full_vector = np.array([[0,0,1],[0,1,0],[1,0,0]])
actual_x = np.array([1,0,0])
                     
# input variables
alpha = .01
input_dim = 3
hidden_dim = 7
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1


synapse_0_update = np.zeros_like(synapse_0)
synapse_h_update = np.zeros_like(synapse_h)
synapse_1_update = np.zeros_like(synapse_1)


# training logic
for j in range(10000):

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(0,len(full_vector)):
        
        # generate input and output
        X = full_vector[position]
        y = actual_x[position]

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
        

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        if(j % 15 == 0):
            print(layer_2," -- ",y)
    
        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2                                                     
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))    
        overallError += np.abs(layer_2_error[0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    if (j % 5000 == 0):
        print("")
    
    for position in range(0, len(full_vector)):
        
        X = full_vector[position]
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        """ get the middle delta by multiplying the next layer delta by the synapse weights and the derivative of the activation at the middle layer"""

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(np.atleast_2d(layer_2_delta))
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(np.atleast_2d(layer_1_delta))
        synapse_0_update += np.atleast_2d(X).T.dot(np.atleast_2d(layer_1_delta))
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0


        
