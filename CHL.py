#%% [markdown]
# # Contrastive Hebbian Learning
# 
# Contrastive Hebbian Learning (CHL) is an algorithm that can be used to perform supervised learning in a neural network. It unites the power of backpropagation with the plausibility of Hebbian learning, and is thus a favorite of researchers in computational neuroscience. In terms of power, Xie (2006) has shown that, under most conditions, CHL is actually equivalent to backpropagation.
# 
# ## "Learn" and "Unlearn" phases
# 
# CHL works by performing Hebbian updates in two distinct phases, which are indicated as the __Learn__ (+) or the __Unlearn__ (-) phases. Synaptic weights are updated according to the difference of the two phases:
# 
#  $$
#  w_{i,j} \leftarrow w_{i,j} + \eta (y^+_i y^+_j - y^-_i y^-_j)
#  $$
# 
# Where $y$ represents, as usual, the output of a neuron.
# 
# ## Synchronous and Asynchronous
# 
# In the canonical equation (above), the two terms $y^+_i y^+_j$ and $y^-_i y^-_j$ are computed at different times but updated at the same moment. Because of this, the canonical form is called __synchronous__. This form is efficient but implausible, because it requires storing the products $y^+_i y^+_j$ and $-y^-_i y^-_j$ until the update is performed.
# 
# A more plausible alternative is to perform __asynchronous__  updates, with the product $y_i y_j$ because it is calculated and used immediately (just like in canonical Hebbian learning) with the sign of the update being dependent upon the phase.
# 
#  $$
#  w_{i,j} \leftarrow w_{i,j} +
#  \begin{cases}
#   + \eta (y_i y_j) & \mathrm{if~phase~is~Learn} \\
#   - \eta (y_i y_j) & \mathrm{if~phase~is~Unlearn}
#  \end{cases}
#  $$
# 
#  ## Connectivity
# 
# Because of its very nature, CHL requires the network to be __recurrent__, that is, synaptic matrices that connect two adjacent layers both forward and backward.
# 
# In turn, recurrent networks are intrinsically unstable, as they require multiple passes to converge towards a stable solution. The number of passes is sometimes used as a proxy for response times or similar behavioral measures.
# 
#  ## The Network
# 
# The CHL version of the XOR network is defined in these few lines of code.

#%%
import numpy as np

#%% [markdown]
#  Here are the functions that support the network

#%%

n_inputs: int
n_hidden : int
n_outputs: int
min_error: int
max_epochs: int

# Numerical methods
def sigmoid(x, deriv = False):
    """Sigmoid logistic function (with derivative)"""
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors) and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9)) / N
    return ce

def mean_squared_error(p1, p2):
    """Calculates the mean squared error between two vectors'"""
    return 0.5 * np.sum(((p1 - p2) ** 2))

# Definition of the network

def initialize(num_inputs, num_hidden, num_outputs, training_data, test_data, desired_response_function: callable, collect_statistics_function :callable):
    # network parameters
    global n_inputs, n_hidden, n_outputs, x, h, o, w_xh, w_ho, eta, min_error, max_epochs
    # data
    global patterns, analogies
    # functions
    global target, collect_statistics   

    np.random.seed(0)

    n_inputs  = num_inputs
    n_hidden  = num_hidden
    n_outputs = num_outputs

    x    = np.zeros((1, n_inputs))                                      # Input layer
    h    = np.zeros((1, n_hidden))                                      # Hidden layer
    o    = np.zeros((1, n_outputs))                                     # Output layer
    w_xh = np.random.random((n_inputs, n_hidden)) * 2 - 1.0             # First layer of synapses between input and hidden
    w_ho = np.random.random((n_hidden, n_outputs)) * 2 - 1.0            # Second layer of synapses between hidden and output

    min_error = 0.01
    max_epochs = 20000

    eta = 0.05  # Learning rate

    patterns = training_data
    analogies = test_data

    target = desired_response_function
    collect_statistics = collect_statistics_function

def set_inputs(pattern):
    """Sets a given XOR pattern into the input value"""
    global x
    x = np.array(pattern).reshape((1,n_inputs))
    
def set_outputs(vals):
    """Sets the output variables"""
    global o
    o = np.array(vals).reshape((1,n_outputs))

def set_hidden(vals):
    """Sets the output variables"""
    global h
    h = vals

def reset_hidden_to_rest():
    set_hidden(np.zeros((1, n_hidden)))

def reset_outputs_to_rest():
    set_outputs(np.zeros((1, n_outputs)))

def propagate(clamped_output = False):
    """Spreads activation through a network"""
    global h
    global o
    
    # First propagate forward from input to hidden layer
    h_input = x @ w_xh
    # Then propagate backward from output to hidden layer
    h_input += o @ w_ho.T
    h = sigmoid(h_input)
    
    if not clamped_output:
        # Propagate from the hidden layer to the output layer
        o_input = h @ w_ho
        o = sigmoid(o_input)

def activation(clamped_output = False, convergence = 0.00001, max_cycles = 1000, is_primed = False):
    """Repeatedly spreads activation through a network until it settles"""
    if not is_primed:
        reset_hidden_to_rest()
    
    previous_h = np.copy(h)
    propagate(clamped_output)
    diff = mean_squared_error(previous_h, h)
    
    i = 0
    while diff > convergence and i < max_cycles:
        previous_h = np.copy(h)
        propagate(clamped_output)
        diff = mean_squared_error(previous_h, h)
        i += 1
    return i

def calculate_response(p, is_primed = False):
    """Calculate the response for a given network's input"""
    set_inputs(p)
    reset_outputs_to_rest()
    activation(clamped_output = False, is_primed = is_primed)
    return np.copy(o)

def unlearn(p):
    """Negative, free phase. This is the 'expectation'."""
    set_inputs(p)
    # seems to converge quicker without this reset but I can't justify it.
    reset_outputs_to_rest()
    activation(clamped_output = False)

def learn(p):
    """Positive, clamped phase. This is the 'confirmation'."""
    set_inputs(p)
    set_outputs(target(p))
    activation(clamped_output = True)

def update_weights_positive():
    """Updates weights. Positive Hebbian update (learn)"""
    global w_xh, w_ho
    w_xh += eta * (x.T @ h)
    w_ho += eta * (h.T @ o)
    
def update_weights_negative():
    """Updates weights. Negative Hebbian update (unlearn)"""
    global w_xh, w_ho
    w_xh -= eta * (x.T @ h)
    w_ho -= eta * (h.T @ o)

def update_weights_synchronous(h_plus, h_minus, o_plus, o_minus):
    """Updates weights. Synchronous Hebbian update."""
    global w_xh, w_ho
    w_xh += eta * (x.T @ (h_plus - h_minus))
    w_ho += eta * (h.T @ (o_plus - o_minus))

def asynchronous_chl(min_error = 0.001, max_epochs = 1000):
    """Learns associations by means applying CHL asynchronously"""
    E = [min_error * np.size(patterns, 0) + 1]  ## Error values. Initial error value > min_error
    P = [0] # Number of patterns correct
    A = [0] # Number of analogies correct
    epoch = 0
    while E[-1] > min_error * np.size(patterns, 0) and epoch < max_epochs:
        try:
            e = 0.0
            
            for p in patterns:
                # I cannot get it to converge with positive phase first.
                # Maybe that's ok. Movellan (1990) suggests it won't converge
                # without negative phase first. Also, Leech PhD (2008) 
                # Simulation 5 does negative first, too.
                # And so does Detorakis et al (2019).)

                # negative phase (expectation)
                unlearn(p)
                update_weights_negative()
                # positive phase (confirmation)
                learn(p)
                update_weights_positive()

            # calculate and record statistics for this epoch
            e = collect_statistics(e, E, P, A, epoch)    
            
            epoch += 1
        except KeyboardInterrupt:
            break

    return E[1:], P[1:], A[1:], epoch

def synchronous_chl(min_error = 0.001, max_epochs = 1000):
    """Learns associations by means applying CHL synchronously"""
    E = [min_error * np.size(patterns, 0) + 1]  ## Error values. Initial error value > min_error
    P = [0] # Number of patterns correct
    A = [0] # Number of analogies correct
    epoch = 0
    while E[-1] > min_error * np.size(patterns, 0) and epoch < max_epochs:
        try:
            e = 0.0

            for p in patterns:    
                #positive phase (confirmation)
                learn(p)
                h_plus = np.copy(h)
                o_plus = np.copy(o)

                #negative phase (expectation)
                unlearn(p)
                h_minus = np.copy(h)
                o_minus = np.copy(o)

                update_weights_synchronous(h_plus, h_minus, o_plus, o_minus)

            # calculate and record statistics for this epoch
            e = collect_statistics(e, E, P, A, epoch)    
    
            epoch += 1
        except KeyboardInterrupt:
            break

    return E[1:], P[1:], A[1:], epoch
