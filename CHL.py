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
#from numba import njit
import numpy as np
import time

#%% [markdown]
#  Here are the functions that support the network

#%%

# Numerical methods
#@njit
def sigmoid(x, deriv = False):
    """Sigmoid logistic function (with derivative)"""
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

#@njit
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

#@njit
# def cross_entropy(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-12):
#     """
#     Computes cross entropy between targets (encoded as one-hot vectors) and predictions. 
#     Input: predictions (N, k) ndarray
#            targets (N, k) ndarray        
#     Returns: scalar
#     """
#     out: np.ndarray = None
#     predictions2: np.ndarray = np.clip(predictions, epsilon, 1. - epsilon, out)
#     N = predictions2.shape[0]
#     ce = -np.sum(targets*np.log(predictions2+1e-9)) / N
#     return ce

#@njit
def cross_entropy(predictions: np.ndarray, targets: np.ndarray):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(predictions < 0) or np.any(targets < 0):
        raise ValueError('Negative values exist.')

    # Force to proper probability mass function.
    #predictions = np.array(predictions, dtype=np.float)
    #targets = np.array(targets, dtype=np.float)
    predictions /= np.sum(predictions)
    targets /= np.sum(targets)

    # Ignore zero 'y' elements.
    mask = targets > 0
    x = predictions[mask]
    y = targets[mask]    
    ce = -np.sum(x * np.log(y)) 
    return ce    

#@njit
def mean_squared_error(p1, p2):
    """Calculates the mean squared error between two vectors'"""
    return 0.5 * np.sum(((p1 - p2) ** 2))

#from numba import jitclass          # import the decorator
#from numba import int32, float64    # import the types

# network_spec = [
#     ('n_inputs', int32),
#     ('n_hidden', int32),
#     ('n_outputs', int32),
#     ('eta', float64),
#     ('x', float64[:,:]),
#     ('h', float64[:,:]),
#     ('o', float64[:,:]),
#     ('w_xh', float64[:,:]),
#     ('w_ho', float64[:,:]),
#     ('patterns', float64[:,:]),
#     ('analogies', float64[:,:])
# ]

#@jitclass(network_spec)
class Network:
    # Definition of the network
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, training_data: np.ndarray, test_data: np.ndarray, desired_response_function: callable, collect_statistics_function: callable):
        np.random.seed(0)

        self.n_inputs  = n_inputs
        self.n_hidden  = n_hidden
        self.n_outputs = n_outputs

        self.x    = np.zeros((1, n_inputs))                                      # Input layer
        self.h    = np.zeros((1, n_hidden))                                      # Hidden layer
        self.o    = np.zeros((1, n_outputs))                                     # Output layer
        self.w_xh = np.random.random((n_inputs, n_hidden)) * 2 - 1.0             # First layer of synapses between input and hidden
        self.w_ho = np.random.random((n_hidden, n_outputs)) * 2 - 1.0            # Second layer of synapses between hidden and output

        self.patterns = training_data
        self.analogies = test_data

        self.target: callable = desired_response_function
        self.collect_statistics: callable = collect_statistics_function

    def set_inputs(self, pattern: np.ndarray):
        """Sets a given XOR pattern into the input value"""
        self.x = np.array(pattern).reshape((1,self.n_inputs))
        
    def set_outputs(self, vals: np.ndarray):
        """Sets the output variables"""
        self.o = np.array(vals).reshape((1,self.n_outputs))

    def set_hidden(self, vals: np.ndarray):
        """Sets the output variables"""
        self.h = vals

    def reset_hidden_to_rest(self):
        self.set_hidden(np.zeros((1, self.n_hidden)))

    def reset_outputs_to_rest(self):
        self.set_outputs(np.zeros((1, self.n_outputs)))

    def propagate(self, clamped_output: bool = False):
        """Spreads activation through a network"""        
        # First propagate forward from input to hidden layer
        self.h_input = self.x @ self.w_xh
        # Then propagate backward from output to hidden layer
        self.h_input += self.o @ self.w_ho.T
        self.h = sigmoid(self.h_input)
        
        if not clamped_output:
            # Propagate from the hidden layer to the output layer
            self.o_input = self.h @ self.w_ho
            self.o = sigmoid(self.o_input)

    def activation(self, clamped_output: bool = False, convergence: float = 0.00001, max_cycles: int = 1000, is_primed: bool = False):
        """Repeatedly spreads activation through a network until it settles"""
        if not is_primed:
            self.reset_hidden_to_rest()
        
        previous_h = np.copy(self.h)
        self.propagate(clamped_output)
        diff = mean_squared_error(previous_h, self.h)
        
        i = 0
        while diff > convergence and i < max_cycles:
            previous_h = np.copy(self.h)
            self.propagate(clamped_output)
            diff = mean_squared_error(previous_h, self.h)
            i += 1
        return i

    def calculate_response(self, p: np.ndarray, is_primed: bool = False):
        """Calculate the response for a given network's input"""
        self.set_inputs(p)
        self.reset_outputs_to_rest()
        self.activation(clamped_output = False, is_primed = is_primed)
        return np.copy(self.o)

    def unlearn(self, p: np.ndarray):
        """Negative, free phase. This is the 'expectation'."""
        self.set_inputs(p)
        # seems to converge quicker without this reset but I can't justify it.
        self.reset_outputs_to_rest()
        self.activation(clamped_output = False)

    def learn(self, p: np.ndarray):
        """Positive, clamped phase. This is the 'confirmation'."""
        self.set_inputs(p)
        self.set_outputs(self.target(p))
        self.activation(clamped_output = True)

    def update_weights_positive(self):
        """Updates weights. Positive Hebbian update (learn)"""
        self.w_xh += self.eta * (self.x.T @ self.h)
        self.w_ho += self.eta * (self.h.T @ self.o)
        
    def update_weights_negative(self):
        """Updates weights. Negative Hebbian update (unlearn)"""
        self.w_xh -= self.eta * (self.x.T @ self.h)
        self.w_ho -= self.eta * (self.h.T @ self.o)

    def update_weights_synchronous(self, h_plus, h_minus, o_plus, o_minus):
        """Updates weights. Synchronous Hebbian update."""
        self.w_xh += self.eta * (self.x.T @ (h_plus - h_minus))
        self.w_ho += self.eta * (self.h.T @ (o_plus - o_minus))

    def asynchronous_chl(self, min_error: float = 0.001, max_epochs: int = 1000, eta: float = 0.05) -> (np.ndarray, np.ndarray, np.ndarray, int): 
        """Learns associations by means applying CHL asynchronously"""
        self.min_error = min_error
        self.max_epochs = max_epochs
        self.eta = eta
        self.start_time = time.time()
        self.time_since_statistics = self.start_time
        E = [min_error * np.size(self.patterns, 0) + 1]  ## Error values. Initial error value > min_error
        P = [0] # Number of patterns correct
        A = [0] # Number of analogies correct
        epoch = 0
        while E[-1] > min_error * np.size(self.patterns, 0) and epoch < max_epochs:
            try:                
                for p in self.patterns:
                    # I cannot get it to converge with positive phase first.
                    # Maybe that's ok. Movellan (1990) suggests it won't converge
                    # without negative phase first. Also, Leech PhD (2008) 
                    # Simulation 5 does negative first, too.
                    # And so does Detorakis et al (2019).

                    # negative phase (expectation)
                    self.unlearn(p)
                    self.update_weights_negative()
                    # positive phase (confirmation)
                    self.learn(p)
                    self.update_weights_positive()

                # calculate and record statistics for this epoch
                self.collect_statistics(self, E, P, A, epoch)    
                
                epoch += 1
            except KeyboardInterrupt:
                break
        return E[1:], P[1:], A[1:], epoch

    def synchronous_chl(self, min_error: float = 0.001, max_epochs: int = 1000, eta: float = 0.05) -> (np.ndarray, np.ndarray, np.ndarray, int):
        """Learns associations by means applying CHL synchronously"""
        
        self.min_error = min_error
        self.max_epochs = max_epochs
        self.eta = eta
        
        E = [min_error * np.size(self.patterns, 0) + 1]  ## Error values. Initial error value > min_error
        P = [0] # Number of patterns correct
        A = [0] # Number of analogies correct
        epoch = 0
        while E[-1] > min_error * np.size(self.patterns, 0) and epoch < max_epochs:
            try:
                for p in self.patterns:    
                    #positive phase (confirmation)
                    self.learn(p)
                    h_plus = np.copy(self.h)
                    o_plus = np.copy(self.o)

                    #negative phase (expectation)
                    self.unlearn(p)
                    h_minus = np.copy(self.h)
                    o_minus = np.copy(self.o)

                    self.update_weights_synchronous(h_plus, h_minus, o_plus, o_minus)

                # calculate and record statistics for this epoch
                self.collect_statistics(E, P, A, epoch)    
        
                epoch += 1
            except KeyboardInterrupt:
                break

        return E[1:], P[1:], A[1:], epoch
