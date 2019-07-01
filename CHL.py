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
from dataclasses import dataclass
import numpy as np
import time
from numba import jit, njit
import np_clip_fix

#%% [markdown]
#  Here are the functions that support the network

#%%

# Numerical methods
@njit
def sigmoid(x):
    """Sigmoid logistic function"""
    k = 0.1 # smoothing parameter
    return 1 / (1 + np.exp(-x * k))

@njit
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

@njit
def cross_entropy(predictions: np.ndarray, targets: np.ndarray):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(predictions < 0) or np.any(targets < 0):
        raise ValueError('Negative values exist.')

    if not np.any(predictions):
        predictions = np.full(predictions.shape, 1 / len(predictions))
    
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


@njit
def mean_squared_error(p1, p2):
    """Calculates the mean squared error between two vectors'"""
    return 0.5 * np.sum(((p1 - p2) ** 2))

@njit
def add_noise(p: np.ndarray, noise: float):
    """Add normally distributed noise to a vector."""
    if noise > 0:
        noise_vector = np.random.normal(1, noise, p.size)
        p = p * noise_vector
        p = np.clip(p, 0., 1.)    
    return p    

@dataclass
class Config:
    # Training parameters
    min_error: float = 0.001
    min_error_for_correct: float = 1/16 
    max_epochs: int = 40000
    eta: float = 0.1
    noise: float = 0.
    adaptive_bias: bool = True
    strict_leech: bool = True

class Network:
    # Definition of the network
    def __init__(self, n_inputs: int, n_transformation: int, n_hidden: int, n_outputs: int, training_data: np.ndarray, test_data: np.ndarray, desired_response_function: callable, collect_statistics_function: callable):
        np.random.seed(0)

        self.n_inputs  = n_inputs
        self.n_transformation = n_transformation
        self.n_hidden  = n_hidden
        self.n_outputs = n_outputs

        # input layers
        self.x    = np.zeros((1, n_inputs))                                      # Input layer for object
        self.t    = np.zeros((1, n_transformation))                              # Input layer for transformation

        # hidden layer
        self.h    = np.zeros((1, n_hidden))                                      # Hidden layer

        # output layers
        self.o    = np.zeros((1, n_outputs))                                     # Output layer

        # weights
        self.w_xh = np.random.random((n_inputs, n_hidden)) * 2 - 1               # First layer of synapses between input and hidden
        self.w_th = np.random.random((n_transformation, n_hidden)) * 2 - 1       # First layer of synapses between transformation and hidden

        self.w_ho = np.random.random((n_hidden, n_outputs)) * 2 - 1              # Second layer of synapses between hidden and output
 
        # biases
        self.b_x = np.random.random((1, n_inputs)) * 2 - 1
        self.b_h = np.random.random((1, n_hidden)) * 2 - 1
        self.b_t = np.random.random((1, n_transformation)) * 2 - 1
        self.b_o = np.random.random((1, n_outputs)) * 2 - 1

        assert (training_data >= 0).all()
        assert (test_data <= 1).all()

        self.patterns = training_data
        self.transformations = np.asarray([p[-self.n_transformation:] for p in training_data])
        self.analogies = test_data

        self.target: callable = desired_response_function
        self.collect_statistics: callable = collect_statistics_function

        self.reset_transformation_to_rest()
        self.reset_hidden_to_rest()
        self.reset_outputs_to_rest()

    def set_inputs(self, pattern: np.ndarray):
        """Sets a given input pattern into the input value"""
        self.x = np.array(pattern[:self.n_inputs]).reshape((1,self.n_inputs))

    def set_transformation(self, pattern: np.ndarray):
        """Sets a given XOR pattern into the input value"""
        self.t = np.array(pattern[-self.n_transformation:]).reshape((1,self.n_transformation))

    def set_outputs(self, pattern: np.ndarray):
        """Sets the output variables"""
        self.o = np.array(pattern[:self.n_outputs]).reshape((1,self.n_outputs))

    def set_hidden(self, vals: np.ndarray):
        """Sets the output variables"""
        self.h = vals

    def reset_transformation_to_rest(self):
        self.set_transformation(np.full((1, self.n_transformation), 0.5))

    def reset_hidden_to_rest(self):
        self.set_hidden(np.full((1, self.n_hidden), 0.5))

    def reset_outputs_to_rest(self):
        #self.set_outputs(np.zeros((1, self.n_outputs)))
        shape = np.full((1, 6), 1 / 6)
        shape_param = np.full((1, 1), 0.5)
        features = np.full((1, 4), 0.5)
        self.set_outputs(np.concatenate((shape, shape_param, features), axis=1))

    def propagate(self, clamps = ['input', 'transformation']):
        """Spreads activation through a network"""
        # First propagate forward from input to hidden layer
        h_input = self.x @ self.w_xh

        # Then propagate forward from transformation to hidden layer
        h_input += self.t @ self.w_th

        # Then propagate backward from output to hidden layer
        h_input += self.o @ self.w_ho.T

        # And add biases
        h_input += self.b_h

       # (Makes more sense to me to update this after the updates to the other layers
        # so that they do not interfere with each other. But, this causes loops in the activation
        # and harms learning significantly. 
        self.h = sigmoid(h_input)

        # if input is free, propagate from hidden layer to input
        if not 'input' in clamps:
            # Propagate from the hidden layer to the output layer            
            x_input = self.h @ self.w_xh.T
            # Add bias
            x_input += self.b_x
            self.x = sigmoid(x_input)

        # if transformation is free, propagate from hidden layer to transformation input 
        if not 'transformation' in clamps:
            # Propagate from the hidden layer to the transformation layer
            t_input = self.h @ self.w_th.T
            # Add bias
            t_input += self.b_t
            self.t = sigmoid(t_input)

        # if output is free, propagate from hidden layer to output
        if not 'output' in clamps:
            # Propagate from the hidden layer to the output layer            
            o_input = self.h @ self.w_ho
            # Add bias
            o_input += self.b_o
            self.o = sigmoid(o_input)


    def activation(self, clamps = ['input', 'transformation'], convergence: float = 0.00001, max_cycles: int = 1000, is_primed: bool = False, is_unlearning: bool = False):
        """Repeatedly spreads activation through a network until it settles"""
        if not is_primed:
            self.reset_hidden_to_rest()
        
        i = 0
        diff = 0.
        j = 0
        while True:
            previous_h = np.copy(self.h)

            if not self.config.strict_leech:
                self.propagate(clamps)
            else:
                if not is_unlearning:
                    self.propagate(clamps)
                else:
                    max_cycles = 5
                    if i < 3: 
                        self.propagate(clamps)
                    else:
                        self.propagate([]) #unclamp everything after 2 cycles
                
            previous_diff = diff
            diff = mean_squared_error(previous_h, self.h)
            if diff == previous_diff:
                j += 1
                if j > 5:
                    # we are in a loop: I don't think this should ever happen
                    break
            else:
                j = 0
            if diff < convergence:
                # close enough to settled
                break
            if i > max_cycles:
                # not converging
                break
            i += 1
        return i

    def calculate_transformation(self, p: np.ndarray, o: np.ndarray):
        """Calculate the response for a given network's input"""
        self.set_inputs(p)
        self.set_outputs(o)
        self.reset_transformation_to_rest()
        self.activation(clamps = ['input', 'output'])
        return np.copy(self.t)

    def calculate_response(self, p: np.ndarray, is_primed: bool = False, is_unlearning=False):
        """Calculate the response for a given network's input"""
        self.set_inputs(p)
        if is_primed:
            if self.config.strict_leech:
                clamps = ['input']
                # Not sure about this. Why not leave the primed transformation input?
                self.reset_transformation_to_rest()
            else:          
                clamps = ['input', 'transformation']
                # Not sure about this. Why not leave the primed transformation input?
                self.reset_transformation_to_rest()
        else:
            clamps = ['input', 'transformation']   
            self.set_transformation(p)
        self.reset_outputs_to_rest()
        self.activation(clamps = clamps, is_primed = is_primed)
        return np.copy(self.o)[0]

    def calculate_response2(self, p: np.ndarray, is_primed: bool = False, is_unlearning=False):
        """Calculate the response for a given network's input"""
        self.set_inputs(p)
        clamps = ['input', 'transformation']
        if is_primed:
            if False and self.config.strict_leech:
                clamps = ['input']
                # Not sure about this. Why not leave the primed transformation input?
                self.reset_transformation_to_rest()
        else:
            self.set_transformation(p)
        self.reset_outputs_to_rest()
        self.activation(clamps = clamps, is_primed = is_primed)
        return np.copy(self.o)[0]


    def unlearn(self, p: np.ndarray):
        """Negative, free phase. This is the 'expectation'."""
        self.set_inputs(p)
        self.set_transformation(p)
        self.reset_outputs_to_rest()
        self.activation(clamps = ['input', 'transformation'], is_unlearning=True)

    def unlearn_t(self, p: np.ndarray):
        """Negative, free phase. This is the 'expectation'."""
        target = self.target(p)
        self.set_inputs(p)
        self.set_outputs(target)
        self.reset_transformation_to_rest()
        self.activation(clamps = ['input', 'output'])

    def learn(self, p: np.ndarray):
        """Positive, clamped phase. This is the 'confirmation'."""
        target = self.target(p)
        self.set_inputs(p)
        self.set_transformation(p)
        self.set_outputs(target)
        self.activation(clamps = ['input', 'transformation', 'output'])

    def update_weights_positive(self):
        """Updates weights. Positive Hebbian update (learn)"""
        eta = self.config.eta
        self.w_xh += eta * (self.x.T @ self.h)
        self.w_th += eta * (self.t.T @ self.h)
        self.w_ho += eta * (self.h.T @ self.o)

    def update_biases_positive(self):
        eta = self.config.eta
        self.b_x += eta * self.x
        self.b_t += eta * self.t
        self.b_h += eta * self.h
        self.b_o += eta * self.o

    def update_weights_negative(self):
        """Updates weights. Negative Hebbian update (unlearn)"""
        eta = self.config.eta
        self.w_xh -= eta * (self.x.T @ self.h)
        self.w_th -= eta * (self.t.T @ self.h)
        self.w_ho -= eta * (self.h.T @ self.o) 

    def update_biases_negative(self):
        eta = self.config.eta
        self.b_x -= eta * self.x
        self.b_t -= eta * self.t
        self.b_h -= eta * self.h
        self.b_o -= eta * self.o

    def update_weights_synchronous(self, t_plus, t_minus, h_plus, h_minus, o_plus, o_minus):
        """Updates weights. Synchronous Hebbian update."""
        eta = self.config.eta
        self.w_xh += eta * (self.x.T @ (h_plus - h_minus))
        self.w_th += eta * (self.t.T @ (h_plus - h_minus))
        self.w_ho += eta * (self.h.T @ (o_plus - o_minus))

    def update_biases_synchronous(self, t_plus, t_minus, h_plus, h_minus, o_plus, o_minus):
        eta = self.config.eta
        self.b_t = self.b_t + eta * (t_plus - t_minus)
        self.b_h = self.b_h + eta * (h_plus - h_minus)
        self.b_o = self.b_o + eta * (o_plus - o_minus)

    def asynchronous_chl(self, config: Config) -> (np.ndarray, np.ndarray, np.ndarray, int): 
        """Learns associations by means applying CHL asynchronously"""
        self.config = config

        self.start_time = time.time()
        self.time_since_statistics = self.start_time
        self.data = dict()

        E = [config.min_error * np.size(self.patterns, 0) + 1]  ## Error values. Initial error value > min_error
        P = [0] # Number of patterns correct
        A = [0] # Number of analogies correct
        epoch = 0
        while E[-1] > config.min_error * np.size(self.patterns, 0) and epoch < config.max_epochs:
            try:                
                # calculate and record statistics for this epoch
                self.collect_statistics(self, E, P, A, epoch, self.data)

                for p in self.patterns:
                    # I cannot get it to converge with positive phase first.
                    # Maybe that's ok. Movellan (1990) suggests it won't converge
                    # without negative phase first. Also, Leech PhD (2008) 
                    # Simulation 5 does negative first, too.
                    # And so does Detorakis et al (2019).

                    # add noise
                    p = add_noise(p, config.noise)                    
                    
                    # negative phase (expectation)
                    self.unlearn(p)
                    self.update_weights_negative()
                    if config.adaptive_bias:
                        self.update_biases_negative()
                    # positive phase (confirmation)
                    self.learn(p)
                    self.update_weights_positive()
                    if config.adaptive_bias:
                        self.update_biases_positive()

                    if not config.strict_leech:
                        # negative phase (expectation for transformation)
                        self.unlearn_t(p)
                        self.update_weights_negative()
                        if config.adaptive_bias:
                            self.update_biases_negative()
                        # positive phase (confirmation)
                        self.learn(p)
                        self.update_weights_positive()
                        if config.adaptive_bias:
                            self.update_biases_positive()

                epoch += 1
            except KeyboardInterrupt:
                break
        return E[1:], P[1:], A[1:], epoch, self.data


    def synchronous_chl(self, config: Config) -> (np.ndarray, np.ndarray, np.ndarray, int):
        """Learns associations by means applying CHL synchronously"""
        self.config = config

        self.start_time = time.time()
        self.time_since_statistics = self.start_time
        self.data = dict()

        E = [config.min_error * np.size(self.patterns, 0) + 1]  ## Error values. Initial error value > min_error
        P = [0] # Number of patterns correct
        A = [0] # Number of analogies correct
        epoch = 0
        while E[-1] > config.min_error * np.size(self.patterns, 0) and epoch < config.max_epochs:
            try:
                # calculate and record statistics for this epoch
                self.collect_statistics(self, E, P, A, epoch, self.data)    

                for p in self.patterns:
                    # add noise   
                    p = add_noise(p, config.noise)                    

                    #positive phase (confirmation)
                    self.learn(p)
                    t_plus = np.copy(self.t)
                    h_plus = np.copy(self.h)
                    o_plus = np.copy(self.o)

                    #negative phase (expectation)
                    self.unlearn(p)
                    t_minus = np.copy(self.t)
                    h_minus = np.copy(self.h)
                    o_minus = np.copy(self.o)

                    self.update_weights_synchronous(t_plus, t_minus, h_plus, h_minus, o_plus, o_minus)
                    if config.adaptive_bias:
                        self.update_biases_synchronous(t_plus, t_minus, h_plus, h_minus, o_plus, o_minus)
        
                epoch += 1
            except KeyboardInterrupt:
                break

        return E[1:], P[1:], A[1:], epoch, self.data
