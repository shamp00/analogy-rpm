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
import matplotlib.pyplot as plt

np.random.seed(0)

#%%
def generate_rpm_sample():
    """Generate a vector representing a 2x2 RPM matrix"""
    # scales = np.random.randint(0, 8)
    # rotation = np.random.randint(0, 8)
    # shading = np.random.randint(0, 8)
    # numerosity = np.random.randint(0, 8)

    # Create a vector like this [1 0 0 0] for shape1, say, ellipse
    shape_ints = np.random.choice(range(6), 2, replace=False)
    shape = np.zeros(6)
    shape[shape_ints[0]] = 1

    analogy_shape = np.zeros(6)
    analogy_shape[shape_ints[1]] = 1
    shape_features = np.zeros(4) # for scale, rotation, shading, numerosity
    #shape_features = np.random.randint(8, size=4) / 8

    # To follow the relational priming example, we would need a 'causal agent'.
    #
    # Causal agent is,
    #   shape = transformer 
    #   scale = enlarger/shrinker, 
    #   shading = shader, 
    #   rotation = rotator, 
    #   numerosity = multiplier. 
    #
    # (Seems a little artificial but for now we'll go with it). Also, the
    # causal agent does not have the notion of degree, i.e., a slightly
    # cut apple versus a very cut apple, whereas a shape can be slightly 
    # shaded or slightly rotated.
    #
    # A shape transformation from say, triangle to circle is presumably a 
    # different causal agent than from triangle to square, so we'd end up with a 
    # separate causal agent for each transformation.
    # 
    # But we need to avoid this for the feature changes. We need to be careful that a change of shading from 1 to 2 is 
    # in some way the same causal agent as a change of shading from 3 to 4. Otherwise we end
    # up with each possible transformation having a separate casual agent.
    # In other words, what is the 'shape' equivalent of 
    #   apple, bread, lemon all being acted on by a knife.
    #   circle, triangle, square all being acted on by a modifier with a parameter?

    # scale, shading, rotation or numerosity
    modification_type = np.zeros(4)
    modification_type[np.random.randint(4)] = 1

    parameters = np.random.randint(8)
    modification_parameter = np.array([parameters / 8])

    sample = np.concatenate((shape, shape_features, modification_type, modification_parameter))
    analogy = np.concatenate((analogy_shape, shape_features, modification_type, modification_parameter))
    return (sample, analogy)

#%% [markdown]
#  Here are the functions that support the network

#%%
def logistic(x, deriv = False):
    """Sigmoid logistic function (with derivative)"""
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

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

def target(val):
    """Desired response function, t(p)"""
    shape = np.copy(val[0:6])
    shape_features = np.copy(val[6:10])
    modification_type = np.copy(val[10:14])
    modification_parameter = np.copy(val[14:])
    for i, modification in enumerate(modification_type):
        if modification > 0:
            shape_features[i] = modification * modification_parameter
    return np.concatenate((shape, shape_features)).reshape((1,n_outputs))
    #return np.concatenate((shape, shape_features, modification_type, modification_parameter)).reshape((1,n_outputs))

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def mean_squared_error(p1, p2):
    """Calculates the mean squared error"""
    return 0.5 * np.sum(((p1 - p2) ** 2))

def calculate_error(p1, p2):
    """Calculates the error function"""
    loss = 2 * mean_squared_error(p1[0][6:10], p2[0][6:10]) + 0.5 * cross_entropy(p2[0][0:6], p1[0][0:6])
    is_correct = np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and np.array_equal(np.round(p1[0][6:10] * 8), np.round(p2[0][6:10] * 8))
    return loss, is_correct

def propagate(clamped_output = False):
    """Spreads activation through a network"""
    global h
    global o
    
    # First propagate forward from input to hidden layer
    h_input = x @ w_xh
    # Then propagate backward from output to hidden layer
    h_input += o @ w_ho.T
    h = logistic(h_input)
    
    if not clamped_output:
        # Propagate from the hidden layer to the output layer
        o_input = h @ w_ho
        o = logistic(o_input)

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

def collect_statistics(e, E, P, A, epoch):
    if epoch % 10 == 0:
        num_correct = 0
        num_analogies_correct = 0
        for p, a in zip(patterns, analogies):                
            p_error, is_correct = calculate_error(target(p), calculate_response(p))
            
            # calculate_response(p) has primed the network for input p
            if p_error < min_error:
            #if is_correct:
                num_correct += 1
            e += p_error

            a_error, is_correct = calculate_error(target(a), calculate_response(a, is_primed = True))            
            if a_error < min_error:
            #if is_correct:
                num_analogies_correct += 1

        E.append(e)
        P.append(num_correct)
        A.append(num_analogies_correct)
        print(f'Epoch={epoch}, Error={e:.2f}, Correct={num_correct}, Analogies={num_analogies_correct}')
    return e

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


#%% [markdown]
#  ### Test of CHL
# 
#  Here is a simple test of (asynchronous) CHL:

#%%
n_inputs  = 15
n_hidden  = 14
n_outputs = 10

# The patterns to learn
n_sample_size = 1000
tuples = [generate_rpm_sample() for x in range(n_sample_size)]
#patterns are the training set
#analogies are the test set
patterns, analogies = [item[0] for item in tuples], [item[1] for item in tuples]

x    = np.zeros((1, n_inputs))                                      # Input layer
h    = np.zeros((1, n_hidden))                                      # Hidden layer
o    = np.zeros((1, n_outputs))                                     # Output layer
w_xh = np.random.random((n_inputs, n_hidden)) * 2 - 1.0             # First layer of synapses between input and hidden
w_ho = np.random.random((n_hidden, n_outputs)) * 2 - 1.0            # Second layer of synapses between hidden and output

min_error = 0.01
max_epochs = 10000

eta = 0.05  # Learning rate.

E, P, A, epoch = asynchronous_chl(min_error=min_error, max_epochs=max_epochs)

if E[-1] < min_error * np.size(patterns, 0):
    print(f'Convergeance reached after {epoch} epochs.')
else:
    print(f'Failed to converge after {epoch} epochs.')
        
print(f'Final error = {E[-1]}.')

# output first 20 patterns
for p in patterns[:20]:
    print('')
    print(f'Pattern    = {np.round(p, 2)}')
    print(f'Target     = {np.round(target(p), 2)}')
    print(f'Prediction = {np.round(calculate_response(p), 2)}')
    print(f'Error      = {calculate_error(target(p), calculate_response(p))}')

#%% [markdown]
#  And here is a plot of the error function and the network's learned outputs

#%%
# Plot the Error by epoch

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_title("CHL: Convergence reached after %d epochs" %(len(E)))
ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
ax1.plot(E, color=color)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
#ax2.legend(loc = 0)

ax2.plot(P, color=color, label='Training')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel("Patterns correct")
ax2.set_ylim(0, len(patterns))

color = 'tab:green'
ax2.plot(A, color=color, label='Test')

#fig.tight_layout()
plt.show()

# ## Plot the responses to the XOR patterns

# y_end = [calculate_response(p) for p in patterns[:10]]
# fig, ax = plt.subplots()
# ax.axis([-0.5, 3.5, 0, 1])
# ax.set_xticks(np.arange(5))
# ax.set_xticklabels(["(%s,%s)" % tuple(p) for p in patterns])
# ax.set_ylabel("Activation")
# ax.set_xlabel("Patterns")
# ax.bar(np.arange(4) - 0.25, y_end, 0.5, color='lightblue')
# ax.set_title("Responses to XOR patterns (CHL)")
# plt.show()

#%%
# res = np.zeros((len(patterns[:10]), h.size))

# for p in patterns[:10]:
#     calculate_response(p)
#     i = patterns.index(p) 
#     res[i] = h
    
# plt.imshow(res, interpolation="nearest")
# plt.title("Hidden layer responses by pattern")
# plt.yticks(np.arange(4), patterns[:10])
# plt.ylabel("Stimulus pattern")
# plt.xlabel("neuron")
# plt.show()

#%%
# res = np.zeros((len(patterns[:10]), h.size))

# for p in patterns[:10]:
#     calculate_response(p)
#     i = patterns.index(p) 
#     res[i] = h
    
# plt.imshow(res, interpolation="nearest")
# plt.title("Hidden layer responses by pattern")
# plt.yticks(np.arange(4), patterns[:10])
# plt.ylabel("Stimulus pattern")
# plt.xlabel("neuron")
# plt.show()

#%%
# import cairo
# import dataclasses
# import numpy.random as rd
# import os
# import math
# import copy
# from IPython.display import SVG, display
# import pyRavenMatrices.matrix as mat
# import pyRavenMatrices.element as elt
# import pyRavenMatrices.transformation as tfm
# import pyRavenMatrices.lib.sandia.definitions as defs
# import pyRavenMatrices.lib.sandia.generators as gen

# # pylint: disable-msg=E1101 
# # E1101: Module 'cairo' has no 'foo' member - of course it has! :) 

# def cell_path(cell):
    
#     return os.path.join('.', cell.id + '.svg')    

# def test_element(cell_structure, element):
#     surface = cairo.SVGSurface(cell_path(cell_structure), cell_structure.width, cell_structure.height)
#     ctx = cairo.Context(surface)

#     # set colour of ink to middle grey
#     #ctx.set_source_rgb(0.5, 0.5, 0.5)
    
#     element.draw_in_context(ctx, cell_structure)

#     ctx.stroke()
#     surface.finish()

#     display(SVG(cell_path(cell_structure)))

#%%
# #Generate only BasicElement structures
# i = 0
# #pattern = patterns[i]
# #analogies = analogies[i]

# basic_element = elt.BasicElement()
# basic_element.routine = defs.ellipse
# basic_element.params = { 'r': { 2: 1 / 3, 4: 1 / 3, 8: 1 / 3 } }

# modified_element = elt.ModifiedElement(basic_element, elt.ElementModifier())
# modified_element.modifiers[0].decorator = defs.shading

# parameter = 1/8
# modified_element.modifiers[0].decorator.parameters = parameter

# analogy_element = elt.BasicElement()
# analogy_element.routine = defs.tee
# analogy_element.params = {
#                 'r': {
#                     .25: .2,
#                     .5: .2,
#                     1: .2,
#                     2: .2,
#                     4: .2
#                 }

# modified_analogy_element = elt.ModifiedElement(basic_element, elt.ElementModifier())
# modified_analogy_element.modifiers[0].decorator = defs.shading
# modified_analogy_element.modifiers[0].decorator.parameters = parameter

# j=0
# cell_structure = mat.CellStructure("generated" + str(j), 64, 64, 2, 2)
# test_element(cell_structure, basic_element)

# j=1
# cell_structure = mat.CellStructure("generated" + str(j), 64, 64, 2, 2)
# test_element(cell_structure, modified_element)

# j=2
# cell_structure = mat.CellStructure("generated" + str(j), 64, 64, 2, 2)
# test_element(cell_structure, analogy_element)

# j=3
# cell_structure = mat.CellStructure("generated" + str(j), 64, 64, 2, 2)
# test_element(cell_structure, modified_analogy_element)






