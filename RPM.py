
#%% [markdown]
# Functions which need passing to CHL

import numpy as np
import matplotlib.pyplot as plt
from CHL import Network, mean_squared_error, cross_entropy
from printing import generate_sandia_matrix, test_matrix
import time

def target(val):
    """Desired response function, target(pattern)"""
    shape = np.copy(val[0:6])
    shape_param = np.copy(val[6:7])
    shape_features = np.copy(val[7:11])
    modification_type = np.copy(val[11:15])
    modification_parameters = np.copy(val[15:])
    for i, modification in enumerate(modification_type):
        if modification > 0:
            shape_features[i] = modification_parameters[i]
    return np.concatenate((shape, shape_param, shape_features)).reshape((1, -1))
    #return np.concatenate((shape, shape_features, modification_type, modification_parameter)).reshape((1,n_outputs))

def calculate_error(p1, p2):
    """Loss function loss(target, prediction)"""
    #loss = mean_squared_error(p1[0], p2[0])
    loss = 2 * mean_squared_error(p1[0][6:11], p2[0][6:11]) + 0.5 * cross_entropy(p2[0][0:6], p1[0][0:6])
    is_correct = np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and np.array_equal(np.round(p1[0][6:11] * 8), np.round(p2[0][6:11] * 8))
    return loss, is_correct

def collect_statistics(network: Network, E: np.ndarray, P: np.ndarray, A: np.ndarray, epoch: int):
    """Reporting function collect_statistics(
        E = loss by epoch, 
        P = num training patterns correct
        A = num test patterns [analogies] correct)"""

    statistics_frequency = 50 # report every n epochs

    if epoch % statistics_frequency == 0:
        e = 0. # total loss for this epoch
        min_error = network.min_error
        max_epochs = network.max_epochs
        num_correct = 0
        num_correct_by_num_modifications = [0, 0, 0, 0]
        e_by_num_modifications = [0., 0., 0., 0.]
        num_analogies_correct = 0
        num_analogies_correct_by_num_modifications = [0, 0, 0, 0]
        e_analogies_by_num_modifications = [0., 0., 0., 0.]
        num_total_patterns_by_num_modifications = [0, 0, 0, 0]
        for p, a in zip(patterns, analogies):                
            p_error, is_correct = calculate_error(target(p), network.calculate_response(p))
            
            num_modifications = int(sum(p[11:15]))
            num_total_patterns_by_num_modifications[num_modifications] += 1

            e += p_error

            # calculate_response(p) has primed the network for input p
            if p_error < min_error:
            #if is_correct:
                num_correct += 1
                num_correct_by_num_modifications[num_modifications] += 1
            e_by_num_modifications[num_modifications] += p_error
            
            num_modifications = int(sum(a[11:15]))
            a_error, is_correct = calculate_error(target(a), network.calculate_response(a, is_primed = True))            
            if a_error < min_error:
            #if is_correct:
                num_analogies_correct += 1
                num_analogies_correct_by_num_modifications[num_modifications] += 1
            e_analogies_by_num_modifications[num_modifications] += a_error

        E.append(e)
        P.append(num_correct)
        A.append(num_analogies_correct)

        correct_by_num_modifications = [f'{x[0]}/{x[1]}' for x in zip(num_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        analogies_by_num_modifications = [f'{x[0]}/{x[1]}' for x in zip(num_analogies_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        loss_by_num_modifications = [f'{x:.3f}' for x in e_by_num_modifications]
        loss_analogies_by_num_modifications = [f'{x:.3f}' for x in e_analogies_by_num_modifications]
        
        print()
        print(f'Epoch     = {epoch} of {max_epochs}, Loss = {e:.3f}, Terminating when < {min_error * n_sample_size}')
        print(f'Patterns  = {num_correct:>5}/{n_sample_size:>5}, breakdown = {" ".join(correct_by_num_modifications)}')
        print(f'    Loss  = {np.sum(e_by_num_modifications):>11.3f}, breakdown = {" ".join(loss_by_num_modifications)}')        
        print(f'Analogies = {num_analogies_correct:>5}/{n_sample_size:>5}, breakdown = {" ".join(analogies_by_num_modifications)}')
        print(f'    Loss  = {np.sum(e_analogies_by_num_modifications):>11.3f}, breakdown = {" ".join(loss_analogies_by_num_modifications)}')        

        end = time.time()
        if epoch == 0:
            end = network.start_time
        total_time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.start_time))
        time_per_epoch = f'{1000 * (end - network.time_since_statistics) / statistics_frequency:.3f}'
        network.time_since_statistics = time.time()
        print(f'Average time per epoch {time_per_epoch} milliseconds')
        print(f'Total elapsed time {total_time_elapsed} seconds')

#%% [markdown]
#  ### Test of CHL
# 
#  Here is a simple test of (asynchronous) CHL:

#%%
#%%
def generate_rpm_sample(num_modifications = -1):
    """Generate a vector representing a 2x2 RPM matrix"""
    # scales = np.random.randint(0, 8)
    # rotation = np.random.randint(0, 8)
    # shading = np.random.randint(0, 8)
    # numerosity = np.random.randint(0, 8)

    # Create a vector like this [1 0 0 0] for shape1, say, ellipse
    shape_ints = np.random.choice(range(6), 2, replace=False)
    shape = np.zeros(6)
    shape[shape_ints[0]] = 1

    shape_param = np.zeros(1)
    shape_param[0] = np.random.choice([0.25, 0.5, 1, 2, 4, 8]) / 8

    analogy_shape = np.zeros(6)
    analogy_shape[shape_ints[1]] = 1

    shape_features = np.zeros(4) # for scale, rotation, shading, numerosity
    #shape_features = np.random.randint(4, size=4) / 4

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
    # make 0-3 modifications
    if num_modifications == -1:
        num_modifications = np.random.randint(4)
    modifications = np.random.choice(range(4), num_modifications, replace=False)
    for modification in modifications:
        modification_type[modification] = 1

    modification_parameters = np.zeros(4)
    for modification in modifications:
        parameter = np.random.randint(8)
        modification_parameters[modification] = parameter / 8

    sample = np.concatenate((shape, shape_features, shape_param))
    transformation = np.concatenate((modification_type, modification_parameters))
    analogy = np.concatenate((analogy_shape, shape_features, shape_param))
    return None, sample, transformation, analogy

# The patterns to learn
n_sample_size = 400
min_error = 0.01
max_epochs = 10000
eta = 0.05

tuples = [generate_rpm_sample() for x in range(1 * n_sample_size)]
#tuples = [generate_sandia_matrix() for x in range(1 * n_sample_size)]

#patterns are the training set
#analogies are the test set
patterns, analogies = [np.concatenate((item[1], item[2])) for item in tuples], [np.concatenate((item[3], item[2])) for item in tuples]
matrices = [item[0] for item in tuples]
patterns_array = np.asarray(patterns)
analogies_array = np.asarray(analogies)

network = Network(n_inputs = 19, n_hidden = 14, n_outputs = 11, training_data = patterns_array, test_data = analogies_array, desired_response_function=target, collect_statistics_function=collect_statistics)

start = time.time()
E, P, A, epoch = network.asynchronous_chl(min_error=min_error, max_epochs=max_epochs, eta=eta)
end = time.time()

print()
time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end-start))
print(f'Elapsed time {time_elapsed} seconds')

if E[-1] < min_error * np.size(patterns, 0):
    print(f'Convergeance reached after {epoch} epochs.')
else:
    print(f'Failed to converge after {epoch} epochs.')
        
print(f'Final error = {E[-1]}.')

# output first 10 patterns
for m, p in zip(matrices[:10], patterns[:10]):
    print('')
    test_matrix(m)
    print(f'Pattern    = {np.round(p, 2)}')
    print(f'Target     = {np.round(target(p), 2)}')
    print(f'Prediction = {np.round(network.calculate_response(p), 2)}')
    print(f'Error      = {calculate_error(target(p), network.calculate_response(p))}')

#%% [markdown]
#  And here is a plot of the error function and the network's learned outputs

#%%
# Plot the Error by epoch

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_title(f'CHL: Convergence reached after {epoch} epochs')
ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
ax1.plot(E, color=color)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Error')
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
#ax2.legend(loc = 0)

ax2.plot(P, color=color, label='Training')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Patterns correct')
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
