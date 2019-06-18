
#%% [markdown]
# Functions which need passing to CHL

import os
os.environ['NUMBA_DISABLE_JIT'] = "0"

import numpy as np
import matplotlib.pyplot as plt
from CHL import Network, mean_squared_error, cross_entropy
from printing import generate_sandia_matrix, generate_rpm_sample, test_matrix
import time
from numba import jit, njit

@njit
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

#@njit
def calculate_error(p1, p2, min_error_for_correct):
    """Loss function loss(target, prediction)"""
    #loss = mean_squared_error(p1[0], p2[0])
    features_error = mean_squared_error(p1[0][6:11], p2[0][6:11])
    shape_error = cross_entropy(p2[0][0:6], p1[0][0:6])
    loss = 4 * features_error + 0.5 * shape_error
    is_correct = np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and np.allclose(p1[0][6:11], p2[0][6:11], atol=min_error_for_correct)
    #is_correct = np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and features_error < min_error_for_correct
    return loss, is_correct

def collect_statistics(network: Network, E: np.ndarray, P: np.ndarray, A: np.ndarray, epoch: int, data: dict):
    """Reporting function collect_statistics(
        E = loss by epoch, 
        P = num training patterns correct
        A = num test patterns [analogies] correct)"""

    statistics_frequency = 50 # report every n epochs

    if epoch % statistics_frequency == 0:
        if not 'by0' in data:
            data['by0'] = []
        if not 'by1' in data:
            data['by1'] = []
        if not 'by2' in data:
            data['by2'] = []
        if not 'by3' in data:
            data['by3'] = []

        e = 0. # total loss for this epoch
        min_error = network.min_error
        min_error_for_correct = network.min_error_for_correct
        max_epochs = network.max_epochs
        num_correct = 0
        num_correct_by_num_modifications = [0, 0, 0, 0]
        e_by_num_modifications = [0., 0., 0., 0.]
        num_analogies_correct = 0
        num_analogies_correct_by_num_modifications = [0, 0, 0, 0]
        e_analogies_by_num_modifications = [0., 0., 0., 0.]
        num_total_patterns_by_num_modifications = [0, 0, 0, 0]
        for p, a in zip(network.patterns, network.analogies):                
            p_error, is_correct = calculate_error(target(p), network.calculate_response(p), min_error_for_correct)
            
            num_modifications = int(sum(p[11:15]))
            num_total_patterns_by_num_modifications[num_modifications] += 1

            e += p_error

            # calculate_response(p) has primed the network for input p
            if is_correct:
                num_correct += 1
                num_correct_by_num_modifications[num_modifications] += 1
            e_by_num_modifications[num_modifications] += p_error
            
            num_modifications = int(sum(a[11:15]))
            a_error, is_correct = calculate_error(target(a), network.calculate_response(a, is_primed = True), min_error_for_correct)            
            if is_correct:
                num_analogies_correct += 1
                num_analogies_correct_by_num_modifications[num_modifications] += 1
            e_analogies_by_num_modifications[num_modifications] += a_error

        E.append(e)
        P.append(num_correct)
        A.append(num_analogies_correct)

        data['by0'].append(num_analogies_correct_by_num_modifications[0])
        data['by1'].append(num_analogies_correct_by_num_modifications[1])
        data['by2'].append(num_analogies_correct_by_num_modifications[2])
        data['by3'].append(num_analogies_correct_by_num_modifications[3])

        correct_by_num_modifications = [f'{x[0]}/{x[1]} {100*x[0]/x[1] if x[1] > 0 else 0:.1f}%' for x in zip(num_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        analogies_by_num_modifications = [f'{x[0]}/{x[1]} {100*x[0]/x[1] if x[1] > 0 else 0:.1f}%' for x in zip(num_analogies_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
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
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.time_since_statistics))
        total_time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.start_time))
        time_per_epoch = f'{1000 * (end - network.time_since_statistics) / statistics_frequency:.3f}'
        network.time_since_statistics = time.time()
        print(f'Elapsed time = {time_elapsed}s, Average time per epoch = {time_per_epoch}ms')
        print(f'Total elapsed time = {total_time_elapsed}s')

        update_plots(E[1:], P[1:], A[1:], data, dynamic=True)

def setup_plots():
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title(f'Relational priming for RPMs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    #ax2.legend(loc = 0)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Patterns correct')
    ax2.set_ylim(0, len(patterns))

    #fig.tight_layout()
    plt.ion()
    plt.show()

    return fig, ax1, ax2

def update_plots(E, P, A, data, dynamic=False):
    color = 'tab:red'
    ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
    ax1.plot(E, color=color)

    color = 'tab:blue'
    ax2.plot(P, color=color, label='Training')

    color = 'tab:green'
    ax2.plot(A, color=color, label='Test')

    color = 'tab:green'
    ax2.plot(data['by0'], linestyle='-', linewidth=0.5, color=color, label='Test0')
    ax2.plot(data['by1'], linestyle='-.', linewidth=0.5, color=color, label='Test1')
    ax2.plot(data['by2'], linestyle=(0, (1, 1)), linewidth=0.5, color=color, label='Test2')
    ax2.plot(data['by3'], linestyle=':', linewidth=0.5, color=color, label='Test3')

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if dynamic:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
    else:
        plt.ioff()
        plt.show()
#%% [markdown]
#  ### Test of CHL
# 
#  Here is a simple test of (asynchronous) CHL:

# The patterns to learn
n_sample_size = 400
min_error = 0.01
min_error_for_correct = 1/16 
max_epochs = 10000
eta = 0.05
noise = 0.01

tuples = [generate_rpm_sample() for x in range(1 * n_sample_size)]
#tuples = [generate_sandia_matrix() for x in range(1 * n_sample_size)]

#patterns are the training set
#analogies are the test set
patterns, analogies = [np.concatenate((item[1], item[2])) for item in tuples], [np.concatenate((item[3], item[2])) for item in tuples]
matrices = [item[0] for item in tuples]
patterns_array = np.asarray(patterns)
analogies_array = np.asarray(analogies)

network = Network(n_inputs=19, n_hidden=16, n_outputs=11, training_data=patterns_array, test_data=analogies_array, desired_response_function=target, collect_statistics_function=collect_statistics)

#%%
# Plot the Error by epoch

fig, ax1, ax2 = setup_plots()

start = time.time()
E, P, A, epoch, data = network.asynchronous_chl(min_error=min_error, max_epochs=max_epochs, eta=eta, noise=noise, min_error_for_correct=min_error_for_correct)
end = time.time()

print()
time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end-start))
print(f'Elapsed time {time_elapsed} seconds')

if E[-1] < min_error * np.size(patterns, 0):
    print(f'Convergeance reached after {epoch} epochs.')
else:
    print(f'Failed to converge after {epoch} epochs.')
        
print(f'Final error = {E[-1]}.')
print('')

# output first 10 patterns
for m, p in zip(matrices[:10], patterns[:10]):
    error, is_correct = calculate_error(target(p), network.calculate_response(p), min_error_for_correct)
    test_matrix(m, is_correct=is_correct)
    print(f'Pattern    = {np.round(p, 2)}')
    print(f'Target     = {np.round(target(p), 2)}')
    print(f'Prediction = {np.round(network.calculate_response(p), 2)}')
    print(f'Error      = {error:.3f}')
    print(f'Correct    = {is_correct}')
    print('')

update_plots(E, P, A, data, dynamic=False)

#%%
# Plot the Error by epoch


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
