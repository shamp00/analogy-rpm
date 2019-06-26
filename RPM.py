
#%% [markdown]
# Functions which need passing to CHL

import os
os.environ['NUMBA_DISABLE_JIT'] = "0"

import numpy as np
import matplotlib.pyplot as plt
from CHL import Network, Config, mean_squared_error, cross_entropy
from printing import generate_sandia_matrix, generate_sandia_matrix_2_by_3, generate_rpm_sample, generate_all_sandia_matrices, test_matrix
import time
from numba import jit, njit
from colorama import init, Fore, Style
init()

@njit
def target(val):
    """Desired response function, target(pattern)"""
    old_school = True

    shape = np.copy(val[0:6])
    shape_param = np.copy(val[6:7])
    shape_features = np.copy(val[7:11])
    transformation_parameters = np.copy(val[15:])

    if old_school:
        shape_features = transformation_parameters
    else:
        rotation = 1
        shape_features += transformation_parameters
        if shape_features[rotation] > 1:
            shape_features[rotation] -= 1 # modulo 1 for rotation 
        assert (shape_features <= 1).all()
        assert (shape_features > 0).all()
    return np.concatenate((shape, shape_param, shape_features)).reshape((1, -1))

@njit
def calculate_error(p1, p2):
    """Loss function loss(target, prediction)"""
    features_error = mean_squared_error(p1[0][6:11], p2[0][6:11])
    shape_error = cross_entropy(p1[0][0:6], p2[0][0:6])
    #loss = 2 * features_error + 0.5 * shape_error
    loss = features_error + shape_error
    return loss

@njit
def calculate_transformation_error(p1, p2):
    """Loss function loss(target, prediction)"""
    return mean_squared_error(p1, p2)

def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    #dist_2 = np.sum((nodes - node)**2, axis=1)
    return nodes[np.argmin(dist_2)]

def color_on(color: str, condition: bool) -> str:
    if condition:
        return color
    else:
        return ''

def color_off() -> str:
    return Fore.RESET

def calculate_is_correct(p1, p2, targets, min_error_for_correct):
    #features_error = mean_squared_error(p1[0][6:11], p2[0][6:11])
    #return np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and features_error < min_error_for_correct

    #return np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and np.allclose(p1[0][6:11], p2[0][6:11], atol=min_error_for_correct)
    
    closest = closest_node(p1[0], targets)
    return np.allclose(closest, p2[0])

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

        if not 'aby0' in data:
            data['aby0'] = []
        if not 'aby1' in data:
            data['aby1'] = []
        if not 'aby2' in data:
            data['aby2'] = []
        if not 'aby3' in data:
            data['aby3'] = []
        if not '2by3' in data:
            data['2by3'] = []
        if not '2by3_loss' in data:
            data['2by3_loss'] = []
        if not 't_error' in data:
            data['t_error'] = []
        if not 'tf' in data:
            data['tf'] = []    
        if not 'o_error' in data:
            data['o_error'] = []

        e = 0. # total loss for this epoch
        sum_t_error = 0. # loss for transformation
        sum_o_error = 0. # loss for output
        min_error = network.config.min_error
        min_error_for_correct = network.config.min_error_for_correct
        max_epochs = network.config.max_epochs
        num_correct = 0
        num_correct_by_num_modifications = [0, 0, 0, 0]
        e_by_num_modifications = [0., 0., 0., 0.]
        num_analogies_correct = 0
        num_analogies_correct_by_num_modifications = [0, 0, 0, 0]
        e_analogies_by_num_modifications = [0., 0., 0., 0.]
        num_total_patterns_by_num_modifications = [0, 0, 0, 0]
        num_transformations_correct = 0
        targets = np.asarray([target(p)[0] for p in network.patterns])
        transformations = np.asarray([p[-network.n_transformation:] for p in network.patterns])
        for p, a in zip(network.patterns, network.analogies):
            t = target(p)
            t_error = 0 # the amount of error for the current transformation
            o_error = 0 # the amount of error for the current output object

            process_transformation_error = True
            process_analogy_error = True

            # Calculate loss on the training data. 
            # Present the network with input and transformation.
            # Clamp input and transformation.
            # Let the network settle.
            # Calculate loss (i.e., distance of prediction from target)
            
            # r = network.calculate_transformation(p, t)
            # o_error = calculate_transformation_error(p[-network.n_transformation:], network.t[0])
            # is_correct = False

            r = network.calculate_response(p)
            o_error = calculate_error(r, t)
            is_correct = calculate_is_correct(r, t, targets, min_error_for_correct)
            sum_o_error += o_error
            num_modifications = np.count_nonzero(p[-4:])
            num_total_patterns_by_num_modifications[num_modifications] += 1
            
            if is_correct:
                num_correct += 1
                num_correct_by_num_modifications[num_modifications] += 1
            e_by_num_modifications[num_modifications] += o_error + t_error

            if process_transformation_error:
                # Prime the network, that is, present object p and output target(p).
                # Do not present any transformation. Set the transformation to rest.
                # Clamp input and output. Do not clamp transformation.
                # Let the network settle.
                target_tf = p[-network.n_transformation:].reshape(1,-1)
                tf =network.calculate_transformation(p, t)
                is_correct_tf = calculate_is_correct(tf, target_tf, transformations, min_error_for_correct)
                if is_correct_tf:
                    num_transformations_correct += 1
                t_error = calculate_transformation_error(tf, target_tf)
                sum_t_error += t_error

            # total error for object + transformation
            e += o_error + t_error

            if process_analogy_error:
                # Now calculate the response of the primed network for new input a.
                # Clamp input only. Set output to rest.
                # (Leech paper says to set transformation to rest too.)
                # Let the network settle.
                #primed_t = network.calculate_transformation(p, t) # prime
                #primed_o = network.calculate_response(p) # prime
                t = target(a) 
                r = network.calculate_response(a, is_primed = True)    
                a_error = calculate_error(r, t)
                at_error = calculate_transformation_error(a[-network.n_transformation:], network.t[0])
                is_correct = calculate_is_correct(r, t, targets, min_error_for_correct)
                num_modifications = np.count_nonzero(p[-4:])   
                if is_correct:
                    num_analogies_correct += 1
                    num_analogies_correct_by_num_modifications[num_modifications] += 1
                e_analogies_by_num_modifications[num_modifications] += a_error + at_error

        E.append(e)
        P.append(num_correct)
        A.append(num_analogies_correct)

        data['tf'].append(num_transformations_correct)
        data['t_error'].append(sum_t_error)
        data['o_error'].append(sum_o_error)

        percentage_breakdown = [100*x[0]/x[1] if x[1] > 0 else 0 for x in zip(num_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        data['by0'].append(percentage_breakdown[0])
        data['by1'].append(percentage_breakdown[1])
        data['by2'].append(percentage_breakdown[2])
        data['by3'].append(percentage_breakdown[3])

        percentage_breakdown = [100*x[0]/x[1] if x[1] > 0 else 0 for x in zip(num_analogies_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        data['aby0'].append(percentage_breakdown[0])
        data['aby1'].append(percentage_breakdown[1])
        data['aby2'].append(percentage_breakdown[2])
        data['aby3'].append(percentage_breakdown[3])

        correct_by_num_modifications = [f'{x[0]}/{x[1]} {100*x[0]/x[1] if x[1] > 0 else 0:.1f}%' for x in zip(num_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        analogies_by_num_modifications = [f'{x[0]}/{x[1]} {100*x[0]/x[1] if x[1] > 0 else 0:.1f}%' for x in zip(num_analogies_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        loss_by_num_modifications = [f'{x:.3f}' for x in e_by_num_modifications]
        loss_analogies_by_num_modifications = [f'{x:.3f}' for x in e_analogies_by_num_modifications]

        print()
        print(f'Epoch      = {epoch} of {max_epochs}, Loss = {color_on(Fore.RED, e == min(E))}{e:.3f}{color_off()}, O/T = {color_on(Fore.GREEN, sum_o_error == max(data["o_error"]))}{sum_o_error:.3f}{color_off()}/{color_on(Fore.GREEN, sum_t_error == max(data["t_error"]))}{sum_t_error:.3f}{color_off()}, Terminating when < {min_error * len(patterns):.3f}')
        print(f'Patterns   = {color_on(Fore.GREEN, num_correct == max(P))}{num_correct:>5}{color_off()}/{len(patterns):>5}, breakdown = {" ".join(correct_by_num_modifications)}') 
        print(f'    Loss   = {np.sum(e_by_num_modifications):>11.3f}, breakdown = {" ".join(loss_by_num_modifications)}')        
        print(f'Analogies  = {color_on(Fore.GREEN, num_analogies_correct == max(A))}{num_analogies_correct:>5}{color_off()}/{len(analogies):>5}, breakdown = {" ".join(analogies_by_num_modifications)}')
        print(f'    Loss   = {np.sum(e_analogies_by_num_modifications):>11.3f}, breakdown = {" ".join(loss_analogies_by_num_modifications)}')
        print(f'Transforms = {color_on(Fore.GREEN, num_transformations_correct == max(data["tf"]))}{num_transformations_correct:>5}{color_off()}/{len(patterns):>5}')

        if include_2_by_3:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_23 = 0
            loss_23 = 0
            patterns_23, analogies_23, transformations2 = [np.concatenate((item[1], item[2])) for item in tuples_23], [np.concatenate((item[4], item[2])) for item in tuples_23], [item[3] for item in tuples_23]
            for p, a, transformation2 in zip(patterns_23, analogies_23, transformations2):
                # Prime the network, that is, present object p and output t.
                # Do not present any transformation. Set the transformation to rest.
                # Clamp input and output. Do not clamp transformation.
                # Let the network settle.
                t = target(p)
                network.calculate_transformation(p, t)

                # Now calculate the response of the primed network for new input a.
                # Clamp input only. Set output to rest.
                # Let the network settle.
                r = np.asarray(network.calculate_response(a, is_primed = True))[0]    

                p2 = np.concatenate((t[0], transformation2))
                a2 = np.concatenate((r, transformation2))

                network.calculate_transformation(p2, target(p2))

                #network.calculate_response(p2)
                r2 = network.calculate_response(a2, is_primed = True)

                t2 = target(np.concatenate((np.asarray(target(a))[0], transformation2)))

                loss_23 += calculate_error(r2, t2)
                is_correct_23 = calculate_is_correct(r2, t2, targets, min_error_for_correct)
                if is_correct_23:
                    num_correct_23 += 1

            data['2by3'].append(num_correct_23)
            data['2by3_loss'].append(loss_23)
            print(f'2x3        = {color_on(Fore.GREEN, num_correct_23 == max(data["2by3"]))}{num_correct_23:>5}{color_off()}/{100:>5}')
            print(f'    Loss   = {color_on(Fore.RED, loss_23 == min(data["2by3_loss"]))}{loss_23:>11.3f}{color_off()}')        

        end = time.time()
        if epoch == 0:
            end = network.start_time
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.time_since_statistics))
        total_time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.start_time))
        time_per_epoch = f'{1000 * (end - network.time_since_statistics) / statistics_frequency:.3f}'
        network.time_since_statistics = time.time()
        print(f'Elapsed time = {time_elapsed}s, Average time per epoch = {time_per_epoch}ms')
        print(f'Total elapsed time = {total_time_elapsed}s')

        update_plots(E[1:], P[1:], A[1:], data, dynamic=True, statistics_frequency=statistics_frequency)

def setup_plots():
    fig1 = plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    color = 'tab:red'
    ax1.set_title(f'Relational priming for RPMs')
    #ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Hide the x-axis labels for the first axis
    plt.setp(ax1.get_xticklabels(), visible=False)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    #ax2.legend(loc = 0)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, len(patterns))

    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    color = 'tab:green'
    #ax3.set_title(f'Breakdown by number of mods')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.tick_params(axis='x', labelcolor=color)
    
    #fig.tight_layout()
    plt.ion()
    plt.show()

    return fig1, ax1, ax2, ax3

def update_plots(E, P, A, data, dynamic=False, statistics_frequency=50):
    color = 'tab:red'
    ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
    ax1.plot(E, color=color)
    ax1.plot(data['o_error'], linestyle=':', linewidth=0.5, color=color)
    ax1.plot(data['t_error'], linestyle='-.', linewidth=0.5, color=color)

    color = 'tab:blue'
    ax2.plot(P, color=color, label='Training')

    color = 'tab:green'
    ax2.plot(A, color=color, label='Test')

    color = 'tab:gray'
    ax2.plot(data['tf'], color=color, label='Transformations')

    color = 'tab:blue'
    ax3.axis([0, len(E) + 10, 0, 100])
    if np.any(data['by0']):
        ax3.plot(data['by0'], linestyle='-', linewidth=1, color=color, label='0 mods')
    if np.any(data['by1']):
        ax3.plot(data['by1'], linestyle='-.', linewidth=1, color=color, label='1 mod')
    if np.any(data['by2']):
        ax3.plot(data['by2'], linestyle=(0, (1, 1)), linewidth=1, color=color, label='2 mods')
    if np.any(data['by3']):
        ax3.plot(data['by3'], linestyle=':', linewidth=1, color=color, label='3 mods')
    color = 'tab:green'
    ax3.axis([0, len(E) + 10, 0, 100])
    if np.any(data['aby0']):
        ax3.plot(data['aby0'], linestyle='-', linewidth=1, color=color, label='0 mods')
    if np.any(data['aby1']):
        ax3.plot(data['aby1'], linestyle='-.', linewidth=1, color=color, label='1 mod')
    if np.any(data['aby2']):
        ax3.plot(data['aby2'], linestyle=(0, (1, 1)), linewidth=1, color=color, label='2 mods')
    if np.any(data['aby3']):
        ax3.plot(data['aby3'], linestyle=':', linewidth=1, color=color, label='3 mods')

    color = 'tab:orange'
    if np.any(data['2by3']):
        ax3.plot(data['2by3'], linestyle='-', linewidth=1, color=color, label='2 by 3')

    ticks = ax3.get_xticks().astype('int') * statistics_frequency
    ax3.set_xticklabels(ticks)

    fig1.canvas.draw()
    fig1.canvas.flush_events()

    plt.pause(0.001)
    if not dynamic:
        plt.ioff()
        plt.show()
#%% [markdown]
#  ### Test of CHL
# 
#  Here is a simple test of (asynchronous) CHL:

# The patterns to learn
n_sample_size = 400
include_2_by_3 = True

#tuples = [generate_rpm_sample() for x in range(1 * n_sample_size)]
#tuples = [generate_sandia_matrix() for x in range(1 * n_sample_size)]
tuples = [x for x in generate_all_sandia_matrices(num_modifications = [0, 1, 2], include_shape_variants=False)]

if include_2_by_3:
    tuples_23 = [generate_sandia_matrix_2_by_3(include_shape_variants=False) for x in range(1 * 100)]

#patterns are the training set
#analogies are the test set
patterns, analogies = [np.concatenate((item[1], item[2])) for item in tuples], [np.concatenate((item[3], item[2])) for item in tuples]
matrices = [item[0] for item in tuples]
patterns_array = np.asarray(patterns)
analogies_array = np.asarray(analogies)

network = Network(n_inputs=11, n_transformation=4, n_hidden=17, n_outputs=11, training_data=patterns_array, test_data=analogies_array, desired_response_function=target, collect_statistics_function=collect_statistics)

#%%
# Plot the Error by epoch

fig1, ax1, ax2, ax3 = setup_plots()

config = Config()
config.min_error = 0.001
config.min_error_for_correct = 1/16 
config.max_epochs = 40000
config.eta = 0.05
config.noise = 0.
config.adaptive_bias = True
config.strict_leech = False

start = time.time()
E, P, A, epoch, data = network.asynchronous_chl(config)
end = time.time()

print()
time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end-start))
print(f'Elapsed time {time_elapsed} seconds')

if E[-1] < config.min_error * np.size(patterns, 0):
    print(f'Convergeance reached after {epoch} epochs.')
else:
    print(f'Failed to converge after {epoch} epochs.')
        
print(f'Final error = {E[-1]}.')
print('')

# output first 25 patterns
targets = np.asarray([target(p)[0] for p in network.patterns])
for m, a in zip(matrices[:25], analogies[:25]):
    t = target(a)
    r = network.calculate_response(a)
    error = calculate_error(r, t)
    is_correct = calculate_is_correct(r, t, targets, config.min_error_for_correct)
    test_matrix(m, is_correct=is_correct)
    print(f'Analogy    = {np.round(a, 2)}')
    print(f'Target     = {np.round(target(a), 2)}')
    print(f'Prediction = {np.round(network.calculate_response(a), 2)}')
    print(f'Error      = {error:.3f}')
    print(f'Correct    = {is_correct}')
    print('')

update_plots(E, P, A, data, dynamic=False)

#%%
