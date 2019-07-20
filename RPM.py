
#%% [markdown]
# Functions which need passing to CHL

import os
os.environ['NUMBA_DISABLE_JIT'] = "0"

import numpy as np
import matplotlib.pyplot as plt

from CHL import Network, mean_squared_error, cross_entropy
from printing import Lexicon, generate_rpm_2_by_2_matrix, generate_rpm_2_by_3_matrix, generate_rpm_3_by_3_matrix, test_matrix, target
from config import Config
import time
from numba import jit, njit
from colorama import init, Fore, Style
import pickle
import glob

# Colorama init() fixes Windows console, but prevents colours in IPython
#init()

class Plots:
    fig1: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes

@njit
def calculate_error(p1, p2):
    """Loss function loss(target, prediction)"""
    return mean_squared_error(p1, p2[:len(p1)])
    # features_error = mean_squared_error(p1[6:11], p2[6:11])
    # shape_error = cross_entropy(p1[0:6], p2[0:6])
    # #loss = 2 * features_error + 0.5 * shape_error
    # loss = features_error + shape_error
    # return loss

@njit
def calculate_transformation_error(t1, t2):
    """Loss function loss(target, prediction)"""
    return mean_squared_error(t1, t2)


def closest_node_index(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    #dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def closest_node(node, nodes):
    index = closest_node_index(node, nodes)
    return nodes[index]


def color_on(color: str, condition: bool) -> str:
    if condition:
        return color
    else:
        return ''

def color_off() -> str:
    return Fore.RESET

def calculate_is_correct(p1, p2, targets):
    #features_error = mean_squared_error(p1[0][6:11], p2[0][6:11])
    #return np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and features_error < min_error_for_correct

    #return np.argmax(p1[0][0:6]) == np.argmax(p2[0][0:6]) and np.allclose(p1[0][6:11], p2[0][6:11], atol=min_error_for_correct)
    
    closest = closest_node(p1, targets)
    return np.allclose(closest, p2[:len(p1)])

def collect_statistics(network: Network, E: np.ndarray, P: np.ndarray, A: np.ndarray, epoch: int, data: dict):
    """Reporting function collect_statistics(
        E = loss by epoch, 
        P = num training patterns correct
        A = num test patterns [analogies] correct)"""

    checkpoint_frequency = 50

    if epoch % checkpoint_frequency == 0:
        checkpoint = {
            'epoch' : epoch,
            'network' : network,
            'E' : E,
            'P' : P,
            'A' : A,
            'data' : data
        }

        with open(f'{get_checkpoints_folder(network.config)}/checkpoint.{epoch:05}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(checkpoint, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    statistics_frequency = 50 # report every n epochs

    if epoch % statistics_frequency == 0:

        if not 'a' in data:
            data['a'] = []
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
        
        if not 'eby0' in data:
            data['eby0'] = []
        if not 'eby1' in data:
            data['eby1'] = []
        if not 'eby2' in data:
            data['eby2'] = []
        if not 'eby3' in data:
            data['eby3'] = []

        if not 'eaby0' in data:
            data['eaby0'] = []
        if not 'eaby1' in data:
            data['eaby1'] = []
        if not 'eaby2' in data:
            data['eaby2'] = []
        if not 'eaby3' in data:
            data['eaby3'] = []
        if not '2by2' in data:
            data['2by2'] = []
        if not '2by2_loss' in data:
            data['2by2_loss'] = []
        if not '2by3' in data:
            data['2by3'] = []
        if not '2by3_loss' in data:
            data['2by3_loss'] = []
        if not '3by3' in data:
            data['3by3'] = []
        if not '3by3_loss' in data:
            data['3by3_loss'] = []
        if not 't_error' in data:
            data['t_error'] = []
        if not 'tf' in data:
            data['tf'] = []    
        if not 'tby0' in data:
            data['tby0'] = []    
        if not 'tby1' in data:
            data['tby1'] = []    
        if not 'tby2' in data:
            data['tby2'] = []    
        if not 'tby3' in data:
            data['tby3'] = []    
        if not 'o_error' in data:
            data['o_error'] = []
        if not 'a_error' in data:
            data['a_error'] = []


        e = 0. # total loss for this epoch
        sum_t_error = 0. # loss for transformation
        sum_o_error = 0. # loss for output
        sum_a_error = 0. # loss for analogies
        min_error = network.config.min_error
        max_epochs = network.config.max_epochs
        num_correct = 0
        num_correct_by_num_modifications = [0, 0, 0, 0]
        is_max_num_correct_by_num_modifications = [False, False, False, False]
        is_max_num_analogies_correct_by_num_modifications = [False, False, False, False]
        is_min_e_by_num_modifications = [False, False, False, False]
        is_min_e_analogies_by_num_modifications = [False, False, False, False]
        e_by_num_modifications = [0., 0., 0., 0.]
        num_analogies_correct = 0
        num_analogies_correct_by_num_modifications = [0, 0, 0, 0]
        e_analogies_by_num_modifications = [0., 0., 0., 0.]
        num_total_patterns_by_num_modifications = [0, 0, 0, 0]
        num_transformations_correct = 0
        num_total_transformations_by_type = [0, 0, 0, 0]
        num_correct_by_transformation = [0, 0, 0, 0]
        is_max_num_correct_by_transformation = [False, False, False, False]
        targets = np.asarray([target(p)[:network.n_inputs] for p in network.patterns])
        #a_targets = np.asarray([target(a) for a in network.analogies])

        for p, a, c in zip(network.patterns, network.analogies, network.candidates):
            t = target(p)
            t_error = 0 # the amount of error for the current transformation
            o_error = 0 # the amount of error for the current output object

            process_transformation_error = True
            process_analogy_error = True
            process_2_by_2 = True
            process_2_by_3 = True
            process_3_by_3 = True

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
            is_correct = calculate_is_correct(r, t, targets)
            sum_o_error += o_error
            num_modifications = (p[-4:] != 0.5).sum()
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
                target_tf = p[-network.n_transformation:]
                num_total_transformations_by_type = np.add(num_total_transformations_by_type, [x != 0.5  for x in target_tf])
                tf = network.calculate_transformation(p, t)[0]
                is_correct_tf = calculate_is_correct(tf, target_tf, network.transformations)
                if is_correct_tf:
                    num_transformations_correct += 1
                    num_correct_by_transformation = np.add(num_correct_by_transformation, [x != 0.5  for x in target_tf])
                t_error = calculate_transformation_error(tf, target_tf)
                sum_t_error += t_error

            # total error for object + transformation
            e += o_error + t_error

            if process_analogy_error:
                # Now calculate the response of the primed network for new input a.
                # Clamp input only. Set output to rest.
                # (Leech paper says to set transformation to rest too.)
                # Let the network settle.
                r, t = complete_analogy_22(network, p, a)
                a_error = calculate_error(r, t)
                at_error = calculate_transformation_error(a[-network.n_transformation:], network.t[0])
                is_correct = calculate_is_correct(r, t, c)
                num_modifications = (p[-4:] != 0.5).sum()  
                if is_correct:
                    num_analogies_correct += 1
                    num_analogies_correct_by_num_modifications[num_modifications] += 1
                e_analogies_by_num_modifications[num_modifications] += a_error + at_error
                sum_a_error += a_error

        E.append(e)
        P.append(num_correct)
        A.append(num_analogies_correct)
        data['a'].append(100 * num_analogies_correct / len(network.analogies))

        data['tf'].append(num_transformations_correct)
        data['t_error'].append(sum_t_error)
        data['o_error'].append(sum_o_error)
        data['a_error'].append(sum_a_error)

        percentage_breakdown = [100*x[0]/x[1] if x[1] > 0 else 0 for x in zip(num_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        for i, x in enumerate(percentage_breakdown):
            label = f'by{i}'
            data[label].append(percentage_breakdown[i])
            is_max_num_correct_by_num_modifications[i] = percentage_breakdown[i] > 0.0 and percentage_breakdown[i] == max(data[label])

        percentage_breakdown = [100*x[0]/x[1] if x[1] > 0 else 0 for x in zip(num_analogies_correct_by_num_modifications, num_total_patterns_by_num_modifications)]
        for i, x in enumerate(percentage_breakdown):
            label = f'aby{i}'
            data[label].append(percentage_breakdown[i])
            is_max_num_analogies_correct_by_num_modifications[i] = percentage_breakdown[i] > 0.0 and percentage_breakdown[i] == max(data[label])

        for i, x in enumerate(e_by_num_modifications):
            label = f'eby{i}'
            data[label].append(e_by_num_modifications[i])
            is_min_e_by_num_modifications[i] = any(data[label]) and e_by_num_modifications[i] == min(data[label])

        for i, x in enumerate(e_analogies_by_num_modifications):
            label = f'eaby{i}'
            data[label].append(e_analogies_by_num_modifications[i])
            is_min_e_analogies_by_num_modifications[i] = any(data[label]) and e_analogies_by_num_modifications[i] == min(data[label])

        for i, x in enumerate(num_correct_by_transformation):
            label = f'tby{i}'
            data[label].append(num_correct_by_transformation[i])
            is_max_num_correct_by_transformation[i] = num_correct_by_transformation[i] > 0 and num_correct_by_transformation[i] == max(data[label])

        correct_by_num_modifications = [f'{color_on(Fore.GREEN, x[2])}{x[0]}{color_off()}/{x[1]} {color_on(Fore.GREEN, x[2])}{100*x[0]/x[1] if x[1] > 0 else 0:.1f}%{color_off()}' for x in zip(num_correct_by_num_modifications, num_total_patterns_by_num_modifications, is_max_num_correct_by_num_modifications)]
        analogies_by_num_modifications = [f'{color_on(Fore.GREEN, x[2])}{x[0]}{color_off()}/{x[1]} {color_on(Fore.GREEN, x[2])}{100*x[0]/x[1] if x[1] > 0 else 0:.1f}%{color_off()}' for x in zip(num_analogies_correct_by_num_modifications, num_total_patterns_by_num_modifications, is_max_num_analogies_correct_by_num_modifications)]
        loss_by_num_modifications = [f'{color_on(Fore.RED, x[1])}{x[0]:.3f}{color_off()}' for x in zip(e_by_num_modifications, is_min_e_by_num_modifications)]
        loss_analogies_by_num_modifications = [f'{color_on(Fore.RED, x[1])}{x[0]:.3f}{color_off()}' for x in zip(e_analogies_by_num_modifications, is_min_e_analogies_by_num_modifications)]
        correct_transformations_by_type = [f'{color_on(Fore.GREEN, x[2])}{x[0]}{color_off()}/{x[1]} {color_on(Fore.GREEN, x[2])}{100*x[0]/x[1] if x[1] > 0 else 0:.1f}%{color_off()}' for x in zip(num_correct_by_transformation, num_total_transformations_by_type, is_max_num_correct_by_transformation)]

        tuples_22 = network.tuples_22
        tuples_23 = network.tuples_23
        tuples_33 = network.tuples_33

        print()
        print(f'Epoch      = {epoch} of {max_epochs}, Loss = {color_on(Fore.RED, e == min(E[1:]))}{e:.3f}{color_off()}, O/T = {color_on(Fore.RED, sum_o_error == min(data["o_error"]))}{sum_o_error:.3f}{color_off()}/{color_on(Fore.RED, sum_t_error == min(data["t_error"]))}{sum_t_error:.3f}{color_off()}, Terminating when < {min_error * len(network.patterns):.3f}')
        print(f'Patterns   = {color_on(Fore.GREEN, num_correct == max(P))}{num_correct:>5}{color_off()}/{len(network.patterns):>5}, breakdown = {" ".join(correct_by_num_modifications)}') 
        print(f'    Loss   = {color_on(Fore.RED, any(data["o_error"]) and sum_o_error == min(data["o_error"]))}{sum_o_error:>11.3f}{color_off()}, breakdown = {" ".join(loss_by_num_modifications)}')        
        print(f'Transforms = {color_on(Fore.GREEN, num_transformations_correct == max(data["tf"]))}{num_transformations_correct:>5}{color_off()}/{len(network.patterns):>5}, breakdown = {" ".join(correct_transformations_by_type)} (sz, rt, sh, no)')
        print(f'    Loss   = {color_on(Fore.RED, any(data["t_error"]) and sum_t_error == min(data["t_error"]))}{sum_t_error:>11.3f}{color_off()}')        
        print(f'Analogies  = {color_on(Fore.GREEN, num_analogies_correct == max(A))}{num_analogies_correct:>5}{color_off()}/{len(network.analogies):>5}, breakdown = {" ".join(analogies_by_num_modifications)}')
        print(f'    Loss   = {color_on(Fore.RED, any(data["a_error"]) and sum_a_error == min(data["a_error"]))}{np.sum(e_analogies_by_num_modifications):>11.3f}{color_off()}, breakdown = {" ".join(loss_analogies_by_num_modifications)}')

        if process_2_by_2:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_22 = 0
            loss_22 = 0            
            patterns_22, analogies_22, candidates_22 = [np.concatenate((item[2], item[3])) for item in network.tuples_22], [np.concatenate((item[4], item[3])) for item in tuples_22], [item[1] for item in tuples_22]
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_23, transformations2)])
            for p, a, candidates_for_pattern in zip(patterns_22, analogies_22, candidates_22):
                prediction, actual = complete_analogy_22(network, p, a)

                loss_22 += calculate_error(prediction, actual)
                is_correct_22 = calculate_is_correct(prediction, actual, candidates_for_pattern)
                if is_correct_22:
                    num_correct_22 += 1

            data['2by2'].append(num_correct_22)
            data['2by2_loss'].append(loss_22)
            print(f'2x2        = {color_on(Fore.GREEN, num_correct_22 == max(data["2by2"]))}{num_correct_22:>5}{color_off()}/{100:>5}')
            print(f'    Loss   = {color_on(Fore.RED, loss_22 == min(data["2by2_loss"]))}{loss_22:>11.3f}{color_off()}')        

        if process_2_by_3:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_23 = 0
            loss_23 = 0
            patterns_23, analogies_23, transformations2, candidates = [np.concatenate((item[2], item[3])) for item in tuples_23], [np.concatenate((item[5], item[3])) for item in tuples_23], [item[4] for item in tuples_23], [item[1] for item in tuples_23]
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_23, transformations2)])
            for p, a, transformation2, candidates_for_pattern in zip(patterns_23, analogies_23, transformations2, candidates):
                prediction, actual = complete_analogy_23(network, p, a, transformation2)

                loss_23 += calculate_error(prediction, actual)
                is_correct_23 = calculate_is_correct(prediction, actual, candidates_for_pattern)
                if is_correct_23:
                    num_correct_23 += 1

            data['2by3'].append(num_correct_23)
            data['2by3_loss'].append(loss_23)
            print(f'2x3        = {color_on(Fore.GREEN, num_correct_23 == max(data["2by3"]))}{num_correct_23:>5}{color_off()}/{100:>5}')
            print(f'    Loss   = {color_on(Fore.RED, loss_23 == min(data["2by3_loss"]))}{loss_23:>11.3f}{color_off()}')        

        if process_3_by_3:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_33 = 0
            loss_33 = 0
            patterns_33, analogies_row2_33, analogies_row3_33, transformations2, candidates = [np.concatenate((item[2], item[3])) for item in tuples_33], [np.concatenate((item[5], item[3])) for item in tuples_33], [np.concatenate((item[6], item[3])) for item in tuples_33], [item[4] for item in tuples_33], [item[1] for item in tuples_33]
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_33, transformations2)])
            for p, a1, a2, transformation2, candidates_for_pattern in zip(patterns_33, analogies_row2_33, analogies_row3_33, transformations2, candidates):
                prediction, actual = complete_analogy_33(network, p, a1, a2, transformation2, candidates_for_pattern)
                
                loss_33 += calculate_error(prediction, actual)

                is_correct_33 = calculate_is_correct(prediction, actual, candidates_for_pattern)
                if is_correct_33:
                    num_correct_33 += 1

            data['3by3'].append(num_correct_33)
            data['3by3_loss'].append(loss_33)
            print(f'3x3        = {color_on(Fore.GREEN, num_correct_33 == max(data["3by3"]))}{num_correct_33:>5}{color_off()}/{100:>5}')
            print(f'    Loss   = {color_on(Fore.RED, loss_33 == min(data["3by3_loss"]))}{loss_33:>11.3f}{color_off()}')        

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


def complete_analogy_22(network, p, a):
    t = target(p) 

    # Prime the network, that is, present object p and output t.
    # Do not present any transformation. Set the transformation to rest.
    # Clamp input and output. Do not clamp transformation.
    # Let the network settle.
    network.calculate_transformation(p, t)

    # Now calculate the response of the primed network for new input a.
    # Clamp input only. Set output to rest.
    # (Leech paper says to set transformation to rest too.)
    # Let the network settle.
    actual = target(a) 
    prediction = network.calculate_response(a, is_primed = True)
    return prediction, actual


def complete_analogy_23(network, p1, a1, tf):
    prediction_a2, actual_a2 = complete_analogy_22(network, p1, a1)
 
    # Add second transformation to first target to form second input pattern
    p2 = np.concatenate((target(p1), tf))
    p3 = target(p2)

    # Prime again with the exemplars p2 and p3
    network.calculate_transformation(p2, p3)

    # Calculate second primed response - this is the prediction
    prediction_a3 = network.calculate_response(np.concatenate((prediction_a2, tf)), is_primed = True)

    # calculate actual values of a3
    actual_a3 = target(np.concatenate((actual_a2, tf)))
    return prediction_a3, actual_a3


def complete_analogy_33(network, p, a1, a2, transformation2, candidates_for_pattern):
    prediction1, actual = complete_analogy_23(network, p, a2, transformation2)
    prediction2, actual = complete_analogy_23(network, a1, a2, transformation2)

    # find the closest candidate if row 1 and row 3 are treated as a 2x3 
    closest13 = closest_node(prediction1, candidates_for_pattern)
    # find the closest candidate if row 2 and row 3 are treated as a 2x3
    closest23 = closest_node(prediction2, candidates_for_pattern)

    # prediction is the one with the minimum distance from a candidate
    if mean_squared_error(prediction1, closest13) < mean_squared_error(prediction2, closest23):
        prediction = closest13
    else:
        prediction = closest23
    return prediction, actual


def setup_plots(n_sample_size: int):
    fig1 = plt.figure(figsize=(10, 7))
    fig1.dpi=100

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
    ax2.set_ylim(0, n_sample_size)

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
    fig1 = Plots.fig1
    ax1 = Plots.ax1
    ax2 = Plots.ax2
    ax3 = Plots.ax3

    color = 'tab:red'
    ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
    ax1.plot(E, color=color)
    ax1.plot(data['o_error'], linestyle=':', linewidth=0.5, color=color)
    ax1.plot(data['t_error'], linestyle='-.', linewidth=0.5, color=color)

    color = 'tab:blue'
    ax2.plot(P, color=color, label='Training')

    color = 'tab:gray'
    ax2.plot(data['tf'], color=color, label='Transformations')

    color = 'tab:green'
    if np.any(A):
        ax2.plot(A, linestyle='-', color=color, label='Analogies')

    ax3.axis([0, len(E) + 10, 0, 100])
    # color = 'tab:blue'
    # if np.any(data['by0']):
    #     ax3.plot(data['by0'], linestyle='-', linewidth=1, color=color, label='0 mods')
    # if np.any(data['by1']):
    #     ax3.plot(data['by1'], linestyle='-.', linewidth=1, color=color, label='1 mod')
    # if np.any(data['by2']):
    #     ax3.plot(data['by2'], linestyle=(0, (1, 1)), linewidth=1, color=color, label='2 mods')
    # if np.any(data['by3']):
    #     ax3.plot(data['by3'], linestyle=':', linewidth=1, color=color, label='3 mods')
    # color = 'tab:green'
    # if np.any(data['aby0']):
    #     ax3.plot(data['aby0'], linestyle='-', linewidth=1, color=color, label='0 mods')
    # if np.any(data['aby1']):
    #     ax3.plot(data['aby1'], linestyle='-.', linewidth=1, color=color, label='1 mod')
    # if np.any(data['aby2']):
    #     ax3.plot(data['aby2'], linestyle=(0, (1, 1)), linewidth=1, color=color, label='2 mods')
    # if np.any(data['aby3']):
    #     ax3.plot(data['aby3'], linestyle=':', linewidth=1, color=color, label='3 mods')
 
    color = 'tab:green'
    if np.any(data['2by2']):
        ax3.plot(data['2by2'], linestyle='-', color=color, label='2x2')

    color = 'tab:orange'
    if np.any(data['2by3']):
        ax3.plot(data['2by3'], linestyle='-', color=color, label='3x2')

    color = 'tab:purple'
    if np.any(data['3by3']):
        ax3.plot(data['3by3'], linestyle='-', color=color, label='3x3')

    ticks = ax3.get_xticks().astype('int') * statistics_frequency
    ax3.set_xticklabels(ticks)

    fig1.canvas.draw()
    fig1.canvas.flush_events()

    plt.pause(0.001)
    if not dynamic:
        plt.ioff()
        plt.show()


def get_checkpoints_folder(config: Config):
    if os.path.exists(f'../storage'): # hack for detecting Paperspace Gradient
        checkpoints_folder = f'../storage/{config.experiment_name}'
    else:
        checkpoints_folder = 'checkpoints'
    return checkpoints_folder


#%% [markdown]
#  ### Test of CHL
# 
#  Here is a simple test of (asynchronous) CHL:

def run(config: Config = None, continue_last = False):
    np.random.seed(0)

    # The patterns to learn
    n_sample_size = 1000

    lexicon = Lexicon()

    i = 0
    tuples = []
    keys = []
    while i < n_sample_size + 100:
        tuple1 = generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=[0,1,2,3])
        key = tuple(np.concatenate((tuple1[2], tuple1[3])))
        if not key in keys:
            keys.append(key)
            tuples.append(tuple1)
            i += 1
    assert len(tuples) == n_sample_size + 100

    #tuples = [generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=[0,1,2,3]) for x in range(1 * n_sample_size)]

    # Last 100 are new analogies for training
    tuples_22 = tuples[-100:]
 
    # training data
    tuples = tuples[:n_sample_size]

    tuples_23 = [generate_rpm_2_by_3_matrix(lexicon) for x in range(1 * 100)]
 
    tuples_33 = [generate_rpm_3_by_3_matrix(lexicon) for x in range(1 * 100)]
 
    #patterns are the training set
    #analogies are the test set
    patterns, analogies = [np.concatenate((item[2], item[3])) for item in tuples], [np.concatenate((item[4], item[3])) for item in tuples]
    # matrices = [item[0] for item in tuples]
    candidates = [item[1] for item in tuples]
    patterns_array = np.asarray(patterns)
    analogies_array = np.asarray(analogies)

    test_data = {
        "analogies" : analogies_array,
        "tuples_22" : tuples_22,
        "tuples_23" : tuples_23,
        "tuples_33" : tuples_33
    }

    checkpoint = None
    checkpoints_folder = get_checkpoints_folder(config)
    if continue_last:
        files = sorted(glob.glob(f'{checkpoints_folder}/*'), reverse=True)
        if not files:
            raise "Could not find any checkpoints to continue from."
        else:
            last_checkpoint = files[0]    
            with open(last_checkpoint, 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
                checkpoint = pickle.load(f)            
                network = checkpoint['network']
    else:
        if not os.path.exists(f'{checkpoints_folder}'):
            os.makedirs(f'{checkpoints_folder}')    
        # remove all existing checkpoints
        files = glob.glob(f'{checkpoints_folder}/*')
        for f in files:
            os.remove(f)

        network = Network(config, training_data=patterns_array, test_data=test_data, candidates=candidates, desired_response_function=target, collect_statistics_function=collect_statistics)

    #%%
    # Plot the Error by epoch

    Plots.fig1, Plots.ax1, Plots.ax2, Plots.ax3 = setup_plots(n_sample_size)

    start = time.time()
    E, P, A, epoch, data = network.asynchronous_chl(checkpoint=checkpoint)
    end = time.time()

    print()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end-start))
    print(f'Elapsed time {time_elapsed} seconds')

    if E[-1] < network.config.min_error * np.size(patterns, 0):
        print(f'Convergeance reached after {epoch} epochs.')
    else:
        print(f'Failed to converge after {epoch} epochs.')
            
    print(f'Final error = {E[-1]}.')
    print('')

    num_matrices = 4
    i = 0
    # output first 4 incorrect 2x2 analogies
    matrices_22, patterns_22, analogies_22, candidates_22 = tuples_22_to_rpm(network.tuples_22)
    for m, p, a, c in zip(matrices_22, patterns_22, analogies_22, candidates_22):
        r, t = complete_analogy_22(network, p, a)
        error = calculate_error(r, t)
        selected = closest_node_index(r, c)
        is_correct = calculate_is_correct(r, t, c)
        if is_correct:
            continue
        i += 1
        if i >= num_matrices:
            break    
        test_matrix(m[0], m[1], selected=selected, is_correct=is_correct)
        print(f'Analogy    = {np.round(a, 2)}')
        print(f'Actual     = {np.round(target(a), 2)}')
        print(f'Prediction = {np.round(network.calculate_response(a), 2)}')
        print(f'Error      = {error:.3f}')
        print(f'Selected   = {selected}')
        print(f'Correct    = {is_correct}')
        print('')


    i = 0
    # output first 4 incorrect 2x3 analogies
    matrices_23, patterns_23, analogies_23, transformations_23, candidates_23 = tuples_23_to_rpm(network.tuples_23)
    for m, p, a, tf, c in zip(matrices_23, patterns_23, analogies_23, transformations_23, candidates_23):
        r, t = complete_analogy_23(network, p, a, tf)
        error = calculate_error(r, t)
        selected = closest_node_index(r, c)
        is_correct = calculate_is_correct(r, t, c)
        if is_correct:
            continue
        i += 1
        if i >= num_matrices:
            break    
        test_matrix(m[0], m[1], selected=selected, is_correct=is_correct)
        print(f'Analogy    = {np.round(a, 2)}')
        print(f'Actual     = {np.round(target(a), 2)}')
        print(f'Prediction = {np.round(network.calculate_response(a), 2)}')
        print(f'Error      = {error:.3f}')
        print(f'Selected   = {selected}')
        print(f'Correct    = {is_correct}')
        print('')


    i = 0
    # output first 4 incorrect 3x3 analogies
    matrices_33, patterns_33, analogies_row2_33, analogies_row3_33, transformations2, candidates_33 = tuples_33_to_rpm(network.tuples_33)
    for m, p, a1, a2, tf, c in zip(matrices_33, patterns_33, analogies_row2_33, analogies_row3_33, transformations2, candidates_33):
        r, t = complete_analogy_33(network, p, a1, a2, tf, c)
        error = calculate_error(r, t)
        selected = closest_node_index(r, c)
        is_correct = calculate_is_correct(r, t, c)
        if is_correct:
            continue
        i += 1
        if i >= num_matrices:
            break    
        test_matrix(m[0], m[1], selected=selected, is_correct=is_correct)
        print(f'Analogy    = {np.round(a, 2)}')
        print(f'Actual     = {np.round(target(a), 2)}')
        print(f'Prediction = {np.round(network.calculate_response(a), 2)}')
        print(f'Error      = {error:.3f}')
        print(f'Selected   = {selected}')
        print(f'Correct    = {is_correct}')
        print('')


    update_plots(E, P, A, data, dynamic=False)

def tuples_22_to_rpm(tuples_22: tuple):
    return [item[0] for item in tuples_22], [np.concatenate((item[2], item[3])) for item in tuples_22], [np.concatenate((item[4], item[3])) for item in tuples_22], [item[1] for item in tuples_22]

def tuples_23_to_rpm(tuples_23: tuple):
    return [item[0] for item in tuples_23], [np.concatenate((item[2], item[3])) for item in tuples_23], [np.concatenate((item[5], item[3])) for item in tuples_23], [item[4] for item in tuples_23], [item[1] for item in tuples_23]

def tuples_33_to_rpm(tuples_33: tuple):
    return [item[0] for item in tuples_33], [np.concatenate((item[2], item[3])) for item in tuples_33], [np.concatenate((item[5], item[3])) for item in tuples_33], [np.concatenate((item[6], item[3])) for item in tuples_33], [item[4] for item in tuples_33], [item[1] for item in tuples_33]

#%%
run(Config(), continue_last=True)
