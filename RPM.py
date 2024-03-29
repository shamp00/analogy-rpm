
#%% [markdown]
# Functions which need passing to CHL

import glob
from logger import log_init, log, log_dict
import os
os.environ['NUMBA_DISABLE_JIT'] = "0"

import pickle
import platform
from shutil import rmtree
import time

import matplotlib
import numpy as np
from colorama import Fore, Style, init
from numba import jit, njit

from config import Config
from network2 import Network
from methods import mean_squared_error
from printing import (Lexicon, generate_rpm_2_by_2_matrix,
                      generate_rpm_2_by_3_matrix, generate_rpm_3_by_3_matrix,
                      is_running_from_ipython, is_paperspace, target, test_matrix)

if not is_running_from_ipython():
    if "Darwin" not in platform.platform():
        # Must be run before importing matplotlib.pyplot
        matplotlib.use('agg')

import seaborn as sns
sns.set(font_scale=0.8)
sns.set_style("whitegrid")

import matplotlib.pyplot as plt

if is_paperspace():
    print('{"chart": "Total Loss", "axis": "Epoch"}')
    print('{"chart": "Pattern accuracy", "axis": "Epoch"}')
    print('{"chart": "Transformation accuracy", "axis": "Epoch"}')

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


@njit
def closest_node_index(node: np.array, nodes: np.ndarray) -> int:
    deltas = np.subtract(nodes, node)
    distance = np.sum(deltas ** 2, axis=1)
    return np.argmin(distance)


@njit
def closest_node(node: np.array, nodes: np.ndarray) -> np.array:
    index: int = closest_node_index(node, nodes)
    return nodes[index]


def color_on(color: str, condition: bool) -> str:
    if condition:
        return color
    else:
        return ''


def color_off() -> str:
    return Fore.RESET


def calculate_is_correct(p1, p2, targets):
    closest = closest_node(p1, targets)
    return np.allclose(closest, p2[:len(p1)])


def collect_statistics(network: Network, E: np.ndarray, P: np.ndarray, A: np.ndarray, epoch: int, data: dict):
    """Reporting function collect_statistics(
        E = loss by epoch, 
        P = num training patterns correct
        A = num test patterns [analogies] correct)"""

    if epoch == 0:
        log(f'Experiment: {network.config.experiment_name}')
        log(f'Description: {network.config.experiment_description}')
        log()
        log('Configuration:')
        log_dict(vars(network.config))

    checkpoint_frequency = 500
    plot_frequency = 500
    statistics_frequency = 50 # report every n epochs

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

        # prevent memory leak
        del checkpoint

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
        if not '2by2s' in data:
            data['2by2s'] = []
        if not '2by2s_loss' in data:
            data['2by2s_loss'] = []
        if not '2by2v' in data:
            data['2by2v'] = []
        if not '2by2v_loss' in data:
            data['2by2v_loss'] = []
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
        if not '22by0' in data:
            data['22by0'] = []    
        if not '22by1' in data:
            data['22by1'] = []    
        if not '22by2' in data:
            data['22by2'] = []    
        if not '22by3' in data:
            data['22by3'] = []   
        if not '22by0s' in data:
            data['22by0s'] = []    
        if not '22by1s' in data:
            data['22by1s'] = []    
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
        num_correct_22_by_num_modifications = [0, 0, 0, 0]
        num_correct_22s_by_size = [0, 0]
        is_max_num_correct_by_num_modifications = [False, False, False, False]
        is_max_num_correct_22_by_num_modifications = [False, False, False, False]
        is_max_num_analogies_correct_by_num_modifications = [False, False, False, False]
        is_min_e_by_num_modifications = [False, False, False, False]
        is_min_e_analogies_by_num_modifications = [False, False, False, False]
        is_max_num_correct_22s_by_size = [False, False]
        e_by_num_modifications = [0., 0., 0., 0.]
        num_analogies_correct = 0
        num_analogies_correct_by_num_modifications = [0, 0, 0, 0]
        e_analogies_by_num_modifications = [0., 0., 0., 0.]
        num_total_patterns_by_num_modifications = [0, 0, 0, 0]
        num_total_22_by_num_modifications = [0, 0, 0, 0]
        num_total_22s_by_size = [0, 0]
        num_transformations_correct = 0
        num_total_transformations_by_type = [0, 0, 0, 0]
        num_correct_by_transformation = [0, 0, 0, 0]
        is_max_num_correct_by_transformation = [False, False, False, False]
        targets = np.asarray([target(p)[:network.n_inputs] for p in network.patterns])
        analogy_targets = np.asarray([target(a)[:network.n_inputs] for a in network.analogies])
        #a_targets = np.asarray([target(a) for a in network.analogies])

        for p, a, c in zip(network.patterns, network.analogies, np.asarray(network.candidates)):
            t = target(p)
            t_error = 0 # the amount of error for the current transformation
            o_error = 0 # the amount of error for the current output object

            process_transformation_error = True
            process_analogy_error = True
            process_2_by_2 = True
            process_2_by_2_bysize = True and hasattr(network, 'tuples_22s')
            process_2_by_3 = True
            process_3_by_3 = True
            process_2_by_2_vertical = False

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
                tf = network.calculate_transformation(p, t)
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
                is_correct = calculate_is_correct(r, t, analogy_targets)
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
        if hasattr(network, 'tuples_22s'):
            tuples_22s = network.tuples_22s
        tuples_23 = network.tuples_23
        tuples_33 = network.tuples_33

        log()
        log(f'Epoch      = {epoch} of {max_epochs}, Loss = {color_on(Fore.RED, e == min(E[1:]))}{e:.3f}{color_off()}, O/T = {color_on(Fore.RED, sum_o_error == min(data["o_error"]))}{sum_o_error:.3f}{color_off()}/{color_on(Fore.RED, sum_t_error == min(data["t_error"]))}{sum_t_error:.3f}{color_off()}, Terminating when < {min_error * len(network.patterns):.3f}')
        log(f'Patterns   = {color_on(Fore.GREEN, num_correct == max(P))}{num_correct:>5}{color_off()}/{len(network.patterns):>5}, breakdown = {" ".join(correct_by_num_modifications)}') 
        log(f'    Loss   = {color_on(Fore.RED, any(data["o_error"]) and sum_o_error == min(data["o_error"]))}{sum_o_error:>11.3f}{color_off()}, breakdown = {" ".join(loss_by_num_modifications)}')        
        log(f'Transforms = {color_on(Fore.GREEN, num_transformations_correct == max(data["tf"]))}{num_transformations_correct:>5}{color_off()}/{len(network.patterns):>5}, breakdown = {" ".join(correct_transformations_by_type)} (sz, rt, sh, no)')
        log(f'    Loss   = {color_on(Fore.RED, any(data["t_error"]) and sum_t_error == min(data["t_error"]))}{sum_t_error:>11.3f}{color_off()}')        
        log(f'Analogies  = {color_on(Fore.GREEN, num_analogies_correct == max(A))}{num_analogies_correct:>5}{color_off()}/{len(network.analogies):>5}, breakdown = {" ".join(analogies_by_num_modifications)}')
        log(f'    Loss   = {color_on(Fore.RED, any(data["a_error"]) and sum_a_error == min(data["a_error"]))}{np.sum(e_analogies_by_num_modifications):>11.3f}{color_off()}, breakdown = {" ".join(loss_analogies_by_num_modifications)}')

        if process_2_by_2:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_22 = 0
            loss_22 = 0            
            patterns_22, analogies_22, candidates_22 = [np.concatenate((item[2], item[3])) for item in network.tuples_22], [np.concatenate((item[4], item[3])) for item in tuples_22], np.asarray([item[1] for item in tuples_22])
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_23, transformations2)])
            for p, a, candidates_for_pattern in zip(patterns_22, analogies_22, candidates_22):
                prediction, actual = complete_analogy_22(network, p, a)

                loss_22 += calculate_error(prediction, actual)
                is_correct_22 = calculate_is_correct(prediction, actual, candidates_for_pattern)
                num_modifications = (p[-4:] != 0.5).sum()  
                num_total_22_by_num_modifications[num_modifications] += 1
                if is_correct_22:
                    num_correct_22 += 1
                    num_correct_22_by_num_modifications[num_modifications] += 1                    

            percentage_breakdown = [100*x[0]/x[1] if x[1] > 0 else 0 for x in zip(num_correct_22_by_num_modifications, num_total_22_by_num_modifications)]
            for i, x in enumerate(percentage_breakdown):
                label = f'22by{i}'
                data[label].append(percentage_breakdown[i])
                is_max_num_correct_22_by_num_modifications[i] = percentage_breakdown[i] > 0.0 and percentage_breakdown[i] == max(data[label])    
            
            correct_22_by_num_modifications = [f'{color_on(Fore.GREEN, x[2])}{x[0]}{color_off()}/{x[1]} {color_on(Fore.GREEN, x[2])}{100*x[0]/x[1] if x[1] > 0 else 0:.1f}%{color_off()}' for x in zip(num_correct_22_by_num_modifications, num_total_22_by_num_modifications, is_max_num_correct_22_by_num_modifications)]
            data['2by2'].append(num_correct_22)
            data['2by2_loss'].append(loss_22)
            log(f'2x2        = {color_on(Fore.GREEN, num_correct_22 == max(data["2by2"]))}{num_correct_22:>5}{color_off()}/{100:>5}, breakdown = {" ".join(correct_22_by_num_modifications)}')
            log(f'    Loss   = {color_on(Fore.RED, loss_22 == min(data["2by2_loss"]))}{loss_22:>11.3f}{color_off()}')        

        if process_2_by_2_bysize:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_22s = 0
            loss_22s = 0            
            patterns_22s, analogies_22s, candidates_22s = [np.concatenate((item[2], item[3])) for item in network.tuples_22s], [np.concatenate((item[4], item[3])) for item in tuples_22s], np.asarray([item[1] for item in tuples_22s])
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_23, transformations2)])
            for p, a, candidates_for_pattern in zip(patterns_22s, analogies_22s, candidates_22s):
                prediction, actual = complete_analogy_22(network, p, a)

                loss_22s += calculate_error(prediction, actual)
                is_correct_22s = calculate_is_correct(prediction, actual, candidates_for_pattern)
                size_of_modification = int(abs((p[-4:]).sum() - 2.0) < 0.25)
                num_total_22s_by_size[size_of_modification] += 1
                if is_correct_22s:
                    num_correct_22s += 1
                    num_correct_22s_by_size[size_of_modification] += 1                    

            percentage_breakdown = [100*x[0]/x[1] if x[1] > 0 else 0 for x in zip(num_correct_22s_by_size, num_total_22s_by_size)]
            for i, x in enumerate(percentage_breakdown):
                label = f'22by{i}s'
                data[label].append(percentage_breakdown[i])
                is_max_num_correct_22s_by_size[i] = percentage_breakdown[i] > 0.0 and percentage_breakdown[i] == max(data[label])    
            
            correct_22s_by_num_modifications = [f'{color_on(Fore.GREEN, x[2])}{x[0]}{color_off()}/{x[1]} {color_on(Fore.GREEN, x[2])}{100*x[0]/x[1] if x[1] > 0 else 0:.1f}%{color_off()}' for x in zip(num_correct_22s_by_size, num_total_22s_by_size, is_max_num_correct_22s_by_size)]
            data['2by2s'].append(num_correct_22s)
            data['2by2s_loss'].append(loss_22s)
            log(f'2x2s       = {color_on(Fore.GREEN, num_correct_22s == max(data["2by2s"]))}{num_correct_22s:>5}{color_off()}/{100:>5}, breakdown = {" ".join(correct_22s_by_num_modifications)} (small, large)')
            log(f'    Loss   = {color_on(Fore.RED, loss_22s == min(data["2by2s_loss"]))}{loss_22s:>11.3f}{color_off()}')        


        if process_2_by_2_vertical:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_22v = 0
            loss_22v = 0            
            patterns_22v, analogies_22v, candidates_22v = [np.concatenate((item[2], item[3])) for item in network.tuples_22], [np.concatenate((item[4], item[3])) for item in tuples_22], np.asarray([item[1] for item in tuples_22])
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_23, transformations2)])
            for p, a, candidates_for_pattern in zip(patterns_22, analogies_22, candidates_22):
                prediction, actual = complete_vertical_analogy_22(network, p, a)

                loss_22v += calculate_error(prediction, actual)
                is_correct_22v = calculate_is_correct(prediction, actual, candidates_for_pattern)
                if is_correct_22v:
                    num_correct_22v += 1

            data['2by2v'].append(num_correct_22v)
            data['2by2v_loss'].append(loss_22v)
            log(f'2x2v       = {color_on(Fore.GREEN, num_correct_22v == max(data["2by2v"]))}{num_correct_22v:>5}{color_off()}/{100:>5}')
            log(f'    Loss   = {color_on(Fore.RED, loss_22v == min(data["2by2v_loss"]))}{loss_22v:>11.3f}{color_off()}')

        if process_2_by_3:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_23 = 0
            loss_23 = 0
            patterns_23, analogies_23, transformations2, candidates = [np.concatenate((item[2], item[3])) for item in tuples_23], [np.concatenate((item[5], item[3])) for item in tuples_23], [item[4] for item in tuples_23], np.asarray([item[1] for item in tuples_23])
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_23, transformations2)])
            for p, a, transformation2, candidates_for_pattern in zip(patterns_23, analogies_23, transformations2, candidates):
                prediction, actual = complete_analogy_23(network, p, a, transformation2, candidates_for_pattern)

                loss_23 += calculate_error(prediction, actual)
                is_correct_23 = calculate_is_correct(prediction, actual, candidates_for_pattern)
                if is_correct_23:
                    num_correct_23 += 1

            data['2by3'].append(num_correct_23)
            data['2by3_loss'].append(loss_23)
            log(f'2x3        = {color_on(Fore.GREEN, num_correct_23 == max(data["2by3"]))}{num_correct_23:>5}{color_off()}/{100:>5}')
            log(f'    Loss   = {color_on(Fore.RED, loss_23 == min(data["2by3_loss"]))}{loss_23:>11.3f}{color_off()}')        

        if process_3_by_3:
            #matrix, test, transformation1, transformation2, analogy
            num_correct_33 = 0
            loss_33 = 0
            patterns_33, analogies_row2_33, analogies_row3_33, transformations2, candidates = [np.concatenate((item[2], item[3])) for item in tuples_33], [np.concatenate((item[5], item[3])) for item in tuples_33], [np.concatenate((item[6], item[3])) for item in tuples_33], [item[4] for item in tuples_33], np.asarray([item[1] for item in tuples_33])
            #targets_2_by_3 = np.asarray([target(np.concatenate([target(a), t2])) for a, t2 in zip(analogies_33, transformations2)])
            for p, a1, a2, transformation2, candidates_for_pattern in zip(patterns_33, analogies_row2_33, analogies_row3_33, transformations2, candidates):
                prediction, actual = complete_analogy_33(network, p, a1, a2, transformation2, candidates_for_pattern)
                
                loss_33 += calculate_error(prediction, actual)

                is_correct_33 = calculate_is_correct(prediction, actual, candidates_for_pattern)
                if is_correct_33:
                    num_correct_33 += 1

            data['3by3'].append(num_correct_33)
            data['3by3_loss'].append(loss_33)
            log(f'3x3        = {color_on(Fore.GREEN, num_correct_33 == max(data["3by3"]))}{num_correct_33:>5}{color_off()}/{100:>5}')
            log(f'    Loss   = {color_on(Fore.RED, loss_33 == min(data["3by3_loss"]))}{loss_33:>11.3f}{color_off()}')        

        end = time.time()
        if epoch == 0:
            end = network.start_time
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.time_since_statistics))
        total_time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end - network.start_time))
        time_per_epoch = f'{1000 * (end - network.time_since_statistics) / statistics_frequency:.3f}'
        network.time_since_statistics = time.time()
        log(f'Elapsed time = {time_elapsed}s, Average time per epoch = {time_per_epoch}ms')
        log(f'Total elapsed time = {total_time_elapsed}s')

        update_metrics(epoch, e, num_correct, num_transformations_correct)

        # reduce frequency of plotting
        if epoch <= 500 or epoch % plot_frequency == 0 or not is_paperspace():
            update_plots(E[1:], P[1:], A[1:], data, dynamic=True, statistics_frequency=statistics_frequency, config=network.config)


def complete_vertical_analogy_22(network, p, a):
    t = target(p) 

    # Prime the network, this time with the first *column* of the matrix.
    # That is, present object p and output a.
    # Do not present any transformation. Set the transformation to rest.
    # Clamp input and output. Do not clamp transformation.
    # Let the network settle.
    network.calculate_transformation(p, a)

    # Now calculate the response of the primed network for new input t.
    # That is, the top right cell of the matrix.
    # Clamp input only. Set output to rest.
    # (Leech paper says to set transformation to rest too.)
    # Let the network settle.
    prediction = network.calculate_response(t, is_primed = True)
    actual = target(a)[:network.n_outputs]

    # (The shape is very often wrong, because we've never trained the network on shape change
    # Here we temporarily hardwire the correct shape for the prediction. This gets us about 72%
    # success when horizontal completion is at 75%.)
    #prediction[0:6] = actual[0:6]

    return prediction, actual


def complete_analogy(network, p1, p2, a1):
    # Ensure none fo the patterns include transformations
    p1 = p1[:network.n_outputs]
    p2 = p2[:network.n_outputs]
    a1 = a1[:network.n_outputs]

    # Prime the network, that is, present object p and output t.
    # Do not present any transformation. Set the transformation to rest.
    # Clamp input and output. Do not clamp transformation.
    # Let the network settle.
    network.calculate_transformation(p1, p2)

    # Now calculate the response of the primed network for new input a.
    # Clamp input only. Set output to rest.
    # (Leech paper says to set transformation to rest instead of output.)
    # Let the network settle.
    prediction = network.calculate_response(a1, is_primed = True)[:network.n_outputs]
    return prediction


def complete_analogy_22(network, p1, a1):
    p2 = target(p1)

    prediction = complete_analogy(network, p1, p2, a1)
    actual = target(a1)[:network.n_outputs]

    return prediction, actual


def complete_analogy_23(network, p1, a1, tf, candidates_for_pattern):
    p2 = np.concatenate((target(p1)[:network.n_outputs], tf))
    p3 = target(p2)
    a2 = np.concatenate((target(a1)[:network.n_outputs], tf))

    # First prediction is from considering p1, p3, a1, a3 as a 2x2 matrix
    prediction1 = complete_analogy(network, p1, p3, a1)

    # Second prediction is from considering p2, p3, a2, a3 as a 2x2 matrix
    prediction2 = complete_analogy(network, p2, p3, a2)
 
    # find the closest candidate if row 1 and row 3 are treated as a 2x3 
    closest13 = closest_node(prediction1, candidates_for_pattern)
    # find the closest candidate if row 2 and row 3 are treated as a 2x3
    closest23 = closest_node(prediction2, candidates_for_pattern)

    # prediction is the one with the minimum distance from a candidate
    if mean_squared_error(prediction1, closest13) < mean_squared_error(prediction2, closest23):
        prediction = prediction1
    else:
        prediction = prediction2

    # calculate actual values of a3
    actual = target(a2)[:network.n_outputs]

    return prediction, actual


def complete_analogy_33(network, p, a1, a2, transformation2, candidates_for_pattern):
    if network.config.use_voting_for_3_by_3:
        return complete_analogy_33_by_voting(network, p, a1, a2, transformation2, candidates_for_pattern)
    #else
    prediction1, actual = complete_analogy_23(network, p, a2, transformation2, candidates_for_pattern)
    prediction2, actual = complete_analogy_23(network, a1, a2, transformation2, candidates_for_pattern)

    # find the closest candidate if row 1 and row 3 are treated as a 2x3 
    closest13 = closest_node(prediction1, candidates_for_pattern)
    # find the closest candidate if row 2 and row 3 are treated as a 2x3
    closest23 = closest_node(prediction2, candidates_for_pattern)

    # prediction is the one with the minimum distance from a candidate
    if mean_squared_error(prediction1, closest13) < mean_squared_error(prediction2, closest23):
        prediction = prediction1
    else:
        prediction = prediction2
    return prediction, actual


def complete_analogy_33_by_voting(network, p1, a1, b1, tf, candidates_for_pattern):
    p2 = np.concatenate((target(p1)[:network.n_outputs], tf))
    p3 = target(p2)
    a2 = np.concatenate((target(a1)[:network.n_outputs], tf))
    a3 = target(a2)
    b2 = np.concatenate((target(b1)[:network.n_outputs], tf))
    actual = target(b2)

    # prediction assuming no distribution
    prediction_none, concurrences_none, loss_none = split_into_four_2_by_2_matrices(network, p1, p2, p3, a1, a2, a3, b1, b2, candidates_for_pattern)
    # prediction assuming left distribution of 3
    prediction_left, concurrences_left, loss_left = split_into_four_2_by_2_matrices(network, p3, p1, p2, a2, a3, a1, b1, b2, candidates_for_pattern)
    # prediction assuming right distribution of 3
    prediction_right, concurrences_right, loss_right = split_into_four_2_by_2_matrices(network, p2, p3, p1, a3, a1, a2, b1, b2, candidates_for_pattern)

    predictions = [prediction_none, prediction_left, prediction_right]
    concurrences = [concurrences_none, concurrences_left, concurrences_right]
    losses = [loss_none, loss_left, loss_right]

    max_concurrences = np.where(concurrences == np.amax(concurrences))[0]
    index_of_min_loss_of_max_concurrences = np.argmin([l for i, l in enumerate(losses) if i in max_concurrences])
    prediction = predictions[index_of_min_loss_of_max_concurrences]
    return prediction, actual


def split_into_four_2_by_2_matrices(network, p1, p2, p3, a1, a2, a3, b1, b2, candidates_for_pattern):
    # First prediction is from considering p1, p3, b1, b3 as a 2x2 matrix
    prediction1 = complete_analogy(network, p1, p3, b1)

    # Second prediction is from considering p2, p3, b2, b3 as a 2x2 matrix
    prediction2 = complete_analogy(network, p2, p3, b2)

    # Third prediction is from considering a1, a3, b1, b3 as a 2x2 matrix
    prediction3 = complete_analogy(network, a1, p3, b1)

    # Fourth prediction is from considering a2, a3, b2, b3 as a 2x2 matrix
    prediction4 = complete_analogy(network, a2, a3, b2)

    # find the closest candidates for each prediction 
    predictions = [prediction1, prediction2, prediction3, prediction4]
    closests = [closest_node_index(p, candidates_for_pattern) for p in predictions]

    counts = np.bincount(closests)
    max_counts = np.where(counts == np.amax(counts))[0]
    max_count_indexes = [i for i, c in enumerate(closests) if c in max_counts]

    candidate_closests = [candidates_for_pattern[i] for i in max_count_indexes]
    candidate_predictions = [predictions[i] for i in max_count_indexes]

    mses = [mean_squared_error(p, c) for p, c in zip(candidate_predictions, candidate_closests)]
    loss = min(mses)
    concurrences = np.amax(counts)
    prediction = candidate_predictions[np.argmin(mses)]
    return prediction, concurrences, loss


def update_metrics(epoch, e, num_correct, num_transformations_correct):
    if is_paperspace():
        print(f'{{"chart": "Total Loss", "x": {epoch}, "y": {e} }}')
        print(f'{{"chart": "Pattern accuracy", "x": {epoch}, "y": {num_correct} }}')
        print(f'{{"chart": "Transformation accuracy", "x": {epoch}, "y": {num_transformations_correct} }}')


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
    ax2.grid(False)

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


def update_plots(E, P, A, data, dynamic=False, statistics_frequency=50, config: Config=None):
    fig1 = Plots.fig1
    ax1 = Plots.ax1
    ax2 = Plots.ax2
    ax3 = Plots.ax3

    color = 'tab:red'
    ax1.clear()
    ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.plot(E, color=color)
    ax1.plot(data['o_error'], linestyle=':', linewidth=0.5, color=color)
    ax1.plot(data['t_error'], linestyle='-.', linewidth=0.5, color=color)

    color = 'tab:blue'
    ax2.clear()
    ax2.set_ylim(0, config.n_sample_size)
    ax2.grid(False)

    ax2.plot(P, color=color, label='Training')

    color = 'tab:gray'
    ax2.plot(data['tf'], color=color, label='Transformations')

    color = 'tab:green'
    if np.any(A):
        ax2.plot(A, linestyle='-', color=color, label='Analogies')

    ax3.clear()
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
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure1.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure1.png')

    if is_paperspace():
        # copy for paperspace monitoring without separate notebook
        fig1.savefig(f'_figure1.png')


def get_checkpoints_folder(config: Config = None):
    if is_paperspace():
        checkpoints_folder = f'../storage/{config.experiment_name}'
    else:
        checkpoints_folder = 'checkpoints'
    return checkpoints_folder


#%% [markdown]
#  ### Test of CHL
# 
#  Here is a simple test of (asynchronous) CHL:

def run(config: Config=None, continue_last=False, skip_learning=True):
    np.random.seed(0)

    # The patterns to learn
    n_sample_size = config.n_sample_size

    lexicon = Lexicon()

    # Ensure we have an even distribution of 0-, 1-, 2- and 3-relational patterns
    # Ensure there are no duplicates
    i = 0
    tuples = []
    keys = []
    while i < n_sample_size + 100:
        j = i % 4
        tuple1 = generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=[j])
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

    tuples_22s = [generate_rpm_2_by_2_matrix(lexicon, num_modification_choices=[1]) for x in range(1 * 100)]

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
        "tuples_22s" : tuples_22s,
        "tuples_23" : tuples_23,
        "tuples_33" : tuples_33
    }

    checkpoint = None
    checkpoints_folder = get_checkpoints_folder(config)
    if continue_last:
        files = sorted(glob.glob(f'{checkpoints_folder}/*.pickle'), reverse=True)
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
        if is_paperspace():
            # replace checkpoints subfolder with symlink to output folder.
            if os.path.exists('checkpoints'):
                rmtree('checkpoints')
            os.symlink(checkpoints_folder, 'checkpoints', target_is_directory=True)  
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        # remove all existing checkpoints
        files = glob.glob(f'{checkpoints_folder}/*')
        for f in files:
            os.remove(f)
        
        network = Network(config, training_data=patterns_array, test_data=test_data, candidates=candidates, desired_response_function=target, collect_statistics_function=collect_statistics)

    # initialize logging
    log_filename = f'{checkpoints_folder}/output.log'
    log_init(log_filename)

    #%%
    # Plot the Error by epoch

    Plots.fig1, Plots.ax1, Plots.ax2, Plots.ax3 = setup_plots(n_sample_size)

    start = time.time()
    if network.config.learning_strategy == 'sync':
        E, P, A, epoch, data = network.synchronous_chl(checkpoint=checkpoint, skip_learning=skip_learning)
    else:
        E, P, A, epoch, data = network.asynchronous_chl(checkpoint=checkpoint, skip_learning=skip_learning)

    end = time.time()

    log()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(end-start))
    log(f'Elapsed time {time_elapsed} seconds')

    if E[-1] < network.config.min_error * np.size(patterns, 0):
        log(f'Convergeance reached after {epoch} epochs.')
    else:
        log(f'Failed to converge after {epoch} epochs.')
            
    log(f'Final error = {E[-1]}.')
    log('')

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
        test_matrix(m[0], m[1], selected=selected)
        log(f'Analogy    = {np.round(a, 2)}')
        log(f'Actual     = {np.round(t, 2)}')
        log(f'Prediction = {np.round(r, 2)}')
        log(f'Error      = {error:.3f}')
        log(f'Selected   = {selected}')
        log(f'Correct    = {is_correct}')
        log('')


    i = 0
    # output first 4 incorrect 2x3 analogies
    matrices_23, patterns_23, analogies_23, transformations_23, candidates_23 = tuples_23_to_rpm(network.tuples_23)
    for m, p, a, tf, c in zip(matrices_23, patterns_23, analogies_23, transformations_23, candidates_23):
        r, t = complete_analogy_23(network, p, a, tf, c)
        error = calculate_error(r, t)
        selected = closest_node_index(r, c)
        is_correct = calculate_is_correct(r, t, c)
        if is_correct:
            continue
        i += 1
        if i >= num_matrices:
            break    
        test_matrix(m[0], m[1], selected=selected)
        log(f'Analogy    = {np.round(a, 2)}')
        log(f'Actual     = {np.round(t, 2)}')
        log(f'Prediction = {np.round(r, 2)}')
        log(f'Error      = {error:.3f}')
        log(f'Selected   = {selected}')
        log(f'Correct    = {is_correct}')
        log('')


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
        test_matrix(m[0], m[1], selected=selected)
        log(f'Analogy    = {np.round(a, 2)}')
        log(f'Actual     = {np.round(t, 2)}')
        log(f'Prediction = {np.round(r, 2)}')
        log(f'Error      = {error:.3f}')
        log(f'Selected   = {selected}')
        log(f'Correct    = {is_correct}')
        log('')

    # i = 0
    # num_correct = 0
    # num_matrices = 100
    # matrices_22, patterns_22, analogies_22, candidates_22 = tuples_22_to_rpm(network.tuples_22)
    # for m, p, a, c in zip(matrices_22, patterns_22, analogies_22, candidates_22):
    #     r, t = complete_vertical_analogy_22(network, p, a)
    #     error = calculate_error(r, t)
    #     selected = closest_node_index(r, c)
    #     is_correct = calculate_is_correct(r, t, c)
    #     if is_correct:
    #         num_correct += 1
    #     else:    
    #         continue
    #     i += 1
    #     if i >= num_matrices:
    #         break    
    #     test_matrix(m[0], m[1], selected=selected)
    #     print(f'Analogy    = {np.round(a, 2)}')
    #     print(f'Actual     = {np.round(t, 2)}')
    #     print(f'Prediction = {np.round(r, 2)}')
    #     print(f'Error      = {error:.3f}')
    #     print(f'Selected   = {selected}')
    #     print(f'Correct    = {is_correct}')
    #     print('')
    # print()
    # print(f'Total number of vertical analogies completed = {num_correct}')

    update_plots(E, P, A, data, dynamic=False, config=network.config)

    if not is_paperspace():
        export_figure_2(E, P, A, data, dynamic=False, config=network.config)
        plot_figure2(E, P, A, data, dynamic=False, config=network.config)

def tuples_22_to_rpm(tuples_22: tuple):
    return np.asarray([item[0] for item in tuples_22]), np.asarray([np.concatenate((item[2], item[3])) for item in tuples_22]), np.asarray([np.concatenate((item[4], item[3])) for item in tuples_22]), np.asarray([item[1] for item in tuples_22])

def tuples_23_to_rpm(tuples_23: tuple):
    return np.asarray([item[0] for item in tuples_23]), np.asarray([np.concatenate((item[2], item[3])) for item in tuples_23]), np.asarray([np.concatenate((item[5], item[3])) for item in tuples_23]), np.asarray([item[4] for item in tuples_23]), np.asarray([item[1] for item in tuples_23])

def tuples_33_to_rpm(tuples_33: tuple):
    return np.asarray([item[0] for item in tuples_33]), np.asarray([np.concatenate((item[2], item[3])) for item in tuples_33]), np.asarray([np.concatenate((item[5], item[3])) for item in tuples_33]), np.asarray([np.concatenate((item[6], item[3])) for item in tuples_33]), np.asarray([item[4] for item in tuples_33]), np.asarray([item[1] for item in tuples_33])

#%%

#import cProfile
#cProfile.run('run(Config(), continue_last=False, skip_learning=False)')

def export_figure_2(E, P, A, data, dynamic=False, statistics_frequency=50, config: Config=None):
    import xlwt
    import xlrd
    from xlutils.copy import copy

    filename = 'figure2.xls'
    if not os.path.exists(filename): 
        book = xlwt.Workbook()
        sheet1 = book.add_sheet('sheet1')
    else:
        book = copy(xlrd.open_workbook(filename))
        sheet1 = book.get_sheet(0) 

    col = 0
    epochs = [i * statistics_frequency for i, _ in enumerate(E)]
    sheet1.write(0,col,"Epoch")
    for i,e in enumerate(epochs):
        sheet1.write(i+1,col,e)
    col = col + 1

    accuracy_patterns = [x / 10 for x in P]
    sheet1.write(0,col,"Pattern accuracy")
    for i,e in enumerate(accuracy_patterns):
        sheet1.write(i+1,col,e)
    col = col + 1

    accuracy_transformations = [x / 10 for x in data['tf']]
    sheet1.write(0,col,"Transformation accuracy")
    for i,e in enumerate(accuracy_transformations):
        sheet1.write(i+1,col,e)
    col = col + 1

    accuracy_2by2 = data['2by2']
    sheet1.write(0,col,"2x2 accuracy")
    for i,e in enumerate(accuracy_2by2):
        sheet1.write(i+1,col,e)
    col = col + 1

    accuracy_3by3 = data['3by3']
    sheet1.write(0,col,"3x3 accuracy")
    for i,e in enumerate(accuracy_3by3):
        sheet1.write(i+1,col,e)
    col = col + 1

    loss_patterns = data['o_error']
    sheet1.write(0,col,"Patterns loss")
    for i,e in enumerate(loss_patterns):
        sheet1.write(i+1,col,e)
    col = col + 1

    loss_transformations = data['t_error']
    sheet1.write(0,col,"Transformation loss")
    for i,e in enumerate(loss_transformations):
        sheet1.write(i+1,col,e)
    col = col + 1

    loss_2by2 = data['2by2_loss']
    sheet1.write(0,col,"2x2 loss")
    for i,e in enumerate(loss_2by2):
        sheet1.write(i+1,col,e)
    col = col + 1

    loss_3by3 = data['3by3_loss']
    sheet1.write(0,col,"3x3 loss")
    for i,e in enumerate(loss_3by3):
        sheet1.write(i+1,col,e)
    col = col + 1

    name = "figure2.xls"
    book.save(name)

def annotate_point(x, y, ax=None, text='max', facecolor='black', xytext=(0.35, 0.90)):
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    #arrowprops=dict(facecolor='black', arrowstyle="-|>",connectionstyle="angle,angleA=0,angleB=60")
    arrowprops=dict(facecolor=facecolor, arrowstyle="simple")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props)
    ax.annotate(text, xy=(x, y), xytext=xytext, **kw)


def plot_figure2(E, P, A, data, dynamic=False, statistics_frequency=50, config: Config=None):
    fig1 = plt.figure(figsize=(6, 4))
    plt.dpi=100
 
    plt.title('Experiment 2 - Training Accuracy')
    plt.xlabel('Epoch')
    plt.xlim(0, len(E))
    plt.xticks(range(0, len(E), 40))
    plt.ylim(0, 100)
    plt.ylabel('Percentage correct')
    plt.yticks(range(0, 101, 10))
    plt.plot([x / 10 for x in P], color='blue', label='Output shapes', linestyle='-')
    plt.plot([x / 10 for x in data['tf']], color='orange', label='Transformations', linestyle='-')

    #plt.plot(data['2by2'], label='2x2 validation')
    #plt.plot(data['3by3'], label='3x3 validation')
    
    # Fix x-axis ticks
    ax = plt.axes()
    ticks = ax.get_xticks().astype('int') * statistics_frequency
    ax.set_xticklabels(ticks)
    # Show legend
    plt.legend(loc='lower right', ncol=1)
    plt.show()

    fig1.canvas.draw()
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure2.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure2.png')

    # Now plot loss
    fig1 = plt.figure(figsize=(6, 4))
    plt.dpi=100

    plt.title('Experiment 2 - Training Loss')
    plt.xlabel('Epoch')
    plt.xlim(0, len(E))
    plt.xticks(range(0, len(E), 40))
    plt.ylabel('Loss')
    plt.ylim(0, 20)
    plt.plot(data['o_error'], color='blue', label='Output shapes', linestyle='-')
    plt.plot(data['t_error'], color='orange', label='Transformations', linestyle='-')

#    plt.plot(data['2by2_loss'], label='2x2 validation', linestyle=':')
#    plt.plot(data['3by3_loss'], label='3x3 validation', linestyle='--')

    # Fix x-axis ticks
    ax = plt.axes()
    ticks = ax.get_xticks().astype('int') * statistics_frequency
    ax.set_xticklabels(ticks)
    # Show legend
    plt.legend(loc='upper right', ncol=1)

    plt.show()

    fig1.canvas.draw()
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure3.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure3.png')

    # Now plot test accuracy
    fig1 = plt.figure(figsize=(6, 4))
    plt.dpi=100

    plt.title('Experiment 2 - Accuracy of Analogy Completion by Task')
    plt.xlabel('Epoch')
    plt.xlim(0, len(E))
    plt.xticks(range(0, len(E), 40))
    plt.ylim(0, 100)
    plt.ylabel('Percentage correct')
    plt.yticks(range(0, 101, 10))
    plt.plot(data['2by2'], color='green', label='2x2', linestyle='-')
    plt.plot(data['3by3'], color='purple', label='3x3', linestyle='-')

#    plt.plot(data['2by2_loss'], label='2x2 validation', linestyle=':')
#    plt.plot(data['3by3_loss'], label='3x3 validation', linestyle='--')

    # Fix x-axis ticks
    ax = plt.axes()
    ticks = ax.get_xticks().astype('int') * statistics_frequency
    ax.set_xticklabels(ticks)

    # annotations
    max_x = np.argmax(data['2by2'])
    max_y = max(data['2by2'])
    annotate_point(max_x, max_y, ax=ax, text=f'Max 2x2 = {max_y}', facecolor='black', xytext=(0.03, 0.96))

    max_x = np.argmax(data['3by3'])
    max_y = max(data['3by3'])
    annotate_point(max_x, max_y, ax=ax, text=f'Max 3x3 = {max_y}', facecolor='black', xytext=(0.25, 0.93))

    # Show legend
    plt.legend(loc='lower right', ncol=1)

    plt.show()

    fig1.canvas.draw()
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure4.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure4.png')

    # Now plot accuracy by size
    fig1 = plt.figure(figsize=(6, 4))
    plt.dpi=100

    limited_epochs = 3100 // statistics_frequency

    plt.title('Accuracy by Size of Transformation')
    plt.xlabel('Epoch')
    plt.xlim(0, limited_epochs)
    plt.xticks(range(0, limited_epochs, 20))
    plt.ylim(0, 100)
    plt.ylabel('Percentage correct')
    plt.yticks(range(0, 101, 10))
    plt.plot(data['22by1s'][:limited_epochs], color='purple', label='Large', linestyle='-')
    plt.plot(data['22by0s'][:limited_epochs], color='pink', label='Small', linestyle='-')

#    plt.plot(data['2by2_loss'], label='2x2 validation', linestyle=':')
#    plt.plot(data['3by3_loss'], label='3x3 validation', linestyle='--')

    # Fix x-axis ticks
    ax = plt.axes()
    ticks = ax.get_xticks().astype('int') * statistics_frequency
    ax.set_xticklabels(ticks)
    # Show legend
    plt.legend(loc='lower right', ncol=1)

    plt.show()

    fig1.canvas.draw()
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure5.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure5.png')


    # Now plot accuracy by complexity
    fig1 = plt.figure(figsize=(6, 4))
    plt.dpi=100

    limited_epochs = 3100 // statistics_frequency

    plt.title('Accuracy by Complexity of Transformation')
    plt.xlabel('Epoch')
    plt.xlim(0, limited_epochs)
    plt.xticks(range(0, limited_epochs, 20))
    plt.ylim(0, 100)
    plt.ylabel('Percentage correct')
    plt.yticks(range(0, 101, 10))

    if np.any(data['22by0']):
        plt.plot(data['22by0'], linestyle='-', linewidth=1, color='green', label='0-relational')
    if np.any(data['22by1']):
        plt.plot(data['22by1'], linestyle='-', linewidth=1, color='blue', label='1-relational')
    if np.any(data['22by2']):
        plt.plot(data['22by2'], linestyle='-', linewidth=1, color='red', label='2-relational')
    if np.any(data['22by3']):
        plt.plot(data['22by3'], linestyle='-', linewidth=1, color='orange', label='3-relational')

    # Fix x-axis ticks
    ax = plt.axes()
    ticks = ax.get_xticks().astype('int') * statistics_frequency
    ax.set_xticklabels(ticks)
    # Show legend
    plt.legend(loc='lower right', ncol=1)

    plt.show()

    fig1.canvas.draw()
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure6.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure6.png')


    # fig1 = plt.figure(figsize=(10, 7))
    # fig1.dpi=100
    
    # color = 'tab:red'
    # #ax1.set_title(f'Relational priming for RPMs')
    # #ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Training loss (patterns)')
    # #ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_x_label('Epoch')

    # # ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    # # color = 'tab:green'
    # # #ax3.set_title(f'Breakdown by number of mods')
    # # ax3.set_xlabel('Epoch')
    # # ax3.set_ylabel('Accuracy')
    # # ax3.tick_params(axis='y', labelcolor=color)
    # # ax3.tick_params(axis='x', labelcolor=color)
    
    # color = 'tab:red'
    # ax1.axis([0, len(E) + 10, 0, max(E[3:] + [0.7]) + 0.1])
    # #ax1.plot(E, color=color)
    # ax1.plot(data['o_error'], linestyle='-', linewidth=1.0, color=color)
    # #ax1.plot(data['t_error'], linestyle='-.', linewidth=0.5, color=color)

    # color = 'tab:blue'
    # ax2.set_ylabel('Test loss (patterns)')

    # color = 'tab:gray'
    # ax2.plot(data['tf'], color=color, label='Transformations')

    # color = 'tab:green'
    # if np.any(A):
    #     ax2.plot(A, linestyle='-', color=color, label='Analogies')

    # ax3.clear()
    # ax3.axis([0, len(E) + 10, 0, 100])
    # # color = 'tab:blue'
    # # if np.any(data['by0']):
    # #     ax3.plot(data['by0'], linestyle='-', linewidth=1, color=color, label='0 mods')
    # # if np.any(data['by1']):
    # #     ax3.plot(data['by1'], linestyle='-.', linewidth=1, color=color, label='1 mod')
    # # if np.any(data['by2']):
    # #     ax3.plot(data['by2'], linestyle=(0, (1, 1)), linewidth=1, color=color, label='2 mods')
    # # if np.any(data['by3']):
    # #     ax3.plot(data['by3'], linestyle=':', linewidth=1, color=color, label='3 mods')
    # # color = 'tab:green'
    # # if np.any(data['aby0']):
    # #     ax3.plot(data['aby0'], linestyle='-', linewidth=1, color=color, label='0 mods')
    # # if np.any(data['aby1']):
    # #     ax3.plot(data['aby1'], linestyle='-.', linewidth=1, color=color, label='1 mod')
    # # if np.any(data['aby2']):
    # #     ax3.plot(data['aby2'], linestyle=(0, (1, 1)), linewidth=1, color=color, label='2 mods')
    # # if np.any(data['aby3']):
    # #     ax3.plot(data['aby3'], linestyle=':', linewidth=1, color=color, label='3 mods')
 
    # color = 'tab:green'
    # if np.any(data['2by2']):
    #     ax3.plot(data['2by2'], linestyle='-', color=color, label='2x2')

    # color = 'tab:orange'
    # if np.any(data['2by3']):
    #     ax3.plot(data['2by3'], linestyle='-', color=color, label='3x2')

    # color = 'tab:purple'
    # if np.any(data['3by3']):
    #     ax3.plot(data['3by3'], linestyle='-', color=color, label='3x3')

    # ticks = ax3.get_xticks().astype('int') * statistics_frequency
    # ax3.set_xticklabels(ticks)

    # fig1.canvas.draw()

    # plt.show()
    # fig1.canvas.draw()
    # fig1.savefig(f'{get_checkpoints_folder(config)}/figure2.svg')
    # fig1.savefig(f'{get_checkpoints_folder(config)}/figure2.png')

def load_from_folder(checkpoints_folder):
    files = sorted(glob.glob(f'{checkpoints_folder}/*.pickle'), reverse=True)
    if not files:
        raise "Could not find any checkpoints to continue from."
    else:
        last_checkpoint = files[0]    
        with open(last_checkpoint, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            checkpoint = pickle.load(f)            
            network = checkpoint['network']
            return network.data          
    raise f"Some problem occured loading the checkpoint from the folder {checkpoints_folder}"

def plot():
    config = None
    statistics_frequency = 50

    data50 = load_from_folder('paperspace/experiment_2_55')
    data30 = load_from_folder('paperspace/experiment_2_59')
    data15 = load_from_folder('paperspace/experiment_2_56')
    data0 = load_from_folder('paperspace/experiment_2_57')

    # Now plot test accuracy
    fig1 = plt.figure(figsize=(6, 4))
    plt.dpi=100

    limited_epochs = 10500 // statistics_frequency
    plt.title('Experiment 2 - Comparison of 3x3 Accuracy by Width of Hidden Layer')
    plt.xlabel('Epoch')
    plt.xlim(0, limited_epochs)
    plt.xticks(range(0, limited_epochs, 40))
    plt.ylim(0, 100)
    plt.ylabel('Percentage correct')
    plt.yticks(range(0, 101, 10))
    plt.plot(data50['3by3'][:limited_epochs], color='green', label='50', linestyle='-')
    #plt.plot(data30['3by3'][:limited_epochs], color='red', label='30', linestyle='-')
    plt.plot(data15['3by3'][:limited_epochs], color='blue', label='15', linestyle='-')
    plt.plot(data0['3by3'][:limited_epochs], color='orange', label=' 0', linestyle='-')

#    plt.plot(data['2by2_loss'], label='2x2 validation', linestyle=':')
#    plt.plot(data['3by3_loss'], label='3x3 validation', linestyle='--')

    # Fix x-axis ticks
    ax = plt.axes()
    ticks = ax.get_xticks().astype('int') * statistics_frequency
    ax.set_xticklabels(ticks)

    # # annotations
    # max_x = np.argmax(data['2by2'])
    # max_y = max(data['2by2'])
    # annotate_point(max_x, max_y, ax=ax, text=f'Max 2x2 = {max_y}', facecolor='black', xytext=(0.03, 0.96))

    # max_x = np.argmax(data['3by3'])
    # max_y = max(data['3by3'])
    # annotate_point(max_x, max_y, ax=ax, text=f'Max 3x3 = {max_y}', facecolor='black', xytext=(0.25, 0.93))

    # Show legend
    plt.legend(title='Hidden units', loc='lower right', ncol=1)

    plt.show()

    fig1.canvas.draw()
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure7.svg')
    fig1.savefig(f'{get_checkpoints_folder(config)}/figure7.png')


run(Config(), continue_last=False, skip_learning=False)
#plot()
