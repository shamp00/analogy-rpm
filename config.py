from dataclasses import dataclass

@dataclass
class Config:
    experiment_name = 'experiment_1_33'
    experiment_description = """This is experiment 1. Run 33. 50 hidden units. With noise."""

    # Hyperparameters
    network_type: str = ''
    learning_strategy: str = 'async'
    n_sample_size: int = 1000
    n_inputs: int = 11
    n_transformation: int = 4
    n_hidden: int = 50
    n_outputs: int = 11 
    min_error: float = 0.001
    max_epochs: int = 40000
    max_activation_cycles: int = 100 # The maximum number of times the activation is propagated. 
    max_activation_cycles_fully_unclamped: int = 100
    eta: float = 0.05
    sigmoid_smoothing: float = 0.1
    noise: float = 0.01.
    adaptive_bias: bool = True
    strict_leech: bool = True
    clamp_input_only_during_priming: bool = False
    learn_patterns_explicitly: bool = True
    learn_transformations_explicitly: bool = False
    use_voting_for_3_by_3: bool = True
    smolensky_propagation: bool = True
    unlearn_clamp: str = 'input, transformation' # 'input', 'transformation', 'none'


# @dataclass
# class Config:
#     experiment_name = 'experiment_2_47'
#     experiment_description = """This is experiment 2. Run 47. 50 hidden units."""

#     # Hyperparameters
#     network_type: str = ''
#     learning_strategy: str = 'async'
#     n_sample_size: int = 1000
#     n_inputs: int = 11
#     n_transformation: int = 4
#     n_hidden: int = 50
#     n_outputs: int = 11 
#     min_error: float = 0.001
#     max_epochs: int = 40000
#     max_activation_cycles: int = 100 # The maximum number of times the activation is propagated. 
#     max_activation_cycles_fully_unclamped: int = 0
#     eta: float = 0.0003
#     sigmoid_smoothing: float = 1
#     noise: float = 0.
#     adaptive_bias: bool = True
#     strict_leech: bool = True
#     clamp_input_only_during_priming: bool = False
#     learn_patterns_explicitly: bool = True
#     learn_transformations_explicitly: bool = False
#     use_voting_for_3_by_3: bool = True
#     smolensky_propagation: bool = False
#     unlearn_clamp: str = 'none' # 'input', 'transformation', 'none'
