from dataclasses import dataclass

# @dataclass
# class Config:
#     experiment_name = 'experiment_1_21'
#     experiment_description = """This is experiment 1. Run 21. 14 hidden units. Smolensky."""

#     # Hyperparameters
#     network_type: str = ''
#     learning_strategy: str = 'async'
#     n_sample_size: int = 1000
#     n_inputs: int = 11
#     n_transformation: int = 4
#     n_hidden: int = 14
#     n_outputs: int = 11 
#     min_error: float = 0.001
#     max_epochs: int = 40000
#     max_activation_cycles: int = 100 # The maximum number of times the activation is propagated. 
#     max_activation_cycles_fully_unclamped: int = 100
#     eta: float = 0.05
#     sigmoid_smoothing: float = 0.1
#     noise: float = 0.
#     adaptive_bias: bool = True
#     strict_leech: bool = True
#     clamp_input_only_during_priming: bool = False
#     learn_patterns_explicitly: bool = True
#     learn_transformations_explicitly: bool = False
#     use_voting_for_3_by_3: bool = False
#     smolensky_propagation: bool = True


@dataclass
class Config:
    experiment_name = 'experiment_2_29'
    experiment_description = """This is experiment 2. Run 29. 0 hidden units. Smolensky propagation. Increased eta 0.003."""

    # Hyperparameters
    network_type: str = ''
    learning_strategy: str = 'async'
    n_sample_size: int = 1000
    n_inputs: int = 11
    n_transformation: int = 4
    n_hidden: int = 0
    n_outputs: int = 11 
    min_error: float = 0.001
    max_epochs: int = 40000
    max_activation_cycles: int = 100 # The maximum number of times the activation is propagated. 
    max_activation_cycles_fully_unclamped: int = 0
    eta: float = 0.003
    sigmoid_smoothing: float = 0.1
    noise: float = 0.
    adaptive_bias: bool = True
    strict_leech: bool = True
    clamp_input_only_during_priming: bool = False
    learn_patterns_explicitly: bool = True
    learn_transformations_explicitly: bool = False
    use_voting_for_3_by_3: bool = False
    smolensky_propagation: bool = True
