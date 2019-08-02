from dataclasses import dataclass

@dataclass
class Config:
    experiment_name = 'experiment_1_15'
    experiment_description = """This is experiment 1. Run 15. 50 Hidden units. Same as before. Quick check that nothing is no worse since recent fixes to analogy completion."""

    # Hyperparameters
    network_type: str = ''
    learning_strategy: str = 'async'
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
    noise: float = 0.
    adaptive_bias: bool = True
    strict_leech: bool = True
    clamp_input_only_during_priming: bool = False
    learn_patterns_explicitly: bool = True
    learn_transformations_explicitly: bool = False
