from dataclasses import dataclass

@dataclass
class Config:
    experiment_name = 'experiment_2_18'
    experiment_description = """This is experiment 2. Run 18. 30 Hidden units. Same as 2_17. Ensuring nothing has changed since improvements to matrix generation. Improved analogy completion for 2x3 and 3x3. Also, testing different unlearn scenarios. Here we are using totally unclamped unlearn which seems the smoothest according to experiments 2_14, 2_15 and 2_16. Here we increase eta to 0.0003."""

    # Hyperparameters
    network_type: str = ''
    learning_strategy: str = 'async'
    n_inputs: int = 11
    n_transformation: int = 4
    n_hidden: int = 30
    n_outputs: int = 11 
    min_error: float = 0.001
    max_epochs: int = 40000
    max_activation_cycles: int = 100 # The maximum number of times the activation is propagated. 
    max_activation_cycles_fully_unclamped: int = 0
    eta: float = 0.0003
    sigmoid_smoothing: float = 0.1
    noise: float = 0.
    adaptive_bias: bool = True
    strict_leech: bool = True
    clamp_input_only_during_priming: bool = False
    learn_patterns_explicitly: bool = True
    learn_transformations_explicitly: bool = False
