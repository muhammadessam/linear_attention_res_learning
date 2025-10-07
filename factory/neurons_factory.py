from spikingjelly.activation_based import neuron, surrogate

SURROGATE_FUNCTIONS: dict = {
    'sigmoid': {
        'callable': surrogate.Sigmoid(),
        'backend': 'cupy'
    },
    'atan': {
        'callable': surrogate.ATan(),
        'backend': 'cupy'
    }
}

# Neuron factory with default configurations
NEURON_CONFIGS: dict = {
    'LIF': {
        'class': neuron.LIFNode,
        'default_params': {
            'tau': 2.0,
            'v_threshold': 0.5,  # Lower threshold for better firing
            'detach_reset': True,
        }
    },
    'PLIF': {
        'class': neuron.ParametricLIFNode,
        'default_params': {
            'init_tau': 2.0,
            'v_threshold': 1.0,
            'detach_reset': True,

        }
    }
}


def create_neuron(neuron_type='LIF', surrogate_type='sigmoid', **custom_params):
    """Create a neuron instance with a given type, surrogate function, and parameters."""

    if neuron_type not in NEURON_CONFIGS:
        raise ValueError(f"Unknown neuron type: {neuron_type}. Available: {list(NEURON_CONFIGS.keys())}")

    if surrogate_type not in SURROGATE_FUNCTIONS:
        raise ValueError(f"Unknown surrogate function: {neuron_type}. Available: {list(SURROGATE_FUNCTIONS.keys())}")

    config = NEURON_CONFIGS[neuron_type]

    params = config['default_params'].copy()
    params['step_mode'] = 'm'
    params['surrogate_function'] = SURROGATE_FUNCTIONS[surrogate_type]['callable']
    params['backend'] = SURROGATE_FUNCTIONS[surrogate_type]['backend']

    params.update(custom_params)

    return config['class'](**params)
