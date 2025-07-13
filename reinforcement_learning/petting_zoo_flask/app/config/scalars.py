SEED = 0                        # seed for the pseudo-random number generator.
MINIBATCH_SIZE = 128            # mini-batch size.
TAU = 1e-3                      # soft update parameter.
MEMORY_SIZE = 10_000            # size of memory buffer
GAMMA = 0.997                   # discount factor
ALPHA = 1e-3                    # learning rate
HUMAN_AGENT_NAME = 'first_0'    # human agent selected identifier

EPSILON = 1.0             # initial rate of random modle choices
E_DECAY = 0.95            # e-decay rate for the e-greedy policy.
E_GROW = 1.001            # e-grow rate for the e-greedy policy.
E_MIN = 0.01              # minimum ε value for the e-greedy policy.
E_MAX = 1.0               # max ε value for the e-greedy policy.

INPUT_TIMEOUT = 0.01

ATARI_PRO = "./models/keras/atari_pro.keras"
ATARI_PRO_WEIGHTS = "./models/keras/atari_pro_weights.keras"
