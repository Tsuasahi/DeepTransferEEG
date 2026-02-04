# fix for numpy 1.24+ deprecated aliases
import numpy as np
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.bool = bool

