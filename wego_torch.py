from torch import quantization
from torch.nn import Linear

from torch import no_grad, qint8
from torch.nn.functional import softmax


inference = no_grad


class dtypes:
    int8 = qint8