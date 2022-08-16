import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List

# neural network regression


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np(-1.0 * x))


def init_weights(input_size: int, hidden_size: int) -> Dict[str, ndarray]:

    """
    initialize weights during forward pass
    """

    weights: Dict[str, ndarray] = {}
    weights["W1"] = np.random.randn(input_size, hidden_size)
    weights["B1"] = np.random.randn(1, hidden_size)
    weights["W2"] = np.random.randn(hidden_size, 1)
    weights["B2"] = np.random.randn(1, 1)
    return weights


def forward_loss(
    X: ndarray, y: ndarray, weights: Dict[str, ndarray]
) -> Tuple[Dict[str, ndarray], float]:

    """
    Compute forward pass and loss
    """

    M1 = np.dot(X, weights["W1"])

    N1 = M1 + weights["B1"]

    O1 = sigmoid(N1)

    M2 = np.dot(O1, weights["W2"])

    P = M2 + weights["B2"]

    loss = np.mean(np.power(y - P, 2))

    forward_info: Dict[str, ndarray] = {}
    forward_info["X"] = X
    forward_info["M1"] = M1
    forward_info["N1"] = N1
    forward_info["O1"] = O1
    forward_info["M2"] = M2
    forward_info["P"] = P
    forward_info["y"] = y

    return forward_info, loss


def loss_gradients(
    forward_info: Dict[str, ndarray], weights: Dict[str, ndarray]
) -> Dict[str, ndarray]:

    """
    compute partial derivatives of the loss wrt each of the parameters
    """
    pass
